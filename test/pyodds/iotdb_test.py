import datetime

import numpy as np
import pandas as pd
import pytest

from pyodds.utils.iotdb import (
    connect_iotdb_server,
    insert_iotdb_demo_data,
    query_iotdb_data,
)


class FakeDataSet:
    def __init__(self, frame):
        self.frame = frame
        self.closed = False

    def todf(self):
        return self.frame.copy()

    def close_operation_handle(self):
        self.closed = True


class FakeSession:
    def __init__(self, databases=None, series=None, frames=None):
        self.databases = databases or []
        self.series = series or []
        self.frames = frames or {}
        self.statements = []
        self.created_series = []
        self.storage_groups = []
        self.batch = None
        self.datasets = []

    def execute_query_statement(self, sql):
        self.statements.append(sql)
        if sql == "SHOW DATABASES":
            frame = pd.DataFrame({"Database": self.databases})
        elif sql.startswith("SHOW TIMESERIES"):
            frame = pd.DataFrame({"Timeseries": self.series})
        elif " WHERE " in sql:
            frame = self.frames["filtered"]
        else:
            frame = self.frames["all"]
        result = FakeDataSet(frame)
        self.datasets.append(result)
        return result

    def set_storage_group(self, path):
        self.storage_groups.append(path)

    def create_time_series(self, path, data_type, encoding, compressor):
        self.created_series.append((path, data_type, encoding, compressor))

    def insert_records_of_one_device(self, *args):
        self.batch = args


def test_connect_iotdb_server_uses_configured_tree_session(monkeypatch):
    from iotdb import Session as session_module

    calls = {}

    class SessionStub:
        def __init__(self, *args, **kwargs):
            calls["args"] = args
            calls["kwargs"] = kwargs

        def open(self, compression):
            calls["compression"] = compression

        def close(self):
            calls["closed"] = True

    monkeypatch.setattr(session_module, "Session", SessionStub)
    session = connect_iotdb_server("iotdb.local", "reader", "secret", port=7777)

    assert isinstance(session, SessionStub)
    assert calls["args"] == ("iotdb.local", 7777, "reader", "secret")
    assert calls["kwargs"]["zone_id"] == "UTC"
    assert calls["compression"] is False
    assert not calls.get("closed")


def test_insert_demo_data_creates_missing_tree_schema_and_batches_840_rows():
    from iotdb.utils.IoTDBConstants import Compressor, TSDataType, TSEncoding

    session = FakeSession(
        databases=["root.analytics"],
        series=["root.analytics.sensor.a"],
    )
    np.random.seed(42)

    labels = insert_iotdb_demo_data(
        session,
        "root.analytics",
        "sensor",
        ground_truth_flag=True,
    )

    assert session.storage_groups == []
    assert [item[0] for item in session.created_series] == ["root.analytics.sensor.b"]
    assert session.created_series[0][1:] == (TSDataType.FLOAT, TSEncoding.RLE, Compressor.SNAPPY)
    device, timestamps, measurements, types, values = session.batch
    assert device == "root.analytics.sensor"
    assert len(timestamps) == len(values) == 840
    assert timestamps == sorted(timestamps)
    assert measurements[0] == ["a", "b"]
    assert types[0] == [TSDataType.FLOAT, TSDataType.FLOAT]
    assert len(labels) == 840
    assert np.flatnonzero(labels == -1).tolist() == list(range(400, 420)) + list(range(820, 840))


def test_insert_demo_data_creates_database_and_both_measurements():
    session = FakeSession()

    assert insert_iotdb_demo_data(session, "analytics", "sensor") is None

    assert session.storage_groups == ["root.analytics"]
    assert [item[0] for item in session.created_series] == [
        "root.analytics.sensor.a",
        "root.analytics.sensor.b",
    ]
    assert len(session.batch[1]) == 840


def test_query_data_returns_bounded_tree_rows_and_aligned_ground_truth():
    times = pd.to_datetime(
        ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z"]
    )
    all_rows = pd.DataFrame({"Time": times, "temperature": [1.0, 2.0, 3.0]})
    filtered_rows = all_rows.iloc[1:].reset_index(drop=True)
    session = FakeSession(frames={"all": all_rows, "filtered": filtered_rows})

    data, labels = query_iotdb_data(
        session,
        "analytics",
        "sensor",
        start_time="2024-01-02T00:00:00Z",
        end_time="2024-01-03T00:00:00Z",
        ground_truth=np.array([1, 1, -1]),
        time_serie=True,
        ground_truth_flag=True,
    )

    assert session.statements[0] == (
        "SELECT * FROM root.analytics.sensor WHERE time >= 1704153600000 "
        "AND time <= 1704240000000 ORDER BY time ASC"
    )
    assert len(session.statements) == 2
    assert all(result.closed for result in session.datasets)
    assert list(data.columns) == ["Time", "temperature"]
    assert data["temperature"].tolist() == [2.0, 3.0]
    assert labels.tolist() == [1, -1]


def test_query_data_drops_time_for_static_model_input():
    frame = pd.DataFrame({"Time": [1704067200000], "a": [1.0], "b": [2.0]})
    session = FakeSession(frames={"all": frame})

    data = query_iotdb_data(session, "analytics", "sensor")

    assert list(data.columns) == ["a", "b"]
    assert data.iloc[0].tolist() == [1.0, 2.0]


@pytest.mark.parametrize("database,table", [("root", "sensor"), ("db; DROP DATABASE root.x", "sensor"), ("db", "s.t")])
def test_invalid_path_identifiers_are_rejected_before_query(database, table):
    session = FakeSession()

    with pytest.raises(ValueError):
        query_iotdb_data(session, database, table)

    assert session.statements == []


def test_invalid_time_range_is_rejected():
    session = FakeSession()
    start = datetime.datetime(2024, 1, 2)
    end = datetime.datetime(2024, 1, 1)

    with pytest.raises(ValueError, match="start_time"):
        query_iotdb_data(session, "analytics", "sensor", start_time=start, end_time=end)

    assert session.statements == []


def test_ground_truth_length_must_match_full_query():
    frame = pd.DataFrame({"Time": pd.to_datetime(["2024-01-01T00:00:00Z"]), "a": [1.0]})
    session = FakeSession(frames={"all": frame})

    with pytest.raises(ValueError, match="ground_truth length"):
        query_iotdb_data(
            session,
            "analytics",
            "sensor",
            ground_truth=np.array([1, -1]),
            ground_truth_flag=True,
        )
