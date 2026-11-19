"""Apache IoTDB Tree-model storage helpers for PyODDS."""

import datetime
import re

import numpy as np
import pandas as pd


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _database_path(database):
    """Return a validated IoTDB database path rooted below ``root``."""
    value = str(database).strip()
    if value == "root":
        raise ValueError("database must name a child below root")
    if value.startswith("root."):
        value = value[5:]
    parts = value.split(".")
    if not parts or any(not _IDENTIFIER.fullmatch(part) for part in parts):
        raise ValueError("database must contain only simple IoTDB path identifiers")
    return "root." + ".".join(parts)


def _device_path(database, table):
    table = str(table).strip()
    if not _IDENTIFIER.fullmatch(table):
        raise ValueError("table must be a simple IoTDB device identifier")
    return _database_path(database) + "." + table


def _time_bound(value):
    if value is None or value == "":
        return None
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("time bounds must be valid date/time values") from exc
    if pd.isna(parsed):
        raise ValueError("time bounds must be valid date/time values")
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    return parsed.value // 1_000_000


def _query_frame(session, sql):
    data_set = session.execute_query_statement(sql)
    if data_set is None:
        return pd.DataFrame()
    try:
        return data_set.todf()
    finally:
        close = getattr(data_set, "close_operation_handle", None)
        if close is None:
            close = getattr(data_set, "close", None)
        if close is not None:
            close()


def _time_column(frame):
    for column in frame.columns:
        if str(column).lower() == "time":
            return column
    raise ValueError("IoTDB query did not return the Tree model Time column")


def _as_utc_times(values):
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_datetime(values, unit="ms", utc=True)
    return pd.to_datetime(values, utc=True)


def connect_iotdb_server(host, user, password, port=6667, fetch_size=5000):
    """Open and return an Apache IoTDB 2.x Session using UTC timestamps."""
    if not host or not user or password is None:
        raise ValueError("host, user, and password are required")
    port = int(port)
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    from iotdb.Session import Session

    session = Session(
        host,
        port,
        user,
        password,
        fetch_size=int(fetch_size),
        zone_id="UTC",
        connection_timeout_in_ms=10000,
    )
    try:
        session.open(False)
    except Exception:
        session.close()
        raise
    return session


def insert_iotdb_demo_data(session, database, table, ground_truth_flag=False):
    """Create the demo Tree schema and batch-write the same 840 points as TDengine."""
    from iotdb.utils.IoTDBConstants import Compressor, TSDataType, TSEncoding

    device = _device_path(database, table)
    database_path = _database_path(database)

    databases = _query_frame(session, "SHOW DATABASES")
    database_column = next(
        (column for column in databases.columns if str(column).lower() == "database"),
        None,
    )
    known_databases = set()
    if database_column is not None:
        known_databases = {str(value) for value in databases[database_column].dropna()}
    if database_path not in known_databases:
        session.set_storage_group(database_path)

    existing = _query_frame(session, "SHOW TIMESERIES {}.*".format(device))
    series_column = next(
        (column for column in existing.columns if str(column).lower() == "timeseries"),
        None,
    )
    known_series = set()
    if series_column is not None:
        known_series = {str(value) for value in existing[series_column].dropna()}
    for measurement in ("a", "b"):
        series = device + "." + measurement
        if series not in known_series:
            session.create_time_series(
                series,
                TSDataType.FLOAT,
                TSEncoding.RLE,
                Compressor.SNAPPY,
            )

    timestamps = []
    values = []
    start_time = datetime.datetime(2019, 8, 1, tzinfo=datetime.timezone.utc)
    interval = datetime.timedelta(seconds=60)
    for _ in range(200):
        timestamps.append(int(start_time.timestamp() * 1000))
        values.append([float(0.3 * np.random.randn() - 2), float(0.3 * np.random.randn() - 2)])
        start_time += interval
    for _ in range(200):
        timestamps.append(int(start_time.timestamp() * 1000))
        values.append([float(0.3 * np.random.randn() + 2), float(0.3 * np.random.randn() + 2)])
        start_time += interval
    for _ in range(20):
        timestamps.append(int(start_time.timestamp() * 1000))
        values.append([float(np.random.uniform(-4, 4)), float(np.random.uniform(-4, 4))])
        start_time += interval

    start_time = datetime.datetime(2019, 9, 1, tzinfo=datetime.timezone.utc)
    for _ in range(200):
        timestamps.append(int(start_time.timestamp() * 1000))
        values.append([float(0.1 * np.random.randn() - 2), float(0.1 * np.random.randn() - 2)])
        start_time += interval
    for _ in range(200):
        timestamps.append(int(start_time.timestamp() * 1000))
        values.append([float(0.1 * np.random.randn() + 2), float(0.1 * np.random.randn() + 2)])
        start_time += interval
    for _ in range(20):
        timestamps.append(int(start_time.timestamp() * 1000))
        values.append([float(np.random.uniform(-4, 4)), float(np.random.uniform(-4, 4))])
        start_time += interval

    measurements = ["a", "b"]
    data_types = [TSDataType.FLOAT, TSDataType.FLOAT]
    session.insert_records_of_one_device(
        device,
        timestamps,
        [measurements] * len(timestamps),
        [data_types] * len(timestamps),
        values,
    )

    if ground_truth_flag:
        ground_truth = np.ones(840, dtype=int)
        ground_truth[-20:] = -1
        ground_truth[400:420] = -1
        return ground_truth
    return None


def query_iotdb_data(
    session,
    database,
    table,
    start_time=None,
    end_time=None,
    ground_truth=None,
    time_serie=False,
    ground_truth_flag=False,
):
    """Read one IoTDB Tree device into a DataFrame, optionally aligning labels."""
    device = _device_path(database, table)
    start_bound = _time_bound(start_time)
    end_bound = _time_bound(end_time)
    if start_bound is not None and end_bound is not None and start_bound > end_bound:
        raise ValueError("start_time must not be later than end_time")

    predicates = []
    if start_bound is not None:
        predicates.append("time >= {}".format(start_bound))
    if end_bound is not None:
        predicates.append("time <= {}".format(end_bound))
    sql = "SELECT * FROM " + device
    if predicates:
        sql += " WHERE " + " AND ".join(predicates)
    sql += " ORDER BY time ASC"
    frame = _query_frame(session, sql)
    time_column = _time_column(frame) if len(frame.columns) else None
    if time_column is None:
        frame = pd.DataFrame(columns=["Time"])
        time_column = "Time"
    times = _as_utc_times(frame[time_column])

    if ground_truth_flag:
        if ground_truth is None:
            raise ValueError("ground_truth is required when ground_truth_flag is enabled")
        full_frame = _query_frame(session, "SELECT * FROM " + device + " ORDER BY time ASC")
        full_time_column = _time_column(full_frame) if len(full_frame.columns) else "Time"
        full_times = _as_utc_times(full_frame[full_time_column]) if len(full_frame.columns) else pd.to_datetime([])
        labels = np.asarray(ground_truth)
        if len(labels) != len(full_times):
            raise ValueError(
                "ground_truth length ({}) does not match IoTDB row count ({})".format(
                    len(labels), len(full_times)
                )
            )
        mask = np.ones(len(full_times), dtype=bool)
        if start_bound is not None:
            mask &= full_times >= pd.Timestamp(start_bound, unit="ms", tz="UTC")
        if end_bound is not None:
            mask &= full_times <= pd.Timestamp(end_bound, unit="ms", tz="UTC")
        labels = labels[mask]
        if len(labels) != len(frame):
            raise ValueError("IoTDB query rows do not align with the requested time window")

    frame = frame.copy()
    frame[time_column] = times
    if not time_serie:
        frame = frame.drop(columns=[time_column])
    frame = frame.ffill().bfill()

    if ground_truth_flag:
        return frame, labels
    return frame
