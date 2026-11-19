import os
import uuid

import pytest

from pyodds.utils.iotdb import connect_iotdb_server, insert_iotdb_demo_data, query_iotdb_data


@pytest.mark.skipif(not os.getenv('IOTDB_TEST_PORT'), reason='requires an isolated Tree RPC endpoint')
def test_demo_and_utc_label_window():
    session = connect_iotdb_server('127.0.0.1', os.environ['IOTDB_TEST_USER'],
                                   os.environ['IOTDB_TEST_PASSWORD'], os.environ['IOTDB_TEST_PORT'])
    database = 'root.pyodds_' + uuid.uuid4().hex
    try:
        labels = insert_iotdb_demo_data(session, database, 'sensor', True)
        full, aligned = query_iotdb_data(session, database, 'sensor', ground_truth=labels,
                                         ground_truth_flag=True, time_serie=True)
        assert len(full) == len(aligned) == 840
        start, end = full['Time'].iloc[399], full['Time'].iloc[402]
        window, subset = query_iotdb_data(session, database, 'sensor', start, end, labels, True, True)
        assert len(window) == 4 and subset.tolist() == [1, -1, -1, -1]
    finally:
        try:
            session.execute_non_query_statement('DROP DATABASE ' + database)
        finally:
            session.close()
