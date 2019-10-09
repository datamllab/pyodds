import pytest
import numpy as np
import time

from pyodds.utils.utilities import output_performance,insert_demo_data,connect_server,query_data
from pyodds.utils.importAlgorithm import algorithm_selection
from pyodds.utils.plotUtils import visualize_distribution_static,visualize_distribution_time_serie,visualize_outlierscore,visualize_distribution

from pyodds.utils.utilities import check_parameter,standardizer,str2bool

# @pytest.fixture(scope='module')
def test_io_static():
    host = '127.0.0.1'
    user = 'user'
    password = '0906'
    alg = 'iforest'

    rng = np.random.RandomState(42)
    np.random.seed(42)
    conn,cursor=connect_server(host, user, password)
    ground_truth_whole = insert_demo_data(conn, cursor, 'db', 't', ground_truth_flag=True)
    data, ground_truth = query_data(conn, cursor, 'db', 't',time_serie_name='ts', ground_truth=ground_truth_whole,start_time='2019-07-20 00:00:00',end_time='2019-08-20 00:00:00',time_serie=False, ground_truth_flag=True)

    clf = algorithm_selection(alg, random_state=rng, contamination=0.1)
    print('Start processing:')
    start_time = time.clock()
    clf.fit(data)
    prediction_result = clf.predict(data)
    outlierness = clf.decision_function(data)
    output_performance(alg, ground_truth, prediction_result, time.clock() - start_time, outlierness)

    visualize_distribution_static(data, prediction_result, outlierness)
    visualize_distribution(data, prediction_result, outlierness)
    visualize_outlierscore(outlierness, prediction_result, contamination=0.1)

def test_io_time_serie():
    host = '127.0.0.1'
    user = 'user'
    password = '0906'
    alg = 'luminol'

    rng = np.random.RandomState(42)
    np.random.seed(42)
    conn,cursor=connect_server(host, user, password)
    ground_truth_whole = insert_demo_data(conn, cursor, 'db', 't', ground_truth_flag=True)
    data, ground_truth = query_data(conn, cursor, 'db', 't',time_serie_name='ts', ground_truth=ground_truth_whole,start_time='2019-07-20 00:00:00',end_time='2019-08-20 00:00:00',time_serie=True, ground_truth_flag=True)

    clf = algorithm_selection(alg, random_state=rng, contamination=0.1)
    print('Start processing:')
    start_time = time.clock()
    clf.fit(data)
    prediction_result = clf.predict(data)
    outlierness = clf.decision_function(data)
    output_performance(alg, ground_truth, prediction_result, time.clock() - start_time, outlierness)

    visualize_distribution_time_serie(clf.ts, data)

def test_function():
    check_parameter(2,low=1,high=3)
    standardizer(np.random.rand(3,2)*100)
    str2bool('True')

if __name__ == "__main__":
    test_io_static()
    test_io_time_serie()
    test_function()
    print("Everything passed")
