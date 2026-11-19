import argparse
import getpass
import logging
import time
import warnings

import numpy as np

from pyodds.automl.cash import Cash
from pyodds.utils.importAlgorithm import algorithm_selection
from pyodds.utils.iotdb import connect_iotdb_server, insert_iotdb_demo_data, query_iotdb_data
from pyodds.utils.plotUtils import (
    visualize_distribution,
    visualize_distribution_static,
    visualize_distribution_time_serie,
    visualize_outlierscore,
)
from pyodds.utils.utilities import (
    connect_server,
    insert_demo_data,
    output_performance,
    query_data,
    str2bool,
)

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", UserWarning)
logging.disable(logging.WARNING)


def _parser():
    parser = argparse.ArgumentParser(description="Anomaly Detection Platform Settings")
    parser.add_argument("--backend", default="tdengine", choices=["tdengine", "iotdb"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--user", required=True)
    parser.add_argument("--random_seed", default=42, type=int)
    parser.add_argument("--database", default="db")
    parser.add_argument("--table", default="t")
    parser.add_argument("--time_stamp", const=True, type=str2bool, nargs="?")
    parser.add_argument("--visualize_distribution", default=True, const=True, type=str2bool, nargs="?")
    parser.add_argument(
        "--algorithm",
        default="dagmm",
        choices=[
            "iforest", "lof", "ocsvm", "robustcovariance", "staticautoencoder",
            "luminol", "cblof", "knn", "hbos", "sod", "pca", "dagmm",
            "autoencoder", "lstm_ad", "lstm_ed",
        ],
    )
    parser.add_argument("--contamination", default=0.05, type=float)
    parser.add_argument("--start_time", default="2019-07-20 00:00:00")
    parser.add_argument("--end_time", default="2019-08-20 00:00:00")
    parser.add_argument("--time_serie_name", default="ts")
    parser.add_argument("--ground_truth", default=True, const=True, type=str2bool, nargs="?")
    parser.add_argument("--saving_path", default="./output/img")
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if args.backend == "tdengine" and args.port is not None:
        parser.error("--port is only supported with --backend iotdb")

    rng = np.random.RandomState(args.random_seed)
    np.random.seed(args.random_seed)
    password = getpass.getpass("Please input your password:")

    if args.backend == "iotdb":
        conn = connect_iotdb_server(args.host, args.user, password, port=args.port or 6667)
        insert_data = insert_iotdb_demo_data

        def read_data(database, table, **kwargs):
            return query_iotdb_data(conn, database, table, **kwargs)

    else:
        conn, cursor = connect_server(args.host, args.user, password)

        def insert_data(_connection, database, table, ground_truth_flag):
            return insert_demo_data(conn, cursor, database, table, ground_truth_flag)

        def read_data(database, table, **kwargs):
            return query_data(conn, cursor, database, table, **kwargs)

    try:
        print("Load dataset and table")
        start = time.perf_counter()
        ground_truth_whole = insert_data(conn, args.database, args.table, args.ground_truth)

        query_options = {
            "start_time": args.start_time,
            "end_time": args.end_time,
            "time_serie": args.time_stamp,
            "ground_truth_flag": args.ground_truth,
        }
        if args.backend == "iotdb":
            if args.ground_truth:
                data, ground_truth = read_data(
                    args.database,
                    args.table,
                    ground_truth=ground_truth_whole,
                    **query_options,
                )
            else:
                data = read_data(args.database, args.table, **query_options)
        else:
            query_options["time_serie_name"] = args.time_serie_name
            if args.ground_truth:
                query_options["ground_truth"] = ground_truth_whole
                data, ground_truth = read_data(args.database, args.table, **query_options)
            else:
                data = read_data(args.database, args.table, **query_options)

        print("Loading cost: %.6f seconds" % (time.perf_counter() - start))
        print("Load data successful")

        if args.ground_truth:
            alg_selector = Cash(data, ground_truth)
            clf = alg_selector.model_selector(max_evals=2)
        else:
            clf = algorithm_selection(
                args.algorithm,
                random_state=rng,
                contamination=args.contamination,
            )

        start = time.perf_counter()
        clf.fit(data)
        prediction_result = clf.predict(data)
        outlierness = clf.decision_function(data)
        anomaly_scores = clf.anomaly_likelihood(data)

        if args.ground_truth:
            output_performance(
                args.algorithm,
                ground_truth,
                prediction_result,
                time.perf_counter() - start,
                anomaly_scores,
            )

        if args.visualize_distribution and args.ground_truth:
            if not args.time_stamp:
                visualize_distribution_static(data, prediction_result, outlierness, args.saving_path)
                visualize_distribution(data, prediction_result, outlierness, args.saving_path)
                visualize_outlierscore(
                    outlierness,
                    prediction_result,
                    args.contamination,
                    args.saving_path,
                )
            else:
                visualize_distribution_time_serie(clf.ts, data, args.saving_path)
                visualize_outlierscore(
                    outlierness,
                    prediction_result,
                    args.contamination,
                    args.saving_path,
                )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
