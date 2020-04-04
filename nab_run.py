import numpy as np
import argparse
import time
import logging
import pandas as pd
import os
import warnings
import json
from pyodds.algo.luminolFunc import luminolDet
import matplotlib.pyplot as plt
from pyodds.utils.importAlgorithm import algorithm_selection
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from pyodds.utils.utilities import output_performance
from pyodds.utils.plotUtils import visualize_distribution_static,visualize_distribution_time_serie,visualize_outlierscore,visualize_distribution
from pyodds.utils.utilities import str2bool
from pyodds.automl.cash import Cash

import matplotlib
matplotlib.rcParams['figure.dpi'] = 400
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", UserWarning)
logging.disable(logging.WARNING)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Anomaly Detection Platform Settings")
    parser.add_argument('--nab_datadir',default='./with_gt')
    parser.add_argument('--random_seed',default=42, type=int)
    parser.add_argument('--visualize_distribution',default=True,const=True,type=str2bool,nargs='?')
    parser.add_argument('--start_time',default='2019-07-20 00:00:00')
    parser.add_argument('--end_time',default='2019-08-20 00:00:00')
    parser.add_argument('--ground_truth',default=True,const=True,type=str2bool,nargs='?')
    parser.add_argument('--saving_path',default='./output/img')

    args = parser.parse_args()
    result_directory = './nab_results'
    log_dir = './automllogs'
    if not os.path.exists(result_directory):
        os.mkdir(result_directory)

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    gt_directory = args.nab_datadir
    if not os.path.exists(gt_directory):
        print("NAB dataset with ground truth is not available - Terminating program")
    else:

        for file in os.listdir(gt_directory):
            f_name = file.split('.')[0][:-8]
            print("Current dataset ",f_name)
            data = pd.read_csv(os.path.join(gt_directory,file))

            data['value'] = data['value'].astype('float')
            ground_truth = data['label'].values
            labels_01 = np.copy(ground_truth)
            labels_01[labels_01 == 1]=0
            labels_01[labels_01 == -1] = 1

            with open('./combined_windows.json') as f:
                jstring = f.read()
            jdict = json.loads(jstring)

            windows = []
            for k in jdict:
                if f_name in k:
                    windows = jdict[k]
                    break
            final_store = data[['timestamp','value','label']]
            final_store['label']= labels_01
            data.drop(['label','Unnamed: 0'],axis=1,inplace=True)

            if args.ground_truth:
                alg_selector = Cash(data.copy(deep=True), ground_truth,windows)
            else:
                alg_selector = Cash(data.copy(deep=True), None,windows)

            start_time = time.clock()
            clf , results = alg_selector.model_selector(max_evals=50)
            end_time = time.clock()


            if isinstance(clf,luminolDet):
                clf.fit(data)
            else:
                data.drop(["timestamp"],axis=1,inplace=True,errors="ignore")
                clf.fit(data)

            prediction_result = clf.predict(data)
            outlierness = clf.decision_function(data)
            anomaly_scores = clf.anomaly_likelihood(data)

            final_store['anomaly_score'] = anomaly_scores
            final_store.to_csv(result_directory+'/pyodds_'+str(f_name)+'.csv')

            if args.ground_truth:
                prec = precision_score(ground_truth, prediction_result)
                rec = recall_score(ground_truth, prediction_result)
                f1 = f1_score(ground_truth, prediction_result)
                roc = max(roc_auc_score(ground_truth, outlierness),
                                                1 - roc_auc_score(ground_truth,
                                                                  outlierness))
                results.loc[len(results)] = [clf,f1,roc,end_time-start_time,prec,rec]
                results.to_csv(log_dir+'/results_'+str(f_name)+'.csv')



