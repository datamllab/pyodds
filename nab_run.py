import numpy as np
import argparse
import time
import logging
import pandas as pd
import os
import io
import contextlib
import getpass
import warnings
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from pyodds.utils.utilities import output_performance,insert_demo_data,connect_server,query_data
from pyodds.utils.importAlgorithm import algorithm_selection
from pyodds.utils.plotUtils import visualize_distribution_static,visualize_distribution_time_serie,visualize_outlierscore,visualize_distribution
from pyodds.utils.utilities import str2bool
from pyodds.automl.cash import Cash
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
    result_directory = './results'
    if not os.path.exists(result_directory):
        os.mkdir(result_directory)
    gt_directory = args.nab_datadir
    result_table = pd.DataFrame(columns=['data','Prec','Recall','F1','ROC','time','model'])

    for file in os.listdir(gt_directory):
        print(file)
        data = pd.read_csv(os.path.join(gt_directory,file))
        data['value'] = data['value'].astype('float')
        ground_truth = data['label'].values
        data.drop(['label','timestamp','Unnamed: 0'],axis=1,inplace=True)

        if args.ground_truth:
            alg_selector = Cash(data, ground_truth)
        else:
            alg_selector = Cash(data, None)

        print('Start AutoML:')
        start_time = time.clock()
        clf , results = alg_selector.model_selector(max_evals=50)


        print('End AutoML:')
        end_time = time.clock()

        clf.fit(data)
        prediction_result = clf.predict(data)
        outlierness = clf.decision_function(data)

        print('Auto ML complete')
        results += "\n\n>>> >>> >>> >>> >>> >>> >>> === >>> >>> >>> >>> >>> >>> >>>\n\nFINAL RESULT AFTER RETRAINING\n"

        if args.ground_truth:

            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                output_performance(clf, ground_truth, prediction_result,
                                   end_time - start_time, outlierness)
            results += f.getvalue()
            f.close()

            acc = accuracy_score(ground_truth, prediction_result)
            prec = precision_score(ground_truth, prediction_result)
            rec = recall_score(ground_truth, prediction_result)
            f1 = f1_score(ground_truth, prediction_result)
            roc = max(roc_auc_score(ground_truth, outlierness),
                                            1 - roc_auc_score(ground_truth,
                                                              outlierness))

            with open('./results/results'+str(file)+'.txt','w') as f:
                f.write(results)
            row = [file,prec,rec,f1,roc,time.clock() - start_time,str(clf)]
            result_table.loc[len(result_table)] = row

        print('Final result complete')

    result_table.to_csv('NAB_FINALS.csv')
