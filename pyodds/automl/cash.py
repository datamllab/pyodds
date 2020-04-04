from pyodds.automl.config_space import construct_search_space, plot_predictions, construct_classifier
from sklearn.metrics import roc_curve,precision_score,recall_score,f1_score
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import json
import datetime
from pyodds.automl.sweeper import Sweeper
from pyodds.automl.optimizer import optimizeThreshold
from pyodds.utils.utilities import output_performance
from hyperopt import fmin,tpe, Trials,space_eval,rand
from sklearn.metrics import roc_auc_score,mean_squared_error


class Cash():
	def __init__(self,data,ground_truth,windows=None):
		self.results = pd.DataFrame(columns=['Algorithm','F1_Score','ROC','Time','Precision','Recall'])

		self.luminol_data = data.copy(deep=True)
		data.drop(["timestamp"],axis=1,inplace=True,errors="ignore")
		self.data = data
		self.plotter = pd.DataFrame(columns=['Trial','Algorithm','Fpr','Tpr','Thresholds'])
		self.search_space = construct_search_space(self.data.shape[0],self.data.shape[1])
		self.gt = ground_truth
		self.count = 0
		self.nab = False
		self.best_classifier = None
		if windows is not None:
			self.nab = True
			self.windows = windows
			cm = {"tpWeight": 1.0, "fnWeight": 1.0, "fpWeight": 0.11, "tnWeight": 1.0}
			self.sweeper = Sweeper(probationPercent=0.15, costMatrix=cm)

	def objective_function(self,param,count):
		clf = construct_classifier(param)
		print("classifier type",param["type"])
		start = time.time()

		if param["type"] == "luminol":
			clf.fit(self.luminol_data)
		else:
			clf.fit(self.data)
		predictions = clf.predict(self.data)
		outlierness = clf.decision_function(self.data)
		anomaly_scores = clf.anomaly_likelihood(self.data)
		if self.nab:
			slots = []
			for w in self.windows:
				slots.append([datetime.datetime.strptime(w[0],"%Y-%m-%d %H:%M:%S.%f"),datetime.datetime.strptime(w[1],"%Y-%m-%d %H:%M:%S.%f")])

			t_dict = optimizeThreshold((self.sweeper,slots,self.luminol_data['timestamp'].values,anomaly_scores))
			(scores, bestRow) = self.sweeper.scoreDataSet(self.luminol_data['timestamp'].values,anomaly_scores,slots,t_dict['threshold'])

		end = time.time()
		labels_01 = np.copy(self.gt)
		labels_01[labels_01 == 1] = 0
		labels_01[labels_01 == -1] = 1
		if len(np.unique(self.gt)) == 1:
			roc = 0
			metric = f1_score(self.gt, predictions)
		else:
			if self.nab :
				roc = 0
				metric = bestRow.score
			else:
				roc = (max(roc_auc_score(self.gt, outlierness),
				           1 - roc_auc_score(self.gt, outlierness)))
				metric = roc
		self.results.loc[len(self.results)] = [str(param),f1_score(self.gt,predictions),roc,end-start,precision_score(self.gt,predictions),recall_score(self.gt,predictions)]
		del labels_01
		loss = -1 * metric

		return loss

	def f(self,params):
		self.count += 1
		return self.objective_function(params,self.count)

	def model_selector(self,max_evals=50):
		trials = Trials()
		best_clf = fmin(self.f, self.search_space, algo=rand.suggest, max_evals=max_evals, trials=trials)
		config = space_eval(self.search_space,best_clf)
		del self.data
		del self.luminol_data
		return construct_classifier(config), self.results