from pyodds.automl.config_space import construct_search_space, plot_predictions, construct_classifier
from sklearn.metrics import roc_curve,precision_score,recall_score,f1_score
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from pyodds.utils.utilities import output_performance
from hyperopt import fmin,tpe, Trials,space_eval,rand
from sklearn.metrics import roc_auc_score,mean_squared_error


class Cash():
	def __init__(self,data,ground_truth):
		self.results = pd.DataFrame(columns=['Trial','Algorithm','F1_Score','ROC','Time','Precision','Recall'])
		self.data = data
		self.plotter = pd.DataFrame(columns=['Trial','Algorithm','Fpr','Tpr','Thresholds'])
		self.search_space = construct_search_space(self.data.shape[0],self.data.shape[1])
		self.gt = ground_truth
		self.count = 0
		self.best_classifier = None

	def objective_function(self,param,count):
		clf = construct_classifier(param)
		start = time.time()
		clf.fit(self.data)
		#outlierness = clf.decision_function(self.data)
		predictions = clf.predict(self.data)
		anomaly_scores = clf.anomaly_likelihood(self.data)
		end = time.time()

		labels_01 = np.copy(self.gt)
		labels_01[labels_01 == 1] = 0
		labels_01[labels_01 == -1] = 1
		if self.gt is not None:
			fpr, tpr, thresholds = roc_curve(self.gt, predictions)
			self.plotter.loc[len(self.plotter)] = [str(count),clf,fpr, tpr,thresholds]
			roc = (max(roc_auc_score(labels_01, anomaly_scores),1-roc_auc_score(labels_01, anomaly_scores)))
			self.results.loc[len(self.results)] = [str(count),clf,f1_score(self.gt,predictions),roc,end-start,precision_score(self.gt,predictions),recall_score(self.gt,predictions)]
			loss = -1* roc
		else:
			# Trade-off to prefer FPs over FNs - Practical application HealthCare industry
			loss = mean_squared_error(predictions,np.asarray([0]*len(predictions)))
		return loss

	def f(self,params):
		self.count +=1
		return self.objective_function(params,self.count)

	def model_selector(self,max_evals=50):
		trials = Trials()
		best_clf = fmin(self.f, self.search_space, algo=rand.suggest, max_evals=max_evals, trials=trials)
		config = space_eval(self.search_space,best_clf)

		return construct_classifier(config) , self.results, self.plotter