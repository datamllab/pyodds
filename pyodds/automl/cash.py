from pyodds.automl.config_space import construct_search_space, plot_predictions, construct_classifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import numpy as np
import contextlib
import io
import time
from pyodds.utils.utilities import output_performance
from hyperopt import fmin,tpe, Trials,space_eval
from sklearn.metrics import roc_auc_score,mean_squared_error


class Cash():
	def __init__(self,data,ground_truth):
		self.results = ""
		self.data = data
		self.search_space = construct_search_space(self.data.shape[0],self.data.shape[1])
		self.gt = ground_truth
		self.count = 0
		self.best_classifier = None


	def objective_function(self,param,count):
		clf = construct_classifier(param)
		start = time.time()

		clf.fit(self.data)
		outlierness = clf.decision_function(self.data)
		predictions = clf.predict(self.data)

		if self.gt is not None:
			# store complete performance metrics for each trial
			f = io.StringIO()
			message = "\nTRIAL " + str(count) + " using " + str(param) +"\n"
			with contextlib.redirect_stdout(f):
				output_performance(clf, self.gt, predictions,
				                   time.time() - start, outlierness)
			message += f.getvalue()
			self.results += message
			f.close()
			# move to an internal method

			loss = -1* roc_auc_score(self.gt,predictions)
		else:
			# Trade-off to prefer FPs over FNs - Practical application HealthCare industry
			loss = mean_squared_error(predictions,np.asarray([0]*len(predictions)))
		return loss

	def f(self,params):
		self.count +=1
		return self.objective_function(params,self.count)

	def model_selector(self,max_evals=50):
		trials = Trials()
		best_clf = fmin(self.f, self.search_space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
		config = space_eval(self.search_space,best_clf)
		return construct_classifier(config) , self.results