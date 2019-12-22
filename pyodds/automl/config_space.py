import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hyperopt import hp,tpe,fmin
from pyodds.algo.iforest import IFOREST
from pyodds.algo.ocsvm import OCSVM
from pyodds.algo.lof import LOF
from pyodds.algo.robustcovariance import RCOV
from pyodds.algo.staticautoencoder import StaticAutoEncoder
from pyodds.algo.luminolFunc import luminolDet
from pyodds.algo.cblof import CBLOF
from pyodds.algo.knn import KNN
from pyodds.algo.hbos import HBOS
from pyodds.algo.sod import SOD
from pyodds.algo.pca import PCA
from pyodds.algo.dagmm import DAGMM
from pyodds.algo.lstmad import LSTMAD
from pyodds.algo.lstmencdec import LSTMED
from pyodds.algo.autoencoder import AUTOENCODER


def construct_search_space():
	activation = hp.choice('activation', ['sigmoid', 'relu', 'hard_sigmoid'])
	random_state = np.random.randint(500)
	contamination = hp.choice('contamination', [0.5, 0.4, 0.3])

	space_config = hp.choice('classifier_type', [
		{
			'type': 'iforest', 'contamination': contamination, 'n_estimators': 100,
			'max_samples': "auto",
			'max_features': 1.,
			'bootstrap': False, 'n_jobs': None,
			'behaviour': 'old',
			'random_state': random_state
		},
		{
			'type': 'ocsvm',
			'gamma': 'auto', 'kernel': 'rbf',
			'degree': 3,
			'coef0': 0.0,
			'tol': 1e-3,
			'nu': 0.5,
			'shrinking': True, 'cache_size': 200, 'verbose': False, 'max_iter': -1,
		},
		{
			'type': 'lof',
			'contamination': contamination, 'n_neighbors': 20, 'algorithm': 'auto',
			'leaf_size': 30, 'metric': 'minkowski', 'p': 2, 'metric_params': None,
			'novelty': True,
		},
		{
			'type': 'robustcovariance',
			'random_state': random_state, 'store_precision': True,
			'assume_centered': False, 'support_fraction': None,
			'contamination': contamination,
		},
		{
			'type': 'staticautoencoder',
			'contamination': contamination, 'epoch': 100, 'dropout_rate': 0.2,
			'regularizer_weight': 0.1, 'activation': activation,
			'kernel_regularizer': 0.01,
			'loss_function': 'mse', 'optimizer': 'adam'
		},
		{
			'type': 'cblof',
			'contamination': contamination, 'n_clusters': 8,
			'clustering_estimator': None,
			'alpha': 0.9, 'beta': 5, 'use_weights': False,
			'random_state': random_state, 'n_jobs': 1,
		},
		{
			'type': 'knn',
			'contamination': contamination, 'n_neighbors': 5, 'method': 'largest',
			'radius': 1.0,
			'algorithm': 'auto', 'leaf_size': 30, 'metric': 'minkowski', 'p': 2,
			'metric_params': None, 'n_jobs': 1,
		},
		{
			'type': 'hbos',
			'contamination': contamination, 'n_bins': 10, 'alpha': 0.1, 'tol': 0.5,
		},
		{
			'type': 'sod',
			'contamination': contamination, 'n_neighbors': 20, 'ref_set': 10,
			'alpha': 0.8,
		},
		{
			'type': 'pca',
			'contamination': contamination, 'n_components': None,
			'n_selected_components': None,
			'copy': True, 'whiten': False, 'svd_solver': 'auto', 'tol': 0.0,
			'iterated_power': 'auto', 'random_state': random_state, 'weighted': True,
			'standardization': True,
		},
		{
			'type': 'dagmm',
			'contamination': contamination, 'num_epochs': 10, 'lambda_energy': 0.1,
			'lambda_cov_diag': 0.005, 'lr': 1e-3, 'batch_size': 50, 'gmm_k': 3,
			'normal_percentile': 80, 'sequence_length': 30, 'autoencoder_args': None,
		},
		# {
		# 	'type':'luminol','contamination':contamination
		# },
		{
			'type': 'autoencoder',
			'contamination': contamination, 'num_epochs': 10, 'batch_size': 20,
			'lr': 1e-3,
			'hidden_size': 5, 'sequence_length': 30,
			'train_gaussian_percentage': 0.25,
		},
		{
			'type': 'lstm_ad',
			'contamination': contamination, 'len_in': 1, 'len_out': 10,
			'num_epochs': 10,
			'lr': 1e-3, 'batch_size': 1,
		},
		{
			'type': 'lstm_ed',
			'contamination': contamination, 'num_epochs': 10, 'batch_size': 20,
			'lr': 1e-3,
			'hidden_size': 5, 'sequence_length': 30,
			'train_gaussian_percentage': 0.25,
		}])

	return space_config


CUMULATIVE_SEARCH_SPACE = construct_search_space()


def construct_classifier(x):
	clf = None
	if x['type'] == 'iforest':
		clf = IFOREST(contamination=x['contamination'], n_estimators=x['n_estimators'],
	                   max_samples=x['max_samples'], max_features=x['max_features'], bootstrap=x['bootstrap'],
	                   n_jobs=x['n_jobs'], behaviour=x['behaviour'], random_state=x['random_state'])
	elif x['type'] == 'ocsvm':
		clf = OCSVM(gamma=x['gamma'], kernel=x['kernel'], degree=x['degree'], coef0=x['coef0'], tol=x['tol'], nu=x['nu'],
	               shrinking=x['shrinking'], cache_size=x['cache_size'], verbose=x['verbose'], max_iter=x['max_iter'])
	elif x['type'] == 'lof':
		clf = LOF(contamination=x['contamination'], n_neighbors=x['n_neighbors'], algorithm=x['algorithm'],
	           leaf_size=x['leaf_size'], metric=x['metric'], p=x['p'], metric_params=x['metric_params'],
	           novelty=True)
	elif x['type'] == 'robustcovariance':
		clf = RCOV(random_state=x['random_state'], store_precision=x['store_precision'],
	                         assume_centered=x['assume_centered'], support_fraction=x['support_fraction'],
	                         contamination=x['contamination'])
	elif x['type'] == 'staticautoencoder':
		clf = StaticAutoEncoder(contamination=x['contamination'], epoch=x['epoch'],
	                                       dropout_rate=x['dropout_rate'], regularizer_weight=x['regularizer_weight'],
	                                       activation=x['activation'],
	                                       kernel_regularizer=x['kernel_regularizer'],
	                                       loss_function=x['loss_function'], optimizer=x['optimizer'])
	elif x['type'] == 'cblof':
		clf = CBLOF(contamination=x['contamination'], n_clusters=x['n_clusters'],
	               clustering_estimator=x['clustering_estimator'], alpha=x['alpha'], beta=x['beta'], use_weights=x['use_weights'],
	               random_state=x['random_state'], n_jobs=x['n_jobs'])
	elif x['type'] == 'knn':
		clf = KNN(contamination=x['contamination'], n_neighbors=x['n_neighbors'], method=x['method'],
	           radius=x['radius'], algorithm=x['algorithm'], leaf_size=x['leaf_size'], metric=x['metric'], p=x['p'],
	           metric_params=x['metric_params'], n_jobs=x['n_jobs'])
	elif x['type'] == 'hbos':
		clf = HBOS(contamination=x['contamination'], n_bins=x['n_bins'], alpha=x['alpha'], tol=x['tol'])
	elif x['type'] == 'sod':
		clf = SOD(contamination=x['contamination'], n_neighbors=x['n_neighbors'], ref_set=x['ref_set'], alpha=x['alpha'])
	elif x['type'] == 'pca':
		clf = PCA(contamination=x['contamination'], n_components=x['n_components'],
	           n_selected_components=x['n_selected_components'], copy=x['copy'], whiten=x['whiten'],
	           svd_solver=x['svd_solver'], tol=x['tol'], iterated_power=x['iterated_power'],
	           random_state=x['random_state'], weighted=x['weighted'], standardization=x['standardization'])
	elif x['type'] == 'dagmm':
		clf = DAGMM(contamination=x['contamination'], num_epochs=x['num_epochs'],
		            lambda_energy=x['lambda_energy'],lambda_cov_diag=x['lambda_cov_diag'],
		            lr=x['lr'], batch_size=x['batch_size'], gmm_k=x['gmm_k'],
		            normal_percentile=x['normal_percentile'], sequence_length=x['sequence_length'],
		            autoencoder_args=x['autoencoder_args'])
	elif x['type'] == 'luminol':
		clf = luminolDet(contamination=x['contamination'])
	elif x['type'] ==  'autoencoder':
		clf = AUTOENCODER(contamination=x['contamination'], num_epochs=x['num_epochs'],
	                           batch_size=x['batch_size'], lr=x['lr'], hidden_size=x['hidden_size'],
	                           sequence_length=x['sequence_length'], train_gaussian_percentage=x['train_gaussian_percentage'])
	elif x['type'] == 'lstm_ad':
		clf = LSTMAD(contamination=x['contamination'], len_in=x['len_in'], len_out=x['len_out'],
	                  num_epochs=x['num_epochs'], lr=x['lr'], batch_size=x['batch_size'])
	elif x['type'] == 'lstm_ed':
		clf = LSTMED(contamination=x['contamination'], num_epochs=x['num_epochs'], batch_size=x['batch_size'],
	                  lr=x['lr'], hidden_size=x['hidden_size'], sequence_length=x['sequence_length'],
	                  train_gaussian_percentage=x['train_gaussian_percentage'])
	else:
		clf = DAGMM(contamination=0.5, num_epochs=10, lambda_energy=0.1,
		      lambda_cov_diag=0.005, lr=1e-3, batch_size=50, gmm_k=3,
		      normal_percentile=80, sequence_length=30, autoencoder_args=None)

	print("Selected classifier is ",clf," for ",x)
	return clf


def plot_predictions(predictions,ground_truth,fname):
	x_ax = list(range(len(predictions)))
	plt.plot(x_ax,ground_truth,'b-')
	plt.plot(x_ax,predictions,'r-')
	plt.savefig(fname)
	return

