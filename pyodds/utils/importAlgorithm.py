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

def algorithm_selection(algorithm,random_state,contamination):
    """
    Select algorithm from tokens.

    Parameters
    ----------
    algorithm: str, optional (default='iforest', choices=['iforest','lof','ocsvm','robustcovariance','staticautoencoder','luminol','cblof','knn','hbos','sod','pca','dagmm','autoencoder','lstm_ad','lstm_ed'])
        The name of the algorithm.
    random_state: np.random.RandomState
        The random state from the given random seeds.
    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set,
        i.e. the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.

    Returns
    -------
    alg: class
        The selected algorithm method.

    """
    algorithm_dic={'iforest':IFOREST(contamination=contamination,n_estimators=100,max_samples="auto", max_features=1.,bootstrap=False,n_jobs=None,behaviour='old',random_state=random_state,verbose=0),
                   'ocsvm':OCSVM(gamma='auto',kernel='rbf', degree=3,coef0=0.0, tol=1e-3, nu=0.5, shrinking=True, cache_size=200,verbose=False, max_iter=-1),
                   'lof': LOF(contamination=contamination,n_neighbors=20, algorithm='auto', leaf_size=30,metric='minkowski', p=2, metric_params=None, novelty=True),
                   'robustcovariance':RCOV(random_state=random_state,store_precision=True, assume_centered=False,support_fraction=None, contamination=0.1),
                   'staticautoencoder':StaticAutoEncoder(contamination=contamination,epoch=100,dropout_rate=0.2,regularizer_weight=0.1,activation='relu',kernel_regularizer=0.01,loss_function='mse',optimizer='adam'),
                   'cblof':CBLOF(contamination=contamination,n_clusters=8, clustering_estimator=None, alpha=0.9, beta=5,use_weights=False, random_state=random_state,n_jobs=1),
                   'knn':KNN(contamination=contamination,n_neighbors=5, method='largest',radius=1.0, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, n_jobs=1),
                   'hbos':HBOS(contamination=contamination, n_bins=10, alpha=0.1, tol=0.5),
                   'sod':SOD(contamination=contamination,n_neighbors=20, ref_set=10,alpha=0.8),
                   'pca':PCA(contamination=contamination, n_components=None, n_selected_components=None, copy=True, whiten=False, svd_solver='auto',tol=0.0, iterated_power='auto',random_state=random_state,weighted=True, standardization=True),
                   'dagmm':DAGMM(contamination=contamination,num_epochs=10, lambda_energy=0.1, lambda_cov_diag=0.005, lr=1e-3, batch_size=50, gmm_k=3, normal_percentile=80, sequence_length=30, autoencoder_args=None),
                   'luminol': luminolDet(contamination=contamination),
                   'autoencoder':AUTOENCODER(contamination=contamination,num_epochs=10, batch_size=20, lr=1e-3,hidden_size=5, sequence_length=30, train_gaussian_percentage=0.25),
                   'lstm_ad':LSTMAD(contamination=contamination,len_in=1, len_out=10, num_epochs=10, lr=1e-3, batch_size=1),
                   'lstm_ed':LSTMED(contamination=contamination,num_epochs=10, batch_size=20, lr=1e-3,hidden_size=5, sequence_length=30, train_gaussian_percentage=0.25)
                   }
    alg = algorithm_dic[algorithm]
    return alg

