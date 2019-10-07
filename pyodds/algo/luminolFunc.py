from pyodds.algo.base import Base
from luminol import anomaly_detector
import numpy as np
import pandas as pd
from luminol.modules.time_series import TimeSeries
from luminol.utils import to_epoch
from sklearn.decomposition import IncrementalPCA

class luminolDet(Base):
    """
    Luminol is a light weight python library for time series data analysis. The two major functionalities it supports are anomaly detection and correlation. It can be used to investigate possible causes of anomaly.

    Parameters
    ----------
    contamination : float in (0., 0.5), optional (default=0.1)
    The amount of contamination of the data set,
    i.e. the proportion of outliers in the data set. Used when fitting to
    define the threshold on the decision function.

    """
    def __init__(self,contamination=0.1):
        self.contamination=contamination

    def fit(self,X):
        """Fit detector.
        Parameters
        ----------
        X : dataframe of shape (n_samples, n_features)
            The input samples.
        """
        # a=str(ts[:,0])
        X=X.to_numpy()
        timestamp = np.asarray(X[:,0].astype(np.datetime64))
        pca = IncrementalPCA(n_components=1)
        value=np.reshape(pca.fit_transform(X[:,1:]),-1)
        X = pd.Series(value, index=timestamp)
        X.index = X.index.map(lambda d: to_epoch(str(d)))
        lts = TimeSeries(X.to_dict())
        self.ts=timestamp
        self.ts_value=value
        self.detector = anomaly_detector.AnomalyDetector(lts)

        return self

    def predict(self,X):
        """Return outliers with -1 and inliers with 1, with the outlierness score calculated from the `decision_function(X)',
        and the threshold `contamination'.
        Parameters
        ----------
        X : dataframe of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        ranking : numpy array of shape (n_samples,)
            The outlierness of the input samples.
        """
        anomalies = np.reshape(self.detector.get_all_scores().values,-1)
        self.decision=anomalies
        ranking = np.sort(anomalies)
        threshold = ranking[int((1-self.contamination)*len(ranking))]
        self.threshold = threshold
        mask = (anomalies>=threshold)
        ranking[mask]=-1
        ranking[np.logical_not(mask)]=1
        return ranking

    def decision_function(self,X):
        """Predict raw anomaly score of X using the fitted detector.

        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.

        Parameters
        ----------
        X : dataframe of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.
        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        return self.decision




