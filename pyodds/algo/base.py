class Base(object):
    """
    Abstract class for all outlier detection algorithms.

    """
    def __init__(self):
        pass


    def fit(self, X):
        """Fit detector.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        """
        pass

    def predict(self, X):

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
        pass

    def decision_function(self,X):
        """Predict raw anomaly scores of X using the fitted detector.
        The anomaly score of an input sample is computed based on the fitted
        detector. For consistency, outliers are assigned with
        higher anomaly scores.
        Parameters
        ----------
        X : dataframe of shape (n_samples, n_features)
            The input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.
        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        pass

    def anomaly_likelihood(self, X):
        """A normalization function to clip and scale the outlier_scores returned
        by self.decision_function(). Normalization is done separately for data
        points falling above and below the threshold
        Parameters
        ----------
        X : dataframe of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.
        Returns
        -------
        normalized_anomaly_scores : numpy array of shape (n_samples,)
            Normalized anomaly scores where 0.5 is the default threshold separating
            low-scoring inliers from high-scoring outliers
        """
        pass

