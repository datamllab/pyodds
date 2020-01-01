from sklearn.neighbors import LocalOutlierFactor
from pyodds.algo.base import Base
import numpy as np

class LOF(LocalOutlierFactor,Base):

    """Unsupervised Outlier Detection using Local Outlier Factor (LOF)
    The anomaly score of each sample is called Local Outlier Factor.
    It measures the local deviation of density of a given sample with
    respect to its neighbors.
    It is local in that the anomaly score depends on how isolated the object
    is with respect to the surrounding neighborhood.
    More precisely, locality is given by k-nearest neighbors, whose distance
    is used to estimate the local density.
    By comparing the local density of a sample to the local densities of
    its neighbors, one can identify samples that have a substantially lower
    density than their neighbors. These are considered outliers.

    Parameters
    ----------
    n_neighbors : int, optional (default=20)
        Number of neighbors to use by default for :meth:`kneighbors` queries.
        If n_neighbors is larger than the number of samples provided,
        all samples will be used.
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        Algorithm used to compute the nearest neighbors:
        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.
        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.
    leaf_size : int, optional (default=30)
        Leaf size passed to :class:`BallTree` or :class:`KDTree`. This can
        affect the speed of the construction and query, as well as the memory
        required to store the tree. The optimal value depends on the
        nature of the problem.
    metric : string or callable, default 'minkowski'
        metric used for the distance computation. Any metric from scikit-learn
        or scipy.spatial.distance can be used.
        If 'precomputed', the training input X is expected to be a distance
        matrix.
        If metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays as input and return one value indicating the
        distance between them. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.
        Valid values for metric are:

        - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
        
        - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
          'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao',
          'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
          'yule']
        See the documentation for scipy.spatial.distance for details on these
        metrics:
        https://docs.scipy.org/doc/scipy/reference/spatial.distance.html
    p : integer, optional (default=2)
        Parameter for the Minkowski metric from
        :func:`sklearn.metrics.pairwise.pairwise_distances`. When p = 1, this
        is equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
    metric_params : dict, optional (default=None)
        Additional keyword arguments for the metric function.
    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. When fitting this is used to define the
        threshold on the decision function. If "auto", the decision function
        threshold is determined as in the original paper.
        .. versionchanged:: 0.20
           The default value of ``contamination`` will change from 0.1 in 0.20
           to ``'auto'`` in 0.22.
    novelty : boolean, default False
        By default, LocalOutlierFactor is only meant to be used for outlier
        detection (novelty=False). Set novelty to True if you want to use
        LocalOutlierFactor for novelty detection. In this case be aware that
        that you should only use predict, decision_function and score_samples
        on new unseen data and not on the training set.
    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        Affects only :meth:`kneighbors` and :meth:`kneighbors_graph` methods.
    Attributes
    ----------
    negative_outlier_factor_ : numpy array, shape (n_samples,)
        The opposite LOF of the training samples. The higher, the more normal.
        Inliers tend to have a LOF score close to 1 (``negative_outlier_factor_``
        close to -1), while outliers tend to have a larger LOF score.
        The local outlier factor (LOF) of a sample captures its
        supposed 'degree of abnormality'.
        It is the average of the ratio of the local reachability density of
        a sample and those of its k-nearest neighbors.
    n_neighbors_ : integer
        The actual number of neighbors used for :meth:`kneighbors` queries.
    offset_ : float
        Offset used to obtain binary labels from the raw scores.
        Observations having a negative_outlier_factor smaller than `offset_`
        are detected as abnormal.
        The offset is set to -1.5 (inliers score around -1), except when a
        contamination parameter different than "auto" is provided. In that
        case, the offset is defined in such a way we obtain the expected
        number of outliers in training.
    References
    ----------
    .. [1] Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000, May).
           LOF: identifying density-based local outliers. In ACM sigmod record.
    """

    def anomaly_likelihood(self, X):
        print("Base implementation called - Threshold 0 and outliers are -ve scores")
        k = self.decision_function(X)

        mask = k < 0

        sc_pos = k.clip(max=0)
        sc_neg = k.clip(min=0)

        lmn = np.copy(k)
        sc_pos = np.interp(sc_pos, (sc_pos.min(), sc_pos.max()), (1, 0.5))
        sc_neg = np.interp(sc_neg, (sc_neg.min(), sc_neg.max()), (0.5, 0.0))

        lmn[mask] = sc_pos[mask]
        lmn[np.logical_not(mask)] = sc_neg[np.logical_not(mask)]

        del k
        del sc_pos
        del sc_neg
        return lmn