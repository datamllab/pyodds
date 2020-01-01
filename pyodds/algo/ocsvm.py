from sklearn.svm import OneClassSVM
from pyodds.algo.base import Base
import numpy as np

class OCSVM(OneClassSVM,Base):
    """Unsupervised Outlier Detection.
    Estimate the support of a high-dimensional distribution.
    The implementation is based on libsvm.
    Read more in the :ref:`User Guide <outlier_detection>`.
    Parameters
    ----------
    kernel : string, optional (default='rbf')
         Specifies the kernel type to be used in the algorithm.
         It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or
         a callable.
         If none is given, 'rbf' will be used. If a callable is given it is
         used to precompute the kernel matrix.
    degree : int, optional (default=3)
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.
    gamma : float, optional (default='auto')
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        Current default is 'auto' which uses 1 / n_features,
        if ``gamma='scale'`` is passed then it uses 1 / (n_features * X.var())
        as value of gamma. The current default of gamma, 'auto', will change
        to 'scale' in version 0.22. 'auto_deprecated', a deprecated version of
        'auto' is used as a default indicating that no explicit value of gamma
        was passed.
    coef0 : float, optional (default=0.0)
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.
    tol : float, optional
        Tolerance for stopping criterion.
    nu : float, optional
        An upper bound on the fraction of training
        errors and a lower bound of the fraction of support
        vectors. Should be in the interval (0, 1]. By default 0.5
        will be taken.
    shrinking : boolean, optional
        Whether to use the shrinking heuristic.
    cache_size : float, optional
        Specify the size of the kernel cache (in MB).
    verbose : bool, default: False
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in libsvm that, if enabled, may not work
        properly in a multithreaded context.
    max_iter : int, optional (default=-1)
        Hard limit on iterations within solver, or -1 for no limit.
    random_state : int, RandomState instance or None, optional (default=None)
        Ignored.
        .. deprecated:: 0.20
           ``random_state`` has been deprecated in 0.20 and will be removed in
           0.22.
    Attributes
    ----------
    support_ : array-like, shape = [n_SV]
        Indices of support vectors.
    support_vectors_ : array-like, shape = [nSV, n_features]
        Support vectors.
    dual_coef_ : array, shape = [1, n_SV]
        Coefficients of the support vectors in the decision function.
    coef_ : array, shape = [1, n_features]
        Weights assigned to the features (coefficients in the primal
        problem). This is only available in the case of a linear kernel.
        `coef_` is readonly property derived from `dual_coef_` and
        `support_vectors_`
    intercept_ : array, shape = [1,]
        Constant in the decision function.
    offset_ : float
        Offset used to define the decision function from the raw scores.
        We have the relation: decision_function = score_samples - `offset_`.
        The offset is the opposite of `intercept_` and is provided for
        consistency with other outlier detection algorithms.
    Examples
    --------
    >>> from sklearn.svm import OneClassSVM
    >>> X = [[0], [0.44], [0.45], [0.46], [1]]
    >>> clf = OneClassSVM(gamma='auto').fit(X)
    >>> clf.predict(X)
    array([-1,  1,  1,  1, -1])
    >>> clf.score_samples(X)  # doctest: +ELLIPSIS
    array([1.7798..., 2.0547..., 2.0556..., 2.0561..., 1.7332...])
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