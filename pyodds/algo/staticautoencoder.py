from pyodds.algo.base import Base
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from sklearn import preprocessing
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
class StaticAutoEncoder(Base):
    def __init__(self,hidden_neurons=None,epoch=100,dropout_rate=0.2,contamination=0.1,regularizer_weight=0.1,activation='relu',kernel_regularizer=0.01,loss_function='mse',optimizer='adam'):
        self.hidden_neurons=hidden_neurons
        self.epoch=epoch
        self.dropout_rate=dropout_rate
        self.contamination=contamination
        self.regularizer_weight=regularizer_weight
        self.activation=activation
        self.kernel_regularizer=kernel_regularizer
        self.loss_function=loss_function
        self.optimizer=optimizer
        self.threshold = None

        if self.hidden_neurons and  self.hidden_neurons != self.hidden_neurons[::-1]:
            print(self.hidden_neurons)
            raise ValueError("Hidden units should be symmetric")

    def _build_model(self):
        model =  tf.keras.Sequential()
        for neuron_num in self.hidden_neurons:
            model.add(layers.Dense(neuron_num,activation=self.activation,kernel_regularizer=tf.keras.regularizers.l1(self.kernel_regularizer)))
            model.add(layers.Dropout(self.dropout_rate))
        model.compile(loss=self.loss_function,optimizer=self.optimizer)
        return model

    def fit(self, X):
        """Fit detector.
        Parameters
        ----------
        X : dataframe of shape (n_samples, n_features)
            The input samples.
        """
        scaler = preprocessing.RobustScaler().fit(X)
        X_train = scaler.transform(X)
        if self.hidden_neurons is None:
            self.hidden_neurons=[X_train.shape[1]//2+1,X_train.shape[1]//4+1,X_train.shape[1]//4+1,X_train.shape[1]//2+1]
        self.batch_size=X_train.shape[0]//10
        self.model=self._build_model()

        self.model.fit(X_train,X_train,epochs=self.epoch,batch_size=self.batch_size)

        return self

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
        reconstruct_error= (np.square(self.model.predict(X)-X)).mean(axis=1)
        ranking = np.sort(reconstruct_error)
        threshold = ranking[int((1-self.contamination)*len(ranking))]
        self.threshold = threshold
        mask = (reconstruct_error>=threshold)
        ranking[mask]=-1
        ranking[np.logical_not(mask)]=1
        return ranking

    def anomaly_likelihood(self, X):
        ourlier_score = self.decision_function(X)
        diff = ourlier_score - self.threshold
        mask = diff > 0

        sc_pos = np.clip(diff, a_min=0, a_max=None)
        sc_neg = np.clip(diff, a_max=0, a_min=None)

        lmn = np.copy(diff)
        sc_pos = np.interp(sc_pos, (sc_pos.min(), sc_pos.max()), (0.5, 1))
        sc_neg = np.interp(sc_neg, (sc_neg.min(), sc_neg.max()), (0.0, 0.5))

        lmn[mask] = sc_pos[mask]
        lmn[np.logical_not(mask)] = sc_neg[np.logical_not(mask)]
        del diff, sc_pos, sc_neg
        return lmn

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
        reconstruct_error= (np.square(self.model.predict(X)-X)).mean(axis=1)
        return reconstruct_error
