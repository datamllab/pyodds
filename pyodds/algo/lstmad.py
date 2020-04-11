import numpy as np
import pandas as pd
import torch
from scipy.stats import multivariate_normal
from torch.autograd import Variable
from tqdm import trange

from pyodds.algo.algorithm_utils import deepBase, PyTorchUtils
from pyodds.algo.base import Base


class LSTMAD(Base,deepBase, PyTorchUtils):
    """Malhotra, Pankaj, et al. "Long short term memory networks for anomaly detection in time series." Proceedings. Presses universitaires de Louvain, 2015.

    Long Short Term Memory (LSTM) networks have been
    demonstrated to be particularly useful for learning sequences containing
    longer term patterns of unknown length, due to their ability to maintain
    long term memory. Stacking recurrent hidden layers in such networks also
    enables the learning of higher level temporal features, for faster learning
    with sparser representations. In this paper, we use stacked LSTM networks for anomaly/fault detection in time series. A network is trained on
    non-anomalous data and used as a predictor over a number of time steps.
    The resulting prediction errors are modeled as a multivariate Gaussian
    distribution, which is used to assess the likelihood of anomalous behavior.

    Parameters
    ----------

    len_in: int, optional (default=1)
        The length of input layer

    len_out: int, optional (default=10)
        The length of output layer

    num_epochs: int, optional (default=100)
        The number of epochs

    lr: float, optional (default=1e-3)
        The speed of learning rate

    seed: int, optional (default=None)
        The random seed

    contamination: float in (0., 0.5), optional (default=0.05)
        The percentage of outliers

    """

    def __init__(self, len_in=1, len_out=10, num_epochs=10, lr=1e-3, batch_size=1,
                 seed: int=None, gpu: int=None, details=True,contamination=0.05):
        deepBase.__init__(self, __name__, 'LSTM-AD', seed, details=details)
        PyTorchUtils.__init__(self, seed, gpu)
        self.num_epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size

        self.len_in = len_in
        self.len_out = len_out

        self.mean, self.cov = None, None
        self.contamination=contamination

    def fit(self, X):
        """Fit detector.
        Parameters
        ----------
        X : dataframe of shape (n_samples, n_features)
            The input samples.
        """
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        self.batch_size = 1
        self._build_model(X.shape[-1], self.batch_size)

        self.model.train()
        split_point = int(0.75 * len(X))
        X_train = X.loc[:split_point, :]
        X_train_gaussian = X.loc[split_point:, :]

        input_data_train, target_data_train = self._input_and_target_data(X_train)
        self._train_model(input_data_train, target_data_train)

        self.model.eval()
        input_data_gaussian, target_data_gaussian = self._input_and_target_data_eval(X_train_gaussian)
        predictions_gaussian = self.model(input_data_gaussian)
        errors = self._calc_errors(predictions_gaussian, target_data_gaussian)
        norm = errors.reshape(errors.shape[0] * errors.shape[1], X.shape[-1] * self.len_out)
        self.mean = np.mean(norm, axis=0)
        self.cov = np.cov(norm.T)

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
        anomalies =self.decision_function(X)
        ranking = np.sort(anomalies)
        threshold = ranking[int((1-self.contamination)*len(ranking))]
        self.threshold = threshold
        mask = (anomalies>=threshold)
        ranking[mask]=-1
        ranking[np.logical_not(mask)]=1
        return ranking

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
            inliers with low scores from outliers with high score
        """
        outlier_score = self.decision_function(X)
        diff = outlier_score - self.threshold
        mask = diff > 0

        sc_pos = diff.clip(min=0)
        sc_neg = diff.clip(max=0)

        lmn = np.copy(diff)
        sc_pos = np.interp(sc_pos, (sc_pos.min(), sc_pos.max()), (0.5, 1))
        sc_neg = np.interp(sc_neg, (sc_neg.min(), sc_neg.max()), (0.0, 0.5))

        lmn[mask] = sc_pos[mask]
        lmn[np.logical_not(mask)] = sc_neg[np.logical_not(mask)]
        del diff, sc_pos, sc_neg
        return lmn

    def decision_function(self, X):
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
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        self.model.eval()
        input_data, target_data = self._input_and_target_data_eval(X)

        predictions = self.model(input_data)
        errors, stacked_preds = self._calc_errors(predictions, target_data, return_stacked_predictions=True)

        if self.details:
            self.prediction_details.update({'predictions_mean': np.pad(
                stacked_preds.mean(axis=3).squeeze(0).T, ((0, 0), (self.len_in + self.len_out - 1, 0)),
                'constant', constant_values=np.nan)})
            self.prediction_details.update({'errors_mean': np.pad(
                errors.mean(axis=3).reshape(-1), (self.len_in + self.len_out - 1, 0),
                'constant', constant_values=np.nan)})

        norm = errors.reshape(errors.shape[0] * errors.shape[1], X.shape[-1] * self.len_out)
        scores = -multivariate_normal.logpdf(norm, mean=self.mean, cov=self.cov, allow_singular=True)
        scores = np.pad(scores, (self.len_in + self.len_out - 1, 0), 'mean')
        return scores

    def _input_and_target_data(self, X: pd.DataFrame):
        X = np.expand_dims(X, axis=0)
        input_data = self.to_var(torch.from_numpy(X[:, :-self.len_out, :]), requires_grad=False)
        target_data = []
        for l in range(self.len_out - 1):
            target_data += [X[:, 1 + l:-self.len_out + 1 + l, :]]
        target_data += [X[:, self.len_out:, :]]
        target_data = self.to_var(torch.from_numpy(np.stack(target_data, axis=3)), requires_grad=False)

        return input_data, target_data

    def _input_and_target_data_eval(self, X: pd.DataFrame):
        X = np.expand_dims(X, axis=0)
        input_data = self.to_var(torch.from_numpy(X), requires_grad=False)
        target_data = self.to_var(torch.from_numpy(X[:, self.len_in + self.len_out - 1:, :]), requires_grad=False)
        return input_data, target_data

    def _calc_errors(self, predictions, target_data, return_stacked_predictions=False):
        errors = [predictions.data.numpy()[:, self.len_out - 1:-self.len_in, :, 0]]
        for l in range(1, self.len_out):
            errors += [predictions.data.numpy()[:, self.len_out - 1 - l:-self.len_in-l, :, l]]
        errors = np.stack(errors, axis=3)
        stacked_predictions = errors
        errors = target_data.data.numpy()[..., np.newaxis] - errors
        return errors if return_stacked_predictions is False else (errors, stacked_predictions)

    def _build_model(self, d, batch_size):
        self.model = LSTMSequence(d, batch_size, len_in=self.len_in, len_out=self.len_out)
        self.to_device(self.model)
        self.model.double()

        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def _train_model(self, input_data, target_data):
        def closure():
            return self._train(input_data, target_data)

        for _ in trange(self.num_epochs):
            self.optimizer.step(closure)

    def _train(self, input_data, target_data):
        self.optimizer.zero_grad()
        output_data = self.model(input_data)
        loss_train = self.loss(output_data, target_data)
        loss_train.backward()
        return loss_train


class LSTMSequence(torch.nn.Module):
    def __init__(self, d, batch_size: int, len_in=1, len_out=10):
        super().__init__()
        self.d = d  # input and output feature dimensionality
        self.batch_size = batch_size
        self.len_in = len_in
        self.len_out = len_out
        self.hidden_size1 = 32
        self.hidden_size2 = 32
        self.lstm1 = torch.nn.LSTMCell(d * len_in, self.hidden_size1)
        self.lstm2 = torch.nn.LSTMCell(self.hidden_size1, self.hidden_size2)
        self.linear = torch.nn.Linear(self.hidden_size2, d * len_out)

        self.register_buffer('h_t', torch.zeros(self.batch_size, self.hidden_size1))
        self.register_buffer('c_t', torch.zeros(self.batch_size, self.hidden_size1))
        self.register_buffer('h_t2', torch.zeros(self.batch_size, self.hidden_size1))
        self.register_buffer('c_t2', torch.zeros(self.batch_size, self.hidden_size1))

    def forward(self, input_x):
        outputs = []
        h_t = Variable(self.h_t.double(), requires_grad=False)
        c_t = Variable(self.c_t.double(), requires_grad=False)
        h_t2 = Variable(self.h_t2.double(), requires_grad=False)
        c_t2 = Variable(self.c_t2.double(), requires_grad=False)

        for input_t in input_x.chunk(input_x.size(1), dim=1):
            h_t, c_t = self.lstm1(input_t.squeeze(dim=1), (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze()  # stack (n, d * len_out) outputs in time dimensionality (dim=1)

        return outputs.view(input_x.size(0), input_x.size(1), self.d, self.len_out)
