import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import multivariate_normal
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import trange

from pyodds.algo.algorithm_utils import deepBase, PyTorchUtils
from pyodds.algo.base import Base


class LSTMED(Base,deepBase, PyTorchUtils):
    """
    Malhotra, Pankaj, et al. "LSTM-based encoder-decoder for multi-sensor anomaly detection." ICML, 2016.


    Mechanical devices such as engines, vehicles, aircrafts, etc., are typically instrumented with numerous sensors to capture the behavior and health of the machine. However, there are often external factors or variables which are not captured by sensors leading to time-series which are inherently unpredictable. For instance, manual controls and/or unmonitored environmental conditions or load may lead to inherently unpredictable time-series. Detecting anomalies in such scenarios becomes challenging using standard approaches based on mathematical models that rely on stationarity, or prediction models that utilize prediction errors to detect anomalies. We propose a Long Short Term Memory Networks based Encoder-Decoder scheme for Anomaly Detection (EncDec-AD) that learns to reconstruct 'normal' time-series behavior, and thereafter uses reconstruction error to detect anomalies. We experiment with three publicly available quasi predictable time-series datasets: power demand, space shuttle, and ECG, and two real-world engine datasets with both predictive and unpredictable behavior.

    Parameters
    ----------

    name: str, optional default='LSTM-ED'
        The name of the algorithm

    num_epochs: int, optional (default=10)
        The number of epochs

    batch_size: int, optional (default=20)
        The number of batch size

    lr: float, optional (default=1e-3)
        The speed of learning rate

    hidden_size: int, optional (default=5)
        The number of hidden layer

    sequence_length: int, optional (default=30)
        The length of sequence

    train_gaussian_percentage: float, optional (default=0.25)
        The percentage for gaussian training

    n_layers: tuple, optional (default=(1,1))
        The number of hidden layers

    use_bias: tuple, optional (default=(True, True))
        Whether use bias or not in hidden layers

    dropout: tuple, optional (default=(0, 0))
        Dropout rates in hidden layers

    seed: int, optional (default=None)
        The random seed

    contamination: float in (0., 0.5), optional (default=0.05)
        The percentage of outliers

    """

    def __init__(self, name: str='LSTM-ED', num_epochs: int=10, batch_size: int=20, lr: float=1e-3,
                 hidden_size: int=5, sequence_length: int=30, train_gaussian_percentage: float=0.25,
                 n_layers: tuple=(1, 1), use_bias: tuple=(True, True), dropout: tuple=(0, 0),
                 seed: int=None, gpu: int = None, details=True,contamination=0.05):
        deepBase.__init__(self, __name__, name, seed, details=details)
        PyTorchUtils.__init__(self, seed, gpu)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr

        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.train_gaussian_percentage = train_gaussian_percentage

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.lstmed = None
        self.mean, self.cov = None, None
        self.contamination=contamination

    def fit(self, X: pd.DataFrame):
        """Fit detector.
        Parameters
        ----------
        X : dataframe of shape (n_samples, n_features)
            The input samples.
        """
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        sequences = [data[i:i + self.sequence_length] for i in range(data.shape[0] - self.sequence_length + 1)]
        indices = np.random.permutation(len(sequences))
        split_point = int(self.train_gaussian_percentage * len(sequences))
        train_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                                  sampler=SubsetRandomSampler(indices[:-split_point]), pin_memory=True)
        train_gaussian_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                                           sampler=SubsetRandomSampler(indices[-split_point:]), pin_memory=True)

        self.lstmed = LSTMEDModule(X.shape[1], self.hidden_size,
                                   self.n_layers, self.use_bias, self.dropout,
                                   seed=self.seed, gpu=self.gpu)
        self.to_device(self.lstmed)
        optimizer = torch.optim.Adam(self.lstmed.parameters(), lr=self.lr)

        self.lstmed.train()
        for epoch in trange(self.num_epochs):
            logging.debug(f'Epoch {epoch+1}/{self.num_epochs}.')
            for ts_batch in train_loader:
                output = self.lstmed(self.to_var(ts_batch))
                loss = nn.MSELoss(size_average=False)(output, self.to_var(ts_batch.float()))
                self.lstmed.zero_grad()
                loss.backward()
                optimizer.step()

        self.lstmed.eval()
        error_vectors = []
        for ts_batch in train_gaussian_loader:
            output = self.lstmed(self.to_var(ts_batch))
            error = nn.L1Loss(reduce=False)(output, self.to_var(ts_batch.float()))
            error_vectors += list(error.view(-1, X.shape[1]).data.cpu().numpy())

        self.mean = np.mean(error_vectors, axis=0)
        self.cov = np.cov(error_vectors, rowvar=False)

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

    def decision_function(self, X: pd.DataFrame):
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
        data = X.values
        sequences = [data[i:i + self.sequence_length] for i in range(data.shape[0] - self.sequence_length + 1)]
        data_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, shuffle=False, drop_last=False)

        self.lstmed.eval()
        mvnormal = multivariate_normal(self.mean, self.cov, allow_singular=True)
        scores = []
        outputs = []
        errors = []
        for _, ts in enumerate(data_loader):
            output = self.lstmed(self.to_var(ts))
            error = nn.L1Loss(reduce=False)(output, self.to_var(ts.float()))
            score = -mvnormal.logpdf(error.view(-1, X.shape[1]).data.cpu().numpy())
            scores.append(score.reshape(ts.size(0), self.sequence_length))
            if self.details:
                outputs.append(output.data.numpy())
                errors.append(error.data.numpy())

        # stores seq_len-many scores per timestamp and averages them
        scores = np.concatenate(scores)
        lattice = np.full((self.sequence_length, data.shape[0]), np.nan)
        for i, score in enumerate(scores):
            lattice[i % self.sequence_length, i:i + self.sequence_length] = score
        scores = np.nanmean(lattice, axis=0)

        if self.details:
            outputs = np.concatenate(outputs)
            lattice = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
            for i, output in enumerate(outputs):
                lattice[i % self.sequence_length, i:i + self.sequence_length, :] = output
            self.prediction_details.update({'reconstructions_mean': np.nanmean(lattice, axis=0).T})

            errors = np.concatenate(errors)
            lattice = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
            for i, error in enumerate(errors):
                lattice[i % self.sequence_length, i:i + self.sequence_length, :] = error
            self.prediction_details.update({'errors_mean': np.nanmean(lattice, axis=0).T})

        return scores


class LSTMEDModule(nn.Module, PyTorchUtils):
    def __init__(self, n_features: int, hidden_size: int,
                 n_layers: tuple, use_bias: tuple, dropout: tuple,
                 seed: int, gpu: int):
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        self.n_features = n_features
        self.hidden_size = hidden_size

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.encoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[0], bias=self.use_bias[0], dropout=self.dropout[0])
        self.to_device(self.encoder)
        self.decoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[1], bias=self.use_bias[1], dropout=self.dropout[1])
        self.to_device(self.decoder)
        self.hidden2output = nn.Linear(self.hidden_size, self.n_features)
        self.to_device(self.hidden2output)

    def _init_hidden(self, batch_size):
        return (self.to_var(torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_()),
                self.to_var(torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_()))

    def forward(self, ts_batch, return_latent: bool=False):
        batch_size = ts_batch.shape[0]

        enc_hidden = self._init_hidden(batch_size)  # initialization with zero
        _, enc_hidden = self.encoder(ts_batch.float(), enc_hidden)  # .float() here or .double() for the model

        dec_hidden = enc_hidden
        output = self.to_var(torch.Tensor(ts_batch.size()).zero_())
        for i in reversed(range(ts_batch.shape[1])):
            output[:, i, :] = self.hidden2output(dec_hidden[0][0, :])

            if self.training:
                _, dec_hidden = self.decoder(ts_batch[:, i].unsqueeze(1).float(), dec_hidden)
            else:
                _, dec_hidden = self.decoder(output[:, i].unsqueeze(1), dec_hidden)

        return (output, enc_hidden[1][-1]) if return_latent else output
