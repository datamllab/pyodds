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


class AUTOENCODER(Base,deepBase, PyTorchUtils):
    """Auto Encoder (AE) is a type of neural networks for learning useful data representations unsupervisedly.
    It could be used to detect outlying objects in the data by calculating the reconstruction errors.

    Parameters
    ----------

    name: str, optional (default='AutoEncoder')
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

    seed: int, optional (default=None)
        The random seed

    contamination: float in (0., 0.5), optional (default=0.05)
        The percentage of outliers

    """
    def __init__(self, name: str='AutoEncoder', num_epochs: int=10, batch_size: int=20, lr: float=1e-3,
                 hidden_size: int=5, sequence_length: int=30, train_gaussian_percentage: float=0.25,
                 seed: int=None, gpu: int=None, details=True,contamination=0.05):
        deepBase.__init__(self, __name__, name, seed, details=details)
        PyTorchUtils.__init__(self, seed, gpu)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr

        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.train_gaussian_percentage = train_gaussian_percentage
        self.contamination=contamination
        self.aed = None
        self.mean, self.cov = None, None

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

        self.aed = AutoEncoderModule(X.shape[1], self.sequence_length, self.hidden_size, seed=self.seed, gpu=self.gpu)
        self.to_device(self.aed)  # .double()
        optimizer = torch.optim.Adam(self.aed.parameters(), lr=self.lr)

        self.aed.train()
        for epoch in trange(self.num_epochs):
            logging.debug(f'Epoch {epoch+1}/{self.num_epochs}.')
            for ts_batch in train_loader:
                output = self.aed(self.to_var(ts_batch))
                loss = nn.MSELoss(size_average=False)(output, self.to_var(ts_batch.float()))
                self.aed.zero_grad()
                loss.backward()
                optimizer.step()

        self.aed.eval()
        error_vectors = []
        for ts_batch in train_gaussian_loader:
            output = self.aed(self.to_var(ts_batch))
            error = nn.L1Loss(reduce=False)(output, self.to_var(ts_batch.float()))
            error_vectors += list(error.view(-1, X.shape[1]).data.cpu().numpy())

        self.mean = np.mean(error_vectors, axis=0)
        self.cov = np.cov(error_vectors, rowvar=False)

    def predict(self, X):
        """Return outliers with -1 and inliers with 1, with the outlierness score calculated from the `decision_function(X)`,
        and the threshold `contamination`.
        Parameters
        ----------
        X : dataframe of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        ranking : numpy array of shape (n_samples,)
            The outlierness of the input samples.
        """
        anomalies = self.decision_function(X)
        ranking = np.sort(anomalies)
        threshold = ranking[int((1 - self.contamination) * len(ranking))]
        self.threshold = threshold
        mask = (anomalies >= threshold)
        ranking[mask] = -1
        ranking[np.logical_not(mask)] = 1
        return ranking



    def decision_function(self, X: pd.DataFrame) -> np.array:
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

        self.aed.eval()
        mvnormal = multivariate_normal(self.mean, self.cov, allow_singular=True)
        scores = []
        outputs = []
        errors = []
        for _, ts in enumerate(data_loader):
            output = self.aed(self.to_var(ts))
            error = nn.L1Loss(reduce=False)(output, self.to_var(ts.float()))
            score = -mvnormal.logpdf(error.view(-1, X.shape[1]).data.cpu().numpy())
            scores.append(score.reshape(ts.size(0), self.sequence_length))
            if self.details:
                outputs.append(output.data.numpy())
                errors.append(error.data.numpy())

        # stores seq_len-many scores per timestamp and averages them
        scores = np.concatenate(scores)
        lattice = np.full((self.sequence_length, X.shape[0]), np.nan)
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

    def anomaly_likelihood(self, X: pd.DataFrame) -> np.array:
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



class AutoEncoderModule(nn.Module, PyTorchUtils):
    def __init__(self, n_features: int, sequence_length: int, hidden_size: int, seed: int, gpu: int):
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        input_length = n_features * sequence_length

        dec_steps = 2 ** np.arange(max(np.ceil(np.log2(hidden_size)), 2), np.log2(input_length))[1:]
        dec_setup = np.concatenate([[hidden_size], dec_steps.repeat(2), [input_length]])
        enc_setup = dec_setup[::-1]

        layers = np.array([[nn.Linear(int(a), int(b)), nn.Tanh()] for a, b in enc_setup.reshape(-1, 2)]).flatten()[:-1]
        self._encoder = nn.Sequential(*layers)
        self.to_device(self._encoder)

        layers = np.array([[nn.Linear(int(a), int(b)), nn.Tanh()] for a, b in dec_setup.reshape(-1, 2)]).flatten()[:-1]
        self._decoder = nn.Sequential(*layers)
        self.to_device(self._decoder)

    def forward(self, ts_batch, return_latent: bool=False):
        flattened_sequence = ts_batch.view(ts_batch.size(0), -1)
        enc = self._encoder(flattened_sequence.float())
        dec = self._decoder(enc)
        reconstructed_sequence = dec.view(ts_batch.size())
        return (reconstructed_sequence, enc) if return_latent else reconstructed_sequence
