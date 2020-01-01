"""Adapted from Daniel Stanley Tan (https://github.com/danieltan07/dagmm)"""
import logging
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import trange

from pyodds.algo.algorithm_utils import deepBase, PyTorchUtils
from pyodds.algo.autoencoder import AutoEncoderModule
from pyodds.algo.lstmencdec import LSTMEDModule
from pyodds.algo.base import Base


class DAGMM(Base,deepBase, PyTorchUtils):
    """
    Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection, Zong et al, 2018.
    Unsupervised anomaly detection on multi- or high-dimensional data is of great importance in both fundamental machine learning research and industrial applications, for which density estimation lies at the core. Although previous approaches based on dimensionality reduction followed by density estimation have made fruitful progress, they mainly suffer from decoupled model learning with inconsistent optimization goals and incapability of preserving essential information in the low-dimensional space. In this paper, we present a Deep Autoencoding Gaussian Mixture Model (DAGMM) for unsupervised anomaly detection. Our model utilizes a deep autoencoder to generate a low-dimensional representation and reconstruction error for each input data point, which is further fed into a Gaussian Mixture Model (GMM). Instead of using decoupled two-stage training and the standard Expectation-Maximization (EM) algorithm, DAGMM jointly optimizes the parameters of the deep autoencoder and the mixture model simultaneously in an end-to-end fashion, leveraging a separate estimation network to facilitate the parameter learning of the mixture model. The joint optimization, which well balances autoencoding reconstruction, density estimation of latent representation, and regularization, helps the autoencoder escape from less attractive local optima and further reduce reconstruction errors, avoiding the need of pre-training.

    Parameters
    ----------
    
    num_epochs: int, optional (default=10)
        The number of epochs
        
    lambda_energy: float, optional (default=0.1)
        The parameter to balance the energy in loss function
    
    lambda_cov_diag: float, optional (default=0.05)
        The parameter to balance the covariance in loss function

    lr: float, optional (default=1e-3)
        The speed of learning rate

    batch_size: int, optional (default=50)
        The number of samples in one batch
    
    gmm_k: int, optional (default=3)
        The number of clusters in the Gaussian Mixture model

    sequence_length: int, optional (default=30)
        The length of sequence

    hidden_size: int, optional (default=5)
        The size of hidden layer

    seed: int, optional (default=None)
        The random seed

    contamination: float in (0., 0.5), optional (default=0.05)
        The percentage of outliers

    """
    class AutoEncoder:
        NN = AutoEncoderModule
        LSTM = LSTMEDModule

    def __init__(self, num_epochs=10, lambda_energy=0.1, lambda_cov_diag=0.005, lr=1e-3, batch_size=50, gmm_k=3,
                 normal_percentile=80, sequence_length=30, autoencoder_type=AutoEncoderModule, autoencoder_args=None,
                 hidden_size: int=5, seed: int=None, gpu: int=None, details=True,contamination=0.05):
        _name = 'LSTM-DAGMM' if autoencoder_type == LSTMEDModule else 'DAGMM'
        deepBase.__init__(self, __name__, _name, seed, details=details)
        PyTorchUtils.__init__(self, seed, gpu)
        self.num_epochs = num_epochs
        self.lambda_energy = lambda_energy
        self.lambda_cov_diag = lambda_cov_diag
        self.lr = lr
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.gmm_k = gmm_k  # Number of Gaussian mixtures
        self.normal_percentile = normal_percentile  # Up to which percentile data should be considered normal
        self.autoencoder_type = autoencoder_type
        if autoencoder_type == AutoEncoderModule:
            self.autoencoder_args = ({'sequence_length': self.sequence_length})
        elif autoencoder_type == LSTMEDModule:
            self.autoencoder_args = ({'n_layers': (1, 1), 'use_bias': (True, True), 'dropout': (0.0, 0.0)})
        self.autoencoder_args.update({'seed': seed, 'gpu': gpu})
        if autoencoder_args is not None:
            self.autoencoder_args.update(autoencoder_args)
        self.hidden_size = hidden_size

        self.dagmm, self.optimizer, self.train_energy, self._threshold = None, None, None, None
        self.contamination=contamination

    def reset_grad(self):
        self.dagmm.zero_grad()

    def dagmm_step(self, input_data):
        self.dagmm.train()
        _, dec, z, gamma = self.dagmm(input_data)
        total_loss, sample_energy, recon_error, cov_diag = self.dagmm.loss_function(input_data, dec, z, gamma,
                                                                                    self.lambda_energy,
                                                                                    self.lambda_cov_diag)
        self.reset_grad()
        total_loss = torch.clamp(total_loss, max=1e7)  # Extremely high loss can cause NaN gradients
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dagmm.parameters(), 5)
        self.optimizer.step()
        return total_loss, sample_energy, recon_error, cov_diag


    def anomaly_likelihood(self, X):
        k = self.decision_function(X)
        diff = k - self.threshold
        mask = diff > 0

        sc_pos = diff.clip(min=0)
        sc_neg = diff.clip(max=0)

        lmn = np.copy(diff)

        sc_pos = np.interp(sc_pos, (sc_pos.min(), sc_pos.max()), (0.5, 1))
        sc_neg = np.interp(sc_neg, (sc_neg.min(), sc_neg.max()), (0.0, 0.5))
        # print(sc_pos,sc_neg)
        lmn[mask] = sc_pos[mask]
        lmn[np.logical_not(mask)] = sc_neg[np.logical_not(mask)]
        del diff
        del sc_pos
        del sc_neg
        return lmn

    def fit(self, X: pd.DataFrame):
        """Learn the mixture probability, mean and covariance for each component k.
        Store the computed energy based on the training data and the aforementioned parameters.
        Parameters
        ----------
        X : dataframe of shape (n_samples, n_features)
            The input samples.
        """
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        sequences = [data[i:i + self.sequence_length] for i in range(X.shape[0] - self.sequence_length + 1)]
        data_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.hidden_size = 5 + int(X.shape[1] / 20)
        autoencoder = self.autoencoder_type(X.shape[1], hidden_size=self.hidden_size, **self.autoencoder_args)
        self.dagmm = DAGMMModule(autoencoder, n_gmm=self.gmm_k, latent_dim=self.hidden_size + 2,
                                 seed=self.seed, gpu=self.gpu)
        self.to_device(self.dagmm)
        self.optimizer = torch.optim.Adam(self.dagmm.parameters(), lr=self.lr)

        for _ in trange(self.num_epochs):
            for input_data in data_loader:
                input_data = self.to_var(input_data)
                self.dagmm_step(input_data.float())

        self.dagmm.eval()
        n = 0
        mu_sum = 0
        cov_sum = 0
        gamma_sum = 0
        for input_data in data_loader:
            input_data = self.to_var(input_data)
            _, _, z, gamma = self.dagmm(input_data.float())
            _, mu, cov = self.dagmm.compute_gmm_params(z, gamma)

            batch_gamma_sum = torch.sum(gamma, dim=0)

            gamma_sum += batch_gamma_sum
            mu_sum += mu * batch_gamma_sum.unsqueeze(-1)  # keep sums of the numerator only
            cov_sum += cov * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1)  # keep sums of the numerator only

            n += input_data.size(0)
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

    def decision_function(self, X: pd.DataFrame):
        """Predict raw anomaly score of X using the fitted detector.
        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.
        Using the learned mixture probability, mean and covariance for each component k, compute the energy on the
        given data.

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
        self.dagmm.eval()
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        sequences = [data[i:i + self.sequence_length] for i in range(len(data) - self.sequence_length + 1)]
        data_loader = DataLoader(dataset=sequences, batch_size=1, shuffle=False)
        test_energy = np.full((self.sequence_length, X.shape[0]), np.nan)

        encodings = np.full((self.sequence_length, X.shape[0], self.hidden_size), np.nan)
        decodings = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
        euc_errors = np.full((self.sequence_length, X.shape[0]), np.nan)
        csn_errors = np.full((self.sequence_length, X.shape[0]), np.nan)

        for i, sequence in enumerate(data_loader):
            enc, dec, z, _ = self.dagmm(self.to_var(sequence).float())
            sample_energy, _ = self.dagmm.compute_energy(z, size_average=False)
            idx = (i % self.sequence_length, np.arange(i, i + self.sequence_length))
            test_energy[idx] = sample_energy.data.numpy()

            if self.details:
                encodings[idx] = enc.data.numpy()
                decodings[idx] = dec.data.numpy()
                euc_errors[idx] = z[:, 1].data.numpy()
                csn_errors[idx] = z[:, 2].data.numpy()

        test_energy = np.nanmean(test_energy, axis=0)

        if self.details:
            self.prediction_details.update({'latent_representations': np.nanmean(encodings, axis=0).T})
            self.prediction_details.update({'reconstructions_mean': np.nanmean(decodings, axis=0).T})
            self.prediction_details.update({'euclidean_errors_mean': np.nanmean(euc_errors, axis=0)})
            self.prediction_details.update({'cosine_errors_mean': np.nanmean(csn_errors, axis=0)})

        return test_energy


class DAGMMModule(nn.Module, PyTorchUtils):
    """Residual Block."""

    def __init__(self, autoencoder, n_gmm, latent_dim, seed: int, gpu: int):
        super(DAGMMModule, self).__init__()
        PyTorchUtils.__init__(self, seed, gpu)

        self.add_module('autoencoder', autoencoder)

        layers = [
            nn.Linear(latent_dim, 10),
            nn.Tanh(),
            nn.Dropout(p=0.5),
            nn.Linear(10, n_gmm),
            nn.Softmax(dim=1)
        ]
        self.estimation = nn.Sequential(*layers)
        self.to_device(self.estimation)

        self.register_buffer('phi', self.to_var(torch.zeros(n_gmm)))
        self.register_buffer('mu', self.to_var(torch.zeros(n_gmm, latent_dim)))
        self.register_buffer('cov', self.to_var(torch.zeros(n_gmm, latent_dim, latent_dim)))

    def relative_euclidean_distance(self, a, b, dim=1):
        return (a - b).norm(2, dim=dim) / torch.clamp(a.norm(2, dim=dim), min=1e-10)

    def forward(self, x):
        dec, enc = self.autoencoder(x, return_latent=True)

        rec_cosine = F.cosine_similarity(x.view(x.shape[0], -1), dec.view(dec.shape[0], -1), dim=1)
        rec_euclidean = self.relative_euclidean_distance(x.view(x.shape[0], -1), dec.view(dec.shape[0], -1), dim=1)

        # Concatenate latent representation, cosine similarity and relative Euclidean distance between x and dec(enc(x))
        z = torch.cat([enc, rec_euclidean.unsqueeze(-1), rec_cosine.unsqueeze(-1)], dim=1)
        gamma = self.estimation(z)

        return enc, dec, z, gamma

    def compute_gmm_params(self, z, gamma):
        N = gamma.size(0)
        # K
        sum_gamma = torch.sum(gamma, dim=0)

        # K
        phi = (sum_gamma / N)

        self.phi = phi.data

        # K x D
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
        self.mu = mu.data
        # z = N x D
        # mu = K x D
        # gamma N x K

        # z_mu = N x K x D
        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))

        # z_mu_outer = N x K x D x D
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

        # K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim=0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        self.cov = cov.data

        return phi, mu, cov

    def compute_energy(self, z, phi=None, mu=None, cov=None, size_average=True):
        if phi is None:
            phi = Variable(self.phi)
        if mu is None:
            mu = Variable(self.mu)
        if cov is None:
            cov = Variable(self.cov)

        k, d, _ = cov.size()

        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))

        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-12
        for i in range(k):
            # K x D x D
            cov_k = cov[i] + self.to_var(torch.eye(d) * eps)
            pinv = np.linalg.pinv(cov_k.data.numpy())
            cov_inverse.append(Variable(torch.from_numpy(pinv)).unsqueeze(0))

            eigvals = np.linalg.eigvals(cov_k.data.cpu().numpy() * (2 * np.pi))
            if np.min(eigvals) < 0:
                logging.warning(f'Determinant was negative! Clipping Eigenvalues to 0+epsilon from {np.min(eigvals)}')
            determinant = np.prod(np.clip(eigvals, a_min=sys.float_info.epsilon, a_max=None))
            det_cov.append(determinant)

            cov_diag = cov_diag + torch.sum(1 / cov_k.diag())

        # K x D x D
        cov_inverse = torch.cat(cov_inverse, dim=0)
        # K
        det_cov = Variable(torch.from_numpy(np.float32(np.array(det_cov))))

        # N x K
        exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        # for stability (logsumexp)
        max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]

        exp_term = torch.exp(exp_term_tmp - max_val)

        sample_energy = -max_val.squeeze() - torch.log(
            torch.sum(self.to_var(phi.unsqueeze(0)) * exp_term / (torch.sqrt(self.to_var(det_cov)) + eps).unsqueeze(0),
                      dim=1) + eps)

        if size_average:
            sample_energy = torch.mean(sample_energy)

        return sample_energy, cov_diag

    def loss_function(self, x, x_hat, z, gamma, lambda_energy, lambda_cov_diag):
        recon_error = torch.mean((x.view(*x_hat.shape) - x_hat) ** 2)
        phi, mu, cov = self.compute_gmm_params(z, gamma)
        sample_energy, cov_diag = self.compute_energy(z, phi, mu, cov)
        loss = recon_error + lambda_energy * sample_energy + lambda_cov_diag * cov_diag
        return loss, sample_energy, recon_error, cov_diag
