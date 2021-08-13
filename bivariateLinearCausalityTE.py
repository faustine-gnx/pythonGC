import numpy as np
from scipy import signal, stats

def bivariateLinearCausalityTE(signals, tau=1, n_lags=5, verbose=False):
    """ Calculate the bivariate Granger causality between each pair of signals.

    :param signals: shape n_rois x n_timesteps (/!\ Matlab opposite
    :param tau:
    :param n_lags: number of past time steps to include in model
    :param verbose: set to True to display result and threshol

    :return TE_xy: transfer entropy (GC) from roi x to roi y
    :return F_stat: F statistics of the GC test
    :return threshold_F: threshold for significance.
    """
    (n_timesteps, n_rois) = np.shape(signals)
    n_pairs = n_rois * (n_rois-1)
    pval = 0.05 # uncorrected
    threshold_F = stats.f.ppf(1 - pval/n_pairs, n_lags, n_timesteps - 2*n_lags)  # statistical threshold (Bonferroni corrected)

    signals = detrend(signals)  # removes the linear trend from the data: makes the data stationary
    signals = normalisa(signals) # Matlab normalisa: mean=0, std=1 for each ROI

    for col in signal.T:



def detrend(signals):
    """ Removes the best straight-line fit linear trend from each column of the matrix signals. """
    return signal.detrend(signals, axis=0)


def embed(signal, embed_dim, tau=1):
    """ Embed data sequence signal. Creates an embedding matrix of dimension embed_dim and tau lags.
    :param signal: vector of the signal to be embedded
    :param embed_dim: embedding dimension = np.shape(embedded_signal, 2)
    :param tau: number of lags for the embedding (keep tau=1 for GC)
    """
    signal = signal.flatten()  # should already be a vector...
    n_timesteps = len(signal)

    return [signal[np.arange(0, n, tau)[:embed_dim] + i] for i in range(n_timesteps + tau - embed_dim * tau)]


def normalisa(signals):
    """ """
    signals_mean = np.mean(signals, 0)  # mean across time (i.e. per roi)
    signals_std = np.std(signals, 0, ddof=1)  # std across time (i.e. per roi); ddof = 1 to have same as matlab's
    # std function: normalized by N-1 --> get unbiased variance
    signals_normalized = (signals - signals_mean) / signals_std
    return signals_normalized
