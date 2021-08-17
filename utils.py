import numpy as np

# useless now --> embed once and for all and then retrieve instead of embedding at each for loop
# def embed(signal, embed_dim, tau=1):
#     """ Embed data sequence signal. Creates an embedding matrix of dimension embed_dim and tau lags.
#     :param signal: vector of the signal to be embedded
#     :param embed_dim: embedding dimension = np.shape(embedded_signal, 2)
#     :param tau: number of lags for the embedding (keep tau=1 for GC)
#     """
#     signal = signal.flatten()  # should already be a vector...
#     n_timesteps = len(signal)
#
#     return np.array([signal[np.arange(0, n_timesteps, tau)[:embed_dim] + i]
#                      for i in range(n_timesteps + tau - embed_dim * tau)])


def lag_signals(signals, n_lags, tau=1):
    """ Create matrix of lagged data sequence signal. Creates a matrix of dimension n_lags and tau lags.
    :param signals: matrix of the signals to be embedded
    :param n_lags: embedding dimension = np.shape(embedded_signal, 2)
    :param tau: number of lags for the embedding (keep tau=1 for GC)
    :return signals_lagged: matrix of embedded signals (shape: (n_rois, n_timesteps + tau - n_lags*tau, embed_dim))
    """
    (n_rois, n_timesteps) = np.shape(signals)
    signals_lagged = np.zeros((n_rois, n_timesteps + tau - n_lags * tau, n_lags))

    for i, x in enumerate(signals):
        signals_lagged[i] = np.array([x[np.arange(0, n_timesteps, tau)[:n_lags] + i]
                                        for i in range(n_timesteps + tau - n_lags * tau)])
    return signals_lagged

def normalisa(signals):
    """ """
    signals_mean = np.mean(signals, axis=1).reshape(np.shape(signals)[0], 1)  # mean across time (i.e. per roi)
    signals_std = np.std(signals, axis=1, ddof=1).reshape(np.shape(signals)[0], 1)  # std across time (i.e. per roi);
    # ddof = 1 to have same as matlab's --> std function: normalized by N-1 --> get unbiased variance
    signals_normalized = (signals - signals_mean) / signals_std
    return signals_normalized
