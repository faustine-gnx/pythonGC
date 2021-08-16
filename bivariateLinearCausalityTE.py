import numpy as np
from scipy import signal, stats

def bivariateLinearCausalityTE(signals, n_lags=5, tau=1, pval=0.05, verbose=False):
    """ Calculate the bivariate Granger causality between each pair of signals.
    cov([...]) --> symmetric square matrix (cov between each pair of elements)
    determinant of cov = generalized variance: measure of multi-dimensional scatter (scalar value) / linked to
    differential entropy -->  the determinant is nonzero if and only if the matrix  is invertible, and the linear map
    represented by the matrix is an isomorphism. det is zero if and only if the column vectors (or the row vectors) of
    the matrix are linearly dependent.
    see slide 32 of GC presentation
    Fstat ~ F(n_lags, n_timesteps - 2*n_lags)

    :param signals: shape n_rois x n_timesteps (/!\ Matlab opposite)
    :param tau: number of time steps between lags kept for the embedding
           --> keep past values at times: [t-tau*i for i in range(n_lags)]
           (tau=1 for GC: keep all values up to n_lags, don't skip any)
    :param n_lags: number of past time steps to include in model (order)
    :param verbose: set to True to display result and threshold

    :return GC_sig: significant values of Granger causality matrix
    :return GC: Granger causality matrix
    :return F_stat: F statistics of the GC test
    :return threshold_F: threshold for significance.
    """
    (n_rois, n_timesteps) = np.shape(signals)
    n_pairs = n_rois * (n_rois-1)
    # From Fstat definition: F_gc ~ F(n_lags, n_timesteps - 2*n_lags)
    threshold_F = stats.f.ppf(1 - pval/n_pairs, n_lags, n_timesteps - 2*n_lags)  # statistical threshold
    # Bonferroni corrected 1 - pval/n_pairs instead of 1 - pval: n_pairs = number of hypotheses tested,
    # pval = significance level

    signals = signal.detrend(signals)  # removes the linear trend from each signal: makes the data stationary
    signals = normalisa(signals)  # Matlab normalisa: mean=0, std=1 for each ROI

    Fstat = np.zeros((n_rois, n_rois))  # matrix of all F_xy
    GC = np.zeros((n_rois, n_rois))  # matrix of all GC_xy
    GC_sig = np.zeros((n_rois, n_rois))  # matrix of all significant GC_xy (if F_xy >= threshold_F)

    for i, x in enumerate(signals): # for each column (each roi)
        x_embedded = embed(x, n_lags+1, tau)
        x_tau = x_embedded[:, :-1]  # past of signal x (embedded)

        for j, y in enumerate(signals):
            if i != j:
                y_embedded = embed(y, n_lags+1, tau)
                y_t = y_embedded[:, -1].reshape((len(y_embedded), 1))   # current value of signal y (embedded)
                y_tau = y_embedded[:, :-1]  # past of signal y (embedded)
                xtau_ytau = np.concatenate((x_tau, y_tau), axis=1)  # both past concatenated
                ytau_yt = np.concatenate((y_tau, y_t), axis=1)  # y's past and current value --> reduced model
                xtau_ytau_yt = np.concatenate((xtau_ytau, y_t), axis=1)  # full model

                # Numpy cov input matrix: each row of m represents a variable, and each column a single observation
                # Matlab cov input matrix: each row is an observation, and each column a variable

                reduced_model = np.linalg.det(np.cov(ytau_yt.T)) / np.linalg.det(np.cov(y_tau.T))
                full_model = np.linalg.det(np.cov(xtau_ytau_yt.T)) / np.linalg.det(np.cov(xtau_ytau.T))

                GC_xy =  0.5 * np.log(reduced_model / full_model)  # GC value
                GC[i, j] = GC_xy

                # residual sum of squares
                RSS_reduced = (n_timesteps - n_lags) * reduced_model
                RSS_full = (n_timesteps - 2*n_lags) * full_model
                F_xy = (n_timesteps - 2*n_lags) / n_lags * (RSS_reduced - RSS_full) / RSS_full
                Fstat[i, j] = F_xy

                if F_xy > threshold_F:
                    GC_sig = GC_xy

    if verbose:
        print("F statistics:", Fstat)
        print("F threshold:", threshold_F)
        print("Significant GC values:", GC)

    return GC_sig, GC, Fstat, threshold_F


def embed(signal, embed_dim, tau=1):
    """ Embed data sequence signal. Creates an embedding matrix of dimension embed_dim and tau lags.
    :param signal: vector of the signal to be embedded
    :param embed_dim: embedding dimension = np.shape(embedded_signal, 2)
    :param tau: number of lags for the embedding (keep tau=1 for GC)
    """
    signal = signal.flatten()  # should already be a vector...
    n_timesteps = len(signal)

    return np.array([signal[np.arange(0, n_timesteps, tau)[:embed_dim] + i]
                     for i in range(n_timesteps + tau - embed_dim * tau)])


def normalisa(signals):
    """ """
    signals_mean = np.mean(signals, axis=1).reshape(np.shape(signals)[0], 1)  # mean across time (i.e. per roi)
    signals_std = np.std(signals, axis=1, ddof=1).reshape(np.shape(signals)[0], 1)  # std across time (i.e. per roi);
    # ddof = 1 to have same as matlab's --> std function: normalized by N-1 --> get unbiased variance
    signals_normalized = (signals - signals_mean) / signals_std
    return signals_normalized



def multiivariateLinearCausalityTE(signals, tau=1, n_lags=5, pval=0.05, verbose=False):
    # @TODO
    """ Calculate the multivariate Granger causality between each pair of signals.
    cov([...]) --> symmetric square matrix (cov between each pair of elements)
    determinant of cov = generalized variance: measure of multi-dimensional scatter (scalar value) / linked to
    differential entropy -->  the determinant is nonzero if and only if the matrix  is invertible, and the linear map
    represented by the matrix is an isomorphism. det is zero if and only if the column vectors (or the row vectors) of
    the matrix are linearly dependent.
    see slide 32 of GC presentation
    Fstat ~ F(n_lags, n_timesteps - 2*n_lags)

    :param signals: shape n_rois x n_timesteps (/!\ Matlab opposite
    :param tau:
    :param n_lags: number of past time steps to include in model
    :param verbose: set to True to display result and threshold

    :return TE_xy: transfer entropy (GC) from roi x to roi y
    :return F_stat: F statistics of the GC test
    :return threshold_F: threshold for significance.
    """
    (n_timesteps, n_rois) = np.shape(signals)
    n_pairs = n_rois * (n_rois-1)
    # From Fstat definition: F_gc ~ F(n_lags, n_timesteps - 2*n_lags)
    threshold_F = stats.f.ppf(1 - pval/n_pairs, n_lags, n_timesteps - 2*n_lags)  # statistical threshold
    # Bonferroni corrected 1 - pval/n_pairs instead of 1 - pval: n_pairs = number of hypotheses tested,
    # pval = significance level


    signals = detrend(signals)  # removes the linear trend from the data: makes the data stationary
    signals = normalisa(signals) # Matlab normalisa: mean=0, std=1 for each ROI

    Fstat = np.zeros((n_rois, n_rois))  # matrix of all F_xy
    GC = np.zeros((n_rois, n_rois))  # matrix of all GC_xy
    GC_sig = np.zeros((n_rois, n_rois))  # matrix of all significant GC_xy (if F_xy >= threshold_F)

    for i,x in enumerate(signal.T): # for each column (each roi)
        x_embedded = embed(x, n_lags+1, tau)
        x_tau = x_embedded[:, :-1]  # past of signal x (embedded)

        for j,y in enumerate(signal.T):
            if i != j:
                y_embedded = embed(y, n_lags+1, tau)
                y_t = y_embedded[:, -1].reshape((len(y_t),1))   # current value of signal y (embedded)
                y_tau = y_embedded[:, :-1]  # past of signal y (embedded)
                xtau_ytau = np.concatenate((x_tau, y_tau), axis=1)  # both past concatenated
                ytau_yt = np.concatenate((y_tau, y_t), axis=1)  # y's past and current value --> reduced model
                xtau_ytau_yt = np.concatenate((xtau_ytau, y_t), axis=1)  # full model

                # Numpy cov input matrix: each row of m represents a variable, and each column a single observation
                # Matlab cov input matrix: each row is an observation, and each column a variable

                reduced_model = np.linalg.det(np.cov(ytau_yt.T)) / np.linalg.det(np.cov(y_tau.T))
                full_model = np.linalg.det(np.cov(xtau_ytau_yt.T)) / np.linalg.det(np.cov(xtau_ytau.T))

                GC_xy =  0.5 * np.log(reduced_model / full_model)  # GC value
                GC[i, j] = GC_xy

                # residual sum of squares
                RSS_reduced = (n_timesteps - n_lags) * reduced_model
                RSS_full = (n_timesteps - 2*n_lags) * full_model
                F_xy = (n_timesteps - 2*n_lags) / n_lags * (RSS_reduced - RSS_full) / RSS_full
                Fstat[i, j] = F_xy

                if F_xy > threshold_F:
                    GC_sig = GC_xy

    if verbose:
        print("F statistics:", Fstat)
        print("F threshold:", threshold_F)
        print("Significant GC values:", GC)

    return GC_sig, GC, Fstat, threshold_F