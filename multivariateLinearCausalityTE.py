import numpy as np
from scipy import signal, stats
from utils import lag_signals  # , normalisa

# http://www.scholarpedia.org/article/Granger_causality


def multivariateLinearCausalityTE(signals, n_lags=5, pval=0.01, tau=1, verbose=False):
    """ Calculate the bivariate Granger causality between each pair of signals.
    cov([...]) --> symmetric square matrix (cov between each pair of elements)
    determinant of cov = generalized variance: measure of multi-dimensional scatter (scalar value) / linked to
    differential entropy -->  the determinant is nonzero if and only if the matrix  is invertible, and the linear map
    represented by the matrix is an isomorphism. det is zero if and only if the column vectors (or the row vectors) of
    the matrix are linearly dependent.
    see slide 32 of GC presentation
    Fstat ~ F(n_lags, n_timesteps - 2*n_lags)

    :param signals: variables in rows and observations in columns --> shape = n_rois x n_timesteps (/!\ Matlab opposite)
    :param n_lags: number of past time steps to include in model (order)
    :param pval: significance level for the F test. The lower it is, the higher threshold_F (does not change GC)
    :param tau: number of time steps between lags --> keep past values at times: [t-tau*i for i in range(n_lags)]
           (tau=1 for GC: keep all values up to n_lags, don't skip any)
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

    # In stattools gc:
    # threshold_F = stats.f.sf
    # cdf(F-function, dfn, dfd): Cumulative distribution function. proba that the variable is LESS than or equal to x
    # sf(F-function, dfn, dfd): Survival function (also defined as 1 - cdf, but sf is sometimes more accurate). also
    # called reliability function --> probability that the variate takes a value GREATER than x
    # ppf(desired pval, dfn, dfd): Percent point function (inverse of cdf -> percentiles) --> for a distribution
    # function we calculate the probability that the variable is LESS than or equal to x for a given x

    signals = signal.detrend(signals)  # removes the linear trend from each signal: makes the data stationary
    # signals = normalisa(signals)  # Matlab normalisa: mean=0, std=1 for each ROI
    # normalization is useless: it does not change GC nor Fstat results
    # detrend --> slightly different GC & Fstat

    Fstat = np.zeros((n_rois, n_rois))  # matrix of all F_xy
    GC = np.zeros((n_rois, n_rois))  # matrix of all GC_xy
    GC_sig = np.zeros((n_rois, n_rois))  # matrix of all significant GC_xy (if F_xy >= threshold_F)

    signals_lagged = lag_signals(signals, n_lags, tau=1)

    for i, x in enumerate(signals):  # for each column (each roi)
        x_lagged = signals_lagged[i]
        x_tau = x_lagged[:, :-1]  # past of signal x (lagged)

        for j, y in enumerate(signals):
            if i != j:
                y_lagged = signals_lagged[j]
                y_t = y_lagged[:, -1].reshape((len(y_lagged), 1))  # current value of signal y (lagged)
                y_tau = y_lagged[:, :-1]  # past of signal y (lagged)
                xtau_ytau = np.concatenate((x_tau, y_tau), axis=1)  # both past concatenated
                ytau_yt = np.concatenate((y_tau, y_t), axis=1)  # y's past and current value --> reduced model
                xtau_ytau_yt = np.concatenate((xtau_ytau, y_t), axis=1)  # full model

                small = min(i, j)
                large = max(i, j)
                z_indices = np.r_[0:small, small + 1:large, large + 1:n_rois]
                z_lagged = signals_lagged[z_indices]

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
                    GC_sig[i,j] = GC_xy

    if verbose:
        print("F statistics:", Fstat)
        print("F threshold:", threshold_F)
        print("Significant GC values:", GC)

    return GC_sig, GC, Fstat, threshold_F



def entr(xy):
    """Entropy of a gaussian variable.
    This function computes the entropy of a gaussian variable for a 2D input.
    """
    # manually compute the covariance (faster)
    n_r, n_c = xy.shape
    xy = xy - xy.mean(axis=1, keepdims=True)
    out = np.empty((n_r, n_r), xy.dtype, order='C')
    np.dot(xy, xy.T, out=out)
    out /= (n_c - 1)
    # compute entropy using the slogdet in numpy rather than np.linalg.det
    # nb: the entropy is the logdet
    (sign, h) = np.linalg.slogdet(out)
    if not sign > 0:
        raise ValueError(f"Can't estimate the entropy properly of the input "
                         f"matrix of shape {xy.shape}. Try to increase the "
                         "step")

    return h