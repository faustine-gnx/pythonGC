import numpy as np
from scipy import signal, stats
from utils import lag_signals  # , normalisa

# http://www.scholarpedia.org/article/Granger_causality


def bivariateLinearCausalityTE(signals, n_lags=5, pval=0.01, tau=1, verbose=False):
    """ Calculate the bivariate Granger causality between each pair of signals.
    cov([...]) --> symmetric square matrix (cov between each pair of elements)
    determinant of cov = generalized variance: measure of multi-dimensional scatter (scalar value) / linked to
    differential entropy -->  the determinant is nonzero if and only if the matrix  is invertible, and the linear map
    represented by the matrix is an isomorphism. det is zero if and only if the column vectors (or the row vectors) of
    the matrix are linearly dependent.
    see slide 32 of GC presentation
    Fstat ~ F(n_lags, n_timesteps - 2*n_lags)

    Numpy cov input matrix: each row of m represents a variable, and each column a single observation
    Matlab cov input matrix: each row is an observation, and each column a variable

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

    # signals = signal.detrend(signals)  # removes the linear trend from each signal: makes the data stationary
    # signals = normalisa(signals)  # Matlab normalisa: mean=0, std=1 for each ROI
    # normalization is useless: it does not change GC nor Fstat results
    # detrend --> slightly different GC & Fstat

    Fstat = np.zeros((n_rois, n_rois))  # matrix of all F_xy
    GC = np.zeros((n_rois, n_rois))  # matrix of all GC_xy
    GC_sig = np.zeros((n_rois, n_rois))  # matrix of all significant GC_xy (if F_xy >= threshold_F)

    signals_lagged = lag_signals(signals, n_lags, tau=1)

    for i, x in enumerate(signals):  # for each column (each roi)
        x_lagged = signals_lagged[i]
        x_past = x_lagged[:, :-1]  # past of signal x (lagged)

        for j, y in enumerate(signals):
            if i != j:
                y_lagged = signals_lagged[j]
                y_present = np.expand_dims(y_lagged[:, -1], axis=1)  # current value of signal y (lagged)
                y_past = y_lagged[:, :-1]  # past of signal y (lagged)
                xy_past = np.concatenate((x_past, y_past), axis=1)  # both past concatenated
                reduced_model = np.concatenate((y_past, y_present), axis=1)  # y's past and current value
                full_model = np.concatenate((xy_past, y_present), axis=1)  # x and y's past and current y value

                # Covariances
                sigma_reduced = np.linalg.det(np.cov(reduced_model.T)) / np.linalg.det(np.cov(y_past.T))
                sigma_full = np.linalg.det(np.cov(full_model.T)) / np.linalg.det(np.cov(xy_past.T))
                # FRITES: compute entropy using the slogdet in numpy rather than np.linalg.det
                #         nb: the entropy is the logdet ***

                GC_xy =  0.5 * np.log(sigma_reduced / sigma_full)  # GC value
                GC[i, j] = GC_xy

                # residual sum of squares
                RSS_reduced = (n_timesteps - n_lags) * sigma_reduced
                RSS_full = (n_timesteps - 2*n_lags) * sigma_full
                F_xy = (n_timesteps - 2*n_lags) / n_lags * (RSS_reduced - RSS_full) / RSS_full
                Fstat[i, j] = F_xy

                if F_xy > threshold_F:
                    GC_sig[i,j] = GC_xy

    if verbose:
        print("F statistics:", Fstat)
        print("F threshold:", threshold_F)
        print("Significant GC values:", GC)

    return GC_sig, GC, Fstat, threshold_F

