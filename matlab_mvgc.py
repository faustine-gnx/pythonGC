# http://erramuzpe.github.io/C-PAC/blog/2015/06/10/multivariate-granger-causality-in-python-for-fmri-timeseries-analysis/
import numpy as np
from matplotlib import pylab

"""
The steps to calculate MVGC are the following: We first have to compute the autocovariance matrix from the timeseries. 
After that, extract the coefficients with the VAR modelling and finally, perform the calculation of the MVGC.
https://users.sussex.ac.uk/~lionelb/MVGC/html/mvgcfuncref.html
"""


def tsdata_to_autocov(signals, n_lags):
    """
    Calculate sample autocovariance sequence from (presumed stationary) time series data.

    Note: This routine is discouraged for VAR numerical modelling, and is only included for completeness; sample
    autocovariances are notoriously noisy and biased (but see the experimental tsdata_to_autocov_debias). The
    recommended practice is to estimate a VAR model via tsdata_to_var and then calculate autocovariance via
    var_to_autocov.

    :param signals: time series data
    :param n_lags: number of lags

    :return autocov: n_lags-lag sample autocovariance sequence
    """
    if len(signals.shape) == 2:
        signal = np.expand_dims(signals, axis=2)
        [n, m, N] = np.shape(signals)
    else:
        [n, m, N] = np.shape(signals)
    signal = pylab.demean(signals, axis=1)
    autocov = np.zeros((n, n, (n_lags+1)))

    for k in range(n_lags+1):
        M = N * (m-k)
        autocov[:, :, k] = np.dot(np.reshape(signals[:, k:m, :], (n, M)), np.reshape(signals[:, 0:m-k, :], (n, M)).conj().T) / M-1

    return autocov


def autocov_to_mvgc(autocov, x_indices, y_indices):
    """
    Calculate conditional time-domain MVGC (multivariate Granger causality): X --> Y | Z.
    https://github.com/SacklerCentre/MVGC2/blob/master/gc/autocov/autocov_to_mvgc.m

    :param autocov: autocovariance sequence
    :param x_indices: vector of indices of source (causal) multi-variable
    :param y_indices: vector of indices of target (causee) multi-variable

    :return GC: time-domain multivariate Granger causality from the variable X (specified by x_indices) to the variable
    Y (specified by y_indices), conditional on all other variables Z represented in autocov, for a stationary VAR
    process with autocovariance sequence autocov.
    """
    n = autocov.shape[0]

    z_indices = np.arange(n)
    z_indices = np.delete(z_indices, [np.array(np.hstack((y_indices, x_indices)))])
    # indices of other variables (to condition out)
    yz_indices = np.array(np.hstack((y_indices, z_indices)))
    xzy_indices = np.array(np.hstack((yz_indices, x_indices)))
    F = 0

    # full regression
    ixgrid1 = np.ix_(xzy_indices, xzy_indices)
    [AF, SIG] = autocov_to_var(autocov[ixgrid1])

    # reduced regression
    ixgrid2 = np.ix_(yz_indices, yz_indices)
    [AF, SIGR] = autocov_to_var(autocov[ixgrid2])

    ixgrid3 = np.ix_(y_indices, y_indices)
    GC = np.log(np.linalg.det(SIGR[ixgrid3])) - np.log(np.linalg.det(SIG[ixgrid3]))

    return GC


def autocov_to_var(autocov):
    """
    Calculate VAR parameters (regression coefficients and residuals covariance matrix) from autocovariance sequence.
    :param autocov:
    :return coeffs_F: VAR coefficient matrix
    :return SIG: residuals covariance matrix
    """
    [n ,m, q1] = autocov.shape
    q = q1 - 1
    qn = q * n
    autocov_0 = autocov[:, :, 0]
    # covariance
    autocov_F = np.reshape(autocov[:, :, 1:], (n, qn)).T
    # backward autocov sequence
    autocov_B = np.reshape(np.transpose(autocov[:, :, 1:], (0, 2, 1)), (qn, n))

    # forward  coefficients
    coeffs_F = np.zeros([n, qn])
    # backward coefficients (reversed compared with Whittle's treatment)
    coeffs_B = np.zeros([n, qn])

    # initialise recursion
    k = 1 # model order
    r = q-k
    kf = np.arange(k*n)
    # forward  indices
    kb = np.arange(r*n, qn)
    # backward indices
    coeffs_F[:, kf] = np.dot(autocov_B[kb, :], np.linalg.inv(autocov_0))
    coeffs_B[:, kb] = np.dot(autocov_F[kf, :], np.linalg.inv(autocov_0))

    for k in np.arange(2, q+1):

        DF = autocov_B[(r-1)*n+1:r*n, :] - np.dot(coeffs_F[:, kf], autocov_B[kb, :])
        VB = autocov_0 - np.dot(coeffs_B[:, kb], autocov_B[kb, :])

        AAF = np.dot(DF, np.linalg.inv(VB))  # DF/VB

        DB = autocov_F[(k-1)*n+1:k*n, :] - np.dot(coeffs_B[:, kb], autocov_F[kf, :])
        VF = np.dot(autocov_0-coeffs_F[:, kf], autocov_F[kf, :])

        AAB = np.dot(DB, np.linalg.inv(VF))  # DB/VF

        AFPREV = coeffs_F[:, kf-1]
        ABPREV = coeffs_B[:, kb-1]
        r = q-k
        kf = np.arange(1, (np.dot(k, n))+1)
        kb = np.arange(np.dot(r, n)+1, (qn) + 1)
        coeffs_F[:, kf-1] = np.array(np.hstack((AFPREV-np.dot(AAF, ABPREV), AAF)))
        coeffs_B[:, kb-1] = np.array(np.hstack((AAB, ABPREV-np.dot(AAB, AFPREV))))

    SIG = autocov_0-np.dot(coeffs_F, autocov_F)
    coeffs_F = np.reshape(coeffs_F, (n, n, q))
    return [coeffs_F, SIG]
