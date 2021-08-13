# http://erramuzpe.github.io/C-PAC/blog/2015/06/10/multivariate-granger-causality-in-python-for-fmri-timeseries-analysis/
import numpy as np
from matplotlib import pylab

def tsdata_to_autocov(X, q):
    if len(X.shape) == 2:
        X = np.expand_dims(X, axis=2)
        [n, m, N] = np.shape(X)
    else:
        [n, m, N] = np.shape(X)
    X = pylab.demean(X, axis=1)
    G = np.zeros((n, n, (q+1)))

    for k in range(q+1):
        M = N * (m-k)
        G[:,:,k] = np.dot(np.reshape(X[:,k:m,:], (n, M)), np.reshape(X[:,0:m-k,:], (n, M)).conj().T) / M-1
    return G


def autocov_to_mvgc(G, x, y):
    n = G.shape[0]

    z = np.arange(n)
    z = np.delete(z, [np.array(np.hstack((x, y)))])
    # indices of other variables (to condition out)
    xz = np.array(np.hstack((x, z)))
    xzy = np.array(np.hstack((xz, y)))
    F = 0

    # full regression
    ixgrid1 = np.ix_(xzy, xzy)
    [AF, SIG] = autocov_to_var(G[ixgrid1])

    # reduced regression
    ixgrid2 = np.ix_(xz, xz)
    [AF, SIGR] = autocov_to_var(G[ixgrid2])

    ixgrid3 = np.ix_(x, x)
    F = np.log(np.linalg.det(SIGR[ixgrid3])) - np.log(np.linalg.det(SIG[ixgrid3]))
    return F

def autocov_to_var(G):
    [n,m,q1] = G.shape
    q = q1 - 1
    qn = q * n
    G0 = G[:,:,0]
    # covariance
    GF = np.reshape(G[:,:,1:], (n, qn)).T
    # backward autocov sequence
    GB = np.reshape(np.transpose(G[:,:,1:], (0, 2, 1)), (qn, n))

    # forward  coefficients
    AF = np.zeros([n, qn])
    # backward coefficients (reversed compared with Whittle's treatment)
    AB = np.zeros([n, qn])

    # initialise recursion
    k = 1 # model order
    r = q-k
    kf = np.arange(k*n)
    # forward  indices
    kb = np.arange(r*n, qn)
    # backward indices
    AF[:,kf] = np.dot(GB[kb,:],np.linalg.inv(G0))
    AB[:,kb] = np.dot(GF[kf,:],np.linalg.inv(G0))

    for k in np.arange(2, q+1):

        DF = GB[(r-1)*n+1:r*n,:] - np.dot(AF[:,kf],GB[kb,:])
        VB = G0 - np.dot(AB[:,kb],GB[kb,:])

        AAF = np.dot(DF,np.linalg.inv(VB)); # DF/VB

        DB = GF[(k-1)*n+1:k*n,:] - np.dot(AB[:,kb],GF[kf,:])
        VF = np.dot(G0-AF[:,kf],GF[kf,:])

        AAB = np.dot(DB,np.linalg.inv(VF)); # DB/VF

        AFPREV = AF[:,kf-1]
        ABPREV = AB[:,kb-1]
        r = q-k
        kf = np.arange(1, (np.dot(k, n))+1)
        kb = np.arange(np.dot(r, n)+1, (qn)+1)
        AF[:,kf-1] = np.array(np.hstack((AFPREV-np.dot(AAF, ABPREV), AAF)))
        AB[:,kb-1] = np.array(np.hstack((AAB, ABPREV-np.dot(AAB, AFPREV))))

    SIG = G0-np.dot(AF, GF)
    AF = np.reshape(AF, (n, n, q))
    return [AF, SIG]
