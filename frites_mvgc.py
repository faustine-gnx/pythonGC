import numpy as np
from scipy.special import ndtri, psi


def cmi_nd_ggg(x, y, z, mvaxis=None, traxis=-1, biascorrect=False,
               demeaned=False, shape_checking=True):
    """Multi-dimentional MI between three Gaussian variables in bits.
    This function is based on ANOVA style model comparison.
    Parameters
    ----------
    x, y, z : array_like
        Arrays to consider for computing the Mutual Information. The three
        input variables x, y and z should have the same shape except on the
        mvaxis (if needed).
    mvaxis : int | None
        Spatial location of the axis to consider if multi-variate analysis
        is needed
    traxis : int | -1
        Spatial location of the trial axis. By default the last axis is
        considered
    biascorrect : bool | True
        Specifies whether bias correction should be applied to the estimated MI
        # if true --> negative values for GC
    demeaned : bool | False
        Specifies whether the input data already has zero mean (true if it has
        been copula-normalized)
    shape_checking : bool | True
        Perform a reshape and check that x and y shapes are consistents. For
        high performances and to avoid extensive memory usage, it's better to
        already have x and y with a shape of (..., mvaxis, traxis) and to set
        this parameter to False
    Returns
    -------
    mi : array_like
        The mutual information with the same shape as x, y and z without the
        mvaxis and traxis
    """
    # print(np.shape(x), np.shape(y), np.shape(z))

    # Multi-dimentional shape checking
    if shape_checking:
        #         x = nd_reshape(x, mvaxis=mvaxis, traxis=traxis)
        #         y = nd_reshape(y, mvaxis=mvaxis, traxis=traxis)
        #         z = nd_reshape(z, mvaxis=mvaxis, traxis=traxis)
        nd_shape_checking(x, y, mvaxis, traxis)
        nd_shape_checking(x, z, mvaxis, traxis)

    # x.shape == y.shape == z.shape (..., x_mvaxis, traxis)
    ntrl = x.shape[-1]
    nvarx, nvary, nvarz = x.shape[-2], y.shape[-2], z.shape[-2]
    nvarxy = nvarx + nvary
    nvaryz = nvary + nvarz
    nvarxy = nvarx + nvary
    nvarxz = nvarx + nvarz
    nvarxyz = nvarx + nvaryz

    # joint variable along the mvaxis
    xyz = np.concatenate((x, y, z), axis=-2)
    if not demeaned:
        xyz -= xyz.mean(axis=-1, keepdims=True)
    cxyz = np.einsum('...ij, ...kj->...ik', xyz, xyz)  # covariance
    cxyz /= float(ntrl - 1.)

    # submatrices of joint covariance
    cz = cxyz[..., nvarxy:, nvarxy:]
    cyz = cxyz[..., nvarx:, nvarx:]
    sh = list(cxyz.shape)
    sh[-1], sh[-2] = nvarxz, nvarxz
    cxz = np.zeros(tuple(sh), dtype=float)
    cxz[..., :nvarx, :nvarx] = cxyz[..., :nvarx, :nvarx]
    cxz[..., :nvarx, nvarx:] = cxyz[..., :nvarx, nvarxy:]
    cxz[..., nvarx:, :nvarx] = cxyz[..., nvarxy:, :nvarx]
    cxz[..., nvarx:, nvarx:] = cxyz[..., nvarxy:, nvarxy:]

    # Cholesky decomposition
    chcz = np.linalg.cholesky(cz)
    chcxz = np.linalg.cholesky(cxz)
    chcyz = np.linalg.cholesky(cyz)
    chcxyz = np.linalg.cholesky(cxyz)

    # entropies in nats
    # normalizations cancel for mutual information
    hz = np.log(np.einsum('...ii->...i', chcz)).sum(-1)
    hxz = np.log(np.einsum('...ii->...i', chcxz)).sum(-1)
    hyz = np.log(np.einsum('...ii->...i', chcyz)).sum(-1)
    hxyz = np.log(np.einsum('...ii->...i', chcxyz)).sum(-1)

    ln2 = np.log(2)
    if biascorrect:
        vec = np.arange(1, nvarxyz + 1)
        psiterms = psi((ntrl - vec).astype(float) / 2.0) / 2.0
        dterm = (ln2 - np.log(ntrl - 1.0)) / 2.0
        hz = hz - nvarz * dterm - psiterms[:nvarz].sum()
        hxz = hxz - nvarxz * dterm - psiterms[:nvarxz].sum()
        hyz = hyz - nvaryz * dterm - psiterms[:nvaryz].sum()
        hxyz = hxyz - nvarxyz * dterm - psiterms[:nvarxyz].sum()

    # MI in bits
    i = (hxz + hyz - hxyz - hz) / ln2
    return i

def multi_gc_frites(signals, n_lags=5, tau=1, t0=[0], dt=None, conditional=True):
    """
    Note: can be used for bivariate GC by setting conditional to False"""
    tau = 1
    if dt is None:
        dt = np.shape(signals)[1]
    rows, cols = np.mgrid[0:n_lags + 1, 0:dt]
    rows = rows[::tau, :]
    cols = cols[::tau, :]
    ind_tx = cols - rows  # lagged matrix

    # 100 * (ind_tx.shape[0] / (step * ind_tx.shape[1])) should be <10-15


def _cond_gccovgc(signals, x_idx, y_idx, ind_tx, conditional=True):
    """ Compute the Gaussian-Copula based covGC for a single pair.
    This function computes the covGC for a single pair, across multiple trials,
    at different time indices.
    :param signals: variables in rows and observations in columns --> shape = n_rois x n_timesteps
    :param
    """
    d_x, d_y = signals[x_idx, :], signals[y_idx, :]
    n_lags, n_dt = ind_tx.shape
    gc = np.empty(3, dtype=d_x.dtype, order='C')
    # define z past
    z_indices = np.array([k for k in range(signals.shape[0]) if k not in [x_idx, y_idx]])
    d_z = signals[z_indices, :]  # other roi selection
    rsh = int(len(z_indices) * (n_lags - 1))  # roi_range = 150-2 = 148; n_lags-1 = 5 --> rsh = 740

    x = d_x[ind_tx]
    y = d_y[ind_tx]
    # temporal selection
    x_pres, x_past = x[0], x[1:]
    y_pres, y_past = y[0], y[1:]
    xy_past = np.concatenate((x[1:], y[1:]), axis=0)
    # conditional granger causality case
    if conditional:
        # condition by the past of every other possible sources
        z_past = d_z[..., ind_tx[1:, :]]  # (lag_past, dt) selection
        z_past = z_past.reshape(rsh, n_dt)
        # cat with past
        yz_past = np.concatenate((y_past, z_past), axis=0)
        xz_past = np.concatenate((x_past, z_past), axis=0)
        xyz_past = np.concatenate((xy_past, z_past), axis=0)
    else:
        yz_past, xz_past, xyz_past = y_past, x_past, xy_past

    # copnorm over the last axis (avoid copnorming several times)
    x_pres = copnorm_nd(x_pres, axis=-1)
    x_pres = np.expand_dims(x_pres, 0)
    x_past = copnorm_nd(x_past, axis=-1)
    y_pres = copnorm_nd(y_pres, axis=-1)
    y_pres = np.expand_dims(y_pres, 0)
    y_past = copnorm_nd(y_past, axis=-1)
    yz_past = copnorm_nd(yz_past, axis=-1)
    xz_past = copnorm_nd(xz_past, axis=-1)
    xyz_past = copnorm_nd(xyz_past, axis=-1)

    # -----------------------------------------------------------------
    # Granger Causality measures
    # -----------------------------------------------------------------
    gc[0] = cmi_nd_ggg(y_pres, x_past, yz_past)
    # gc(pairs(:,2) -> pairs(:,1))
    gc[1] = cmi_nd_ggg(x_pres, y_past, xz_past)
    # gc(pairs(:,2) . pairs(:,1))
    gc[2] = cmi_nd_ggg(x_pres, y_pres, xyz_past)
    return gc


def copnorm_nd(x, axis=-1):
    """Copula normalization for a multidimentional array.
    Parameters
    ----------
    x : array_like
        Array of data
    axis : int | -1
        Epoch (or trial) axis. By default, the last axis is considered
    Returns
    -------
    cx : array_like
        Standard normal samples with the same empirical CDF value as the input.
    """
    assert isinstance(x, np.ndarray) and (x.ndim >= 1)
    return np.apply_along_axis(copnorm_1d, axis, x)


def copnorm_1d(x):
    """Copula normalization for a single vector.
    Parameters
    ----------
    x : array_like
        Array of data of shape (n_epochs,)
    Returns
    -------
    cx : array_like
        Standard normal samples with the same empirical CDF value as the input.
    """
    assert isinstance(x, np.ndarray) and (x.ndim == 1)
    return ndtri(ctransform(x))


def ctransform(x):
    """Copula transformation (empirical CDF).
    Parameters
    ----------
    x : array_like
        Array of data. The trial axis should be the last one
    Returns
    -------
    xr : array_like
        Empirical CDF value along the last axis of x. Data is ranked and scaled
        within [0 1] (open interval)
    """
    xr = np.argsort(np.argsort(x)).astype(float)
    xr += 1.
    xr /= float(xr.shape[-1] + 1)
    return xr


# def nd_reshape(x, mvaxis=None, traxis=-1):
#     """Multi-dimentional reshaping.
#     This function is used to be sure that an nd array has a correct shape
#     of (..., mvaxis, traxis).
#     Parameters
#     ----------
#     x : array_like
#         Multi-dimentional array
#     mvaxis : int | None
#         Spatial location of the axis to consider if multi-variate analysis
#         is needed
#     traxis : int | -1
#         Spatial location of the trial axis. By default the last axis is
#         considered
#     Returns
#     -------
#     x_rsh : array_like
#         The reshaped multi-dimentional array of shape (..., mvaxis, traxis)
#     """
#     assert isinstance(traxis, int)
#     traxis = np.arange(x.ndim)[traxis]

#     # Create an empty mvaxis axis
#     if not isinstance(mvaxis, int):
#         x = x[..., np.newaxis]
#         mvaxis = -1
#     assert isinstance(mvaxis, int)
#     mvaxis = np.arange(x.ndim)[mvaxis]

#     # move the multi-variate and trial axis
#     x = np.moveaxis(x, (mvaxis, traxis), (-2, -1))

#     return x


def nd_shape_checking(x, y, mvaxis, traxis):
    """Check that the shape between two ndarray is consitent.
    x.shape = (nx_1, ..., n_xn, x_mvaxis, traxis)
    y.shape = (nx_1, ..., n_xn, y_mvaxis, traxis)
    """
    assert x.ndim == y.ndim
    dims = np.delete(np.arange(x.ndim), -2)
    assert all([x.shape[k] == y.shape[k] for k in dims])