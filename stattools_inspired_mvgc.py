import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.sm_exceptions import InfeasibleTestError
from statsmodels.tools.tools import add_constant
from statsmodels.tools.validation import (
    array_like,
    bool_like,
    dict_like,
    float_like,
    int_like,
    string_like,
)
from statsmodels.tsa.tsatools import add_trend, lagmat, lagmat2ds


def multigrangercausalitytests(x, maxlag, addconst=True, verbose=True):
    x = array_like(x, "x", ndim=2)
    if not np.isfinite(x).all():
        raise ValueError("x contains NaN or inf values.")
    addconst = bool_like(addconst, "addconst")
    verbose = bool_like(verbose, "verbose")
    try:
        maxlag = int_like(maxlag, "maxlag")
        if maxlag <= 0:
            raise ValueError("maxlag must a a positive integer")
        lags = np.arange(1, maxlag + 1)
    except TypeError:
        lags = np.array([int(lag) for lag in maxlag])
        maxlag = lags.max()
        if lags.min() <= 0 or lags.size == 0:
            raise ValueError(
                "maxlag must be a non-empty list containing only "
                "positive integers"
            )

    if x.shape[0] <= 3 * maxlag + int(addconst):
        raise ValueError(
            "Insufficient observations. Maximum allowable "
            "lag is {0}".format(int((x.shape[0] - int(addconst)) / 3) - 1)
        )

    resli = {}

    n_rois = len(x.T)
    F_stat1 = np.zeros((n_rois, n_rois, len(lags)))
    F_stat2 = np.zeros((n_rois, n_rois, len(lags)))

    for lag_idx, mxlg in enumerate(lags):
        result = {}
        if verbose:
            print("\nGranger Causality")
            print("number of lags (no zero)", mxlg)

        # create lagmat of both time series
        dta = lagmat2ds(x, mxlg, trim="both")  # , dropex=1)

        for i in range(n_rois):
            for j in range(n_rois):
                if i != j:

                    # add constant
                    xyz_past = dta[:, [k * (mxlg + 1) + 1 + m for k in range(n_rois) for m in range(mxlg)]]
                    yz_past = np.delete(xyz_past, [i * mxlg + m for m in range(mxlg)], 1)

                    dtaown = add_constant(yz_past, prepend=False)
                    dtajoint = add_constant(xyz_past, prepend=False)

                    if (
                            dtajoint.shape[1] == (dta.shape[1] - 1)
                            or (dtajoint.max(0) == dtajoint.min(0)).sum() != 1
                    ):
                        raise InfeasibleTestError(
                            "The x values include a column with constant values and so"
                            " the test statistic cannot be computed."
                        )

                    # Run ols on both models without and with lags of second variable
                    res2down = OLS(dta[:, j * mxlg], dtaown).fit()
                    res2djoint = OLS(dta[:, j * mxlg], dtajoint).fit()

                    # print results
                    # for ssr based tests see:
                    # http://support.sas.com/rnd/app/examples/ets/granger/index.htm
                    # the other tests are made-up

                    # Granger Causality test using ssr (F statistic)
                    tss = res2djoint.centered_tss
                    print(
                        tss, res2djoint.ssr, res2down.ssr
                    )

                    if (
                            tss == 0
                            or res2djoint.ssr == 0
                            or np.isnan(res2djoint.rsquared)
                            # or (res2djoint.ssr / tss) < np.finfo(float).eps
                            or res2djoint.params.shape[0] != dtajoint.shape[1]
                    ):
                        raise InfeasibleTestError(
                            "The Granger causality test statistic cannot be compute "
                            "because the VAR has a perfect fit of the data."
                        )
                    fgc1 = (
                            (res2down.ssr - res2djoint.ssr)
                            / res2djoint.ssr
                            / mxlg
                            * res2djoint.df_resid   # 1688
                    )

                    fgc2 = (
                            (res2down.ssr - res2djoint.ssr)
                            / res2djoint.ssr
                            / mxlg
                            * (len(x) - 2 * mxlg)  # 1734
                    )

                    # --> df_denom: no diff in significance (because lots of time points)

                    if verbose:
                        print(
                            "ssr based F test:         their F=%-8.4f, my F=%-8.4f their p=%-8.4f, my p=%-8.4f, df_denom=%d,"
                            " df_num=%d"
                            % (
                                fgc1,
                                fgc2,
                                stats.f.sf(fgc1, mxlg, res2djoint.df_resid),
                                stats.f.sf(fgc2, mxlg, (len(x) - 2 * mxlg)), # lower pval
                                res2djoint.df_resid,
                                mxlg,
                            )
                        )
                    result["ssr_ftest"] = (
                        fgc1,
                        fgc2,
                        stats.f.sf(fgc1, mxlg, res2djoint.df_resid),
                        stats.f.sf(fgc2, mxlg, (len(x) - 2 * mxlg)),
                        res2djoint.df_resid,
                        mxlg,
                    )

                    # F test that all lag coefficients of exog are zero
                    rconstr = np.column_stack(
                        (np.zeros((mxlg, mxlg)), np.eye(mxlg, mxlg), np.zeros((mxlg, 1)))
                    )
                    #                     ftres = res2djoint.f_test(rconstr)
                    #                     if verbose:
                    #                         print(
                    #                             "parameter F test:         F=%-8.4f, p=%-8.4f, df_denom=%d,"
                    #                             " df_num=%d"
                    #                             % (ftres.fvalue, ftres.pvalue, ftres.df_denom, ftres.df_num)
                    #                         )
                    #                     result["params_ftest"] = (
                    #                         np.squeeze(ftres.fvalue)[()],
                    #                         np.squeeze(ftres.pvalue)[()],
                    #                         ftres.df_denom,
                    #                         ftres.df_num,
                    #                     )

                    resli[mxlg] = (result, [res2down, res2djoint, rconstr])
                    F_stat1[i, j, lag_idx] = fgc1
                    F_stat2[i, j, lag_idx] = fgc2

    return resli