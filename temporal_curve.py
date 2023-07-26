"""
This module is used to estimate temporal pattern for flood modelling.
Two key functions are provided:
    - temporal_curve_24hr
    - fut_ddfs


Note:
    Current version only supports creating temporal curve for the maximum duration at 24-hours
    Raw global extreme precipitation gev pattern provided by CLIMsystems.

Ref:
    United States Department of Agriculture. Chapter 4: Storm Rainfall Depth and Distribution.
    In Part 630 Hydrology, National Engineering Handbook; 2019.
    Available online: https://directives.sc.egov.usda.gov/viewerFS.aspx?id=2572.


Dr.Chonghua Yin
20/04/2023
"""

import numpy as np
import pandas as pd
from functools import reduce
from sklearn.linear_model import LinearRegression

# specific durations for temporal curve (24-hours)
DURS_WANTED = [5, 10, 15, 30, 60, 120, 180, 360, 720, 1440]  # in minutes


def _check_miss_durations(durs_raw):
    r"""check missing durations for short and long durations from
    the wanted durations. the durations are checked for short and
    long duration. The threshold is fixed at 60min.

    Note:
        It is assumed that short durations following the log-log linear curve,
        while high durations following another log-log linear curve.
    """

    durs_diff = list(set(DURS_WANTED) - set(durs_raw))

    # separate into shorts and longs
    dur_miss_short = [item for item in durs_diff if item < 60]
    dur_miss_long = [item for item in durs_diff if item >= 60]

    return dur_miss_short, dur_miss_long


def _check_short_items(durs_raw, ddfs_raw):
    r"""extract short durations (<=60min) from original ddfs
    """
    # non-missing short duration and location
    durs_raw_short_info = [(idx, item) for idx, item in enumerate(durs_raw) if item <= 60]

    # pick non-missing duration
    durs_raw_short = [item[1] for item in durs_raw_short_info]
    durs_raw_short_loc = [item[0] for item in durs_raw_short_info]

    ddfs_raw_short = [ddfs_raw[item] for item in durs_raw_short_loc]

    return durs_raw_short, ddfs_raw_short


def _check_long_items(durs_raw, ddfs_raw):
    r"""extract long durations (>=60min) from original ddfs
    """
    durs_raw_long_info = [(idx, item) for idx, item in enumerate(durs_raw) if item >= 60]

    # pick non-missing duration
    durs_raw_long = [item[1] for item in durs_raw_long_info]
    durs_raw_long_loc = [item[0] for item in durs_raw_long_info]

    ddfs_raw_long = [ddfs_raw[item] for item in durs_raw_long_loc]

    return durs_raw_long, ddfs_raw_long


def _interp_missing(val_durs, val_ddfs, mis_durs):
    r"""linear interpolation to fill the missing durations.
    common function is applied for both of shorts and longs
    """
    X = np.log(np.array(val_durs).reshape(-1, 1))
    Y = np.log(np.array(val_ddfs).reshape(-1, 1))

    linear_regressor = LinearRegression()
    linear_regressor.fit(X, Y)

    # new values
    x_new = np.log(np.array(mis_durs).reshape(-1, 1))
    Y_pred = linear_regressor.predict(x_new)  # make predictions

    return np.exp(Y_pred)


def _fill_missings(durs_raw, ddfs_raw):
    r"""fill the missing duration based on the original ddfs.
    But the filling procedure deals with short and long duration, seperately.
    """

    if len(ddfs_raw) < 2:
        raise "At least two durations are needed"

    dur_miss_short, dur_miss_long = _check_miss_durations(durs_raw)

    # no missings and pick only wanted
    if len(dur_miss_short) == 0 and len(dur_miss_long) == 0:
        return durs_raw, ddfs_raw

    # other cases
    durs_new = durs_raw.copy()
    ddfs_new = ddfs_raw.copy()

    # fix short durations
    durs_raw_short, ddfs_raw_short = _check_short_items(durs_raw, ddfs_raw)

    if len(dur_miss_short) > 0:
        if len(durs_raw_short) > 1:
            intp_vals = _interp_missing(durs_raw_short, ddfs_raw_short, dur_miss_short)
        else:
            intp_vals = _interp_missing(durs_raw, ddfs_raw, dur_miss_short)

        # print(dur_miss_short, intp_vals)

        durs_new.extend(dur_miss_short)
        ddfs_new.extend(reduce(lambda a, b: a + b, intp_vals.tolist()))

    # fix long durations
    durs_raw_long, ddfs_raw_long = _check_long_items(durs_raw, ddfs_raw)
    if len(dur_miss_long) > 0:
        if len(durs_raw_long) > 1:
            intp_vals = _interp_missing(durs_raw_long, ddfs_raw_long, dur_miss_long)
        else:
            intp_vals = _interp_missing(durs_raw, ddfs_raw, dur_miss_long)

        # print(dur_miss_long, intp_vals)
        durs_new.extend(dur_miss_long)
        ddfs_new.extend(reduce(lambda a, b: a + b, intp_vals.tolist()))

    return durs_new, ddfs_new


def _standardize_ddfs(df_ddf):
    r""" standardize the raw ddfs to make them using the durations wanted.
    This may involve extrapolation and interpolations

    parameters
    ----------
        df_ddf: pandas.dataframe
            ddf curves with any durations. But the column for extreme precipitation
            must use the name of "ddf"

    returns
    ------------
    df_ddfs: pandas.dataframe
            with the wanted durations (hard code)

    """

    # durs_wanted = [5, 10, 15, 30, 60, 120, 180, 360, 720, 1440]
    durs_raw = df_ddf.index.to_list()
    ddfs_raw = df_ddf["ddf"].to_list()

    durs_new, ddfs_new = _fill_missings(durs_raw, ddfs_raw)
    df_ddfs = pd.DataFrame(data=ddfs_new, index=durs_new, columns=["ddf"]).sort_index()
    df_ddfs = df_ddfs.loc[df_ddfs.index <= DURS_WANTED[-1]]

    return df_ddfs


def temporal_curve_24hr(raw_ddfs: pd.DataFrame) -> pd.DataFrame:
    r"""produce temporal curve from ddf with the maximum duration = 24 hours.

    parameters
    ----------
        raw_ddfs: pandas.dataframe
            ddf curves with any durations. But the columns must contain the
            name of "ddf", while index (i.e., durations) must be in unit of minutes.

    returns
    --------
        df_ccrs: pandas.dataframe
            ccrs from 0 to 24 hours with a 0.1 hour step

    Example
    -------
        crate fake data and transform to dataframe
        durs_raw = [10,   20,   30,  60,   120,  360, 720, 1440]
        ddfs_raw = [23.2, 31.8, 38,  50.7, 66.4, 98.6, 124, 152]

        ddfs_nz = pd.DataFrame(data=ddfs_raw,
                               index=durs_raw,
                               columns=['ddf']
                              )
        ddfs_nz.index.name = "mins"

        derive temporal curve
        df_crrs = temporal_curve_24hr(ddfs_nz)

    Note:
        To create a smoother curve, the durations （in minutes） are fixed at:
                [5, 10, 15, 30, 60, 120, 180, 360, 720, 1440]
        If failed to provide these durations in the raw ddfs, an internal
        statistical method is used to extrapolate or interpolate the curve to include them.
    """

    # step 0: preprocess
    df_ddf = _standardize_ddfs(raw_ddfs)

    # step 1: calculate ratios of shorter duration to 24-hour precipitation
    df_ddf_crr = np.round(df_ddf / df_ddf.iloc[-1], 4)
    df_ddf_crr.columns = ["crr"]

    # step 2: Calculate the left part of preliminary rainfall distribution
    #         based on the ratios from raw ratios
    durations = df_ddf_crr.index[1:-1]  # only using 10min to 12-hours
    lengths = [np.round(12 - duration / 2 / 60, 4)
               for duration in sorted(durations, reverse=True)]

    crrs = [np.round(0.5 - df_ddf_crr['crr'].loc[duration] / 2, 4)
            for duration in sorted(durations, reverse=True)]

    df_steps = pd.DataFrame(data=crrs, index=lengths, columns=["crr"])

    # step 3: Determine cumulative rain ratios for times from 0.0 to 9.0 hours
    step = 0.1  # 6min

    crr6 = df_steps.loc[6].values[0]
    crr9 = df_steps.loc[9].values[0]
    # print(crr6, crr9)

    a = (2 / 3 * crr9 - crr6) / 18
    b = (crr6 - 36 * a) / 6
    # print(a, b)

    lens_to_9 = np.arange(0, 9 + step / 2, step)
    crrs_to_9 = [np.round(a * (t * t) + b * t, 4) for t in lens_to_9]

    df_crrs = pd.DataFrame(data=crrs_to_9,
                           index=lens_to_9,
                           columns=['crr']
                           )

    # Step 4.—Determine cumulative rain ratios for times from 9.0 to 10.5 hours
    crr10p5 = df_steps.loc[10.5].values[0]

    a2 = (9 / 10.5 * crr10p5 - crr9) / 13.5
    b2 = (crr9 - 81 * a2) / 9

    lens_to_10p5 = np.arange(9 + step, 10.5 + step / 2, step)
    crrs_to_10p5 = [np.round(a2 * (t * t) + b2 * t, 4) for t in lens_to_10p5]

    df_crrs_10p5 = pd.DataFrame(data=crrs_to_10p5,
                                index=lens_to_10p5,
                                columns=['crr']
                                )

    df_crrs = pd.concat([df_crrs, df_crrs_10p5], axis=0)

    # Step 5.—Determine cumulative rain ratios for times from 10.5 to 11.5 hours
    crr11 = df_steps.loc[11].values[0]
    crr11p5 = df_steps.loc[11.5].values[0]

    a3 = 2 * (crr11p5 - 2 * crr11 + crr10p5)
    b3 = crr11p5 - crr10p5 - 22 * a3
    c3 = crr11 - 121 * a3 - 11 * b3

    lens_to_11p5 = np.arange(10.5 + step, 11.5 + step / 2, step)
    crrs_to_11p5 = [np.round(a3 * (t * t) + b3 * t + c3, 4) for t in lens_to_11p5]

    df_crrs_11p5 = pd.DataFrame(data=crrs_to_11p5,
                                index=lens_to_11p5,
                                columns=['crr']
                                )

    df_crrs = pd.concat([df_crrs, df_crrs_11p5], axis=0)
    df_crrs.index = np.round(df_crrs.index, 3)

    # Step 6.— Determine cumulative rain ratios for times from 11.6 to 12.0 hours
    crr11p75 = df_steps.loc[11.75].values[0]
    crr11p875 = df_steps.loc[11.875].values[0]
    crr11p9167 = df_steps.loc[11.9167].values[0]
    crrs_11p4 = df_crrs.loc[11.4].values[0]

    # 11.6 hours
    intensity11p5 = (crr11p5 - crrs_11p4) / step
    factor11p6 = -0.867 * intensity11p5 + 0.4337

    if factor11p6 > 0.399:
        factor11p6 = 0.399

    crr11p6 = np.round(crr11p5 + factor11p6 * (crr11p75 - crr11p5), 4)

    # 11.7 hours
    factor11p7 = -0.4917 * intensity11p5 + 0.8182
    if factor11p7 > 0.799:
        factor11p7 = 0.799

    crr11p7 = np.round(crr11p5 + factor11p7 * (crr11p75 - crr11p5), 4)

    # 11.8  and 11.9 hours
    crr11p8 = np.round(crr11p75 + (11.8 - 11.75) / (11.875 - 11.75) * (crr11p875 - crr11p75), 4)
    crr11p9 = np.round(crr11p875 + (11.9 - 11.875) / (11.9167 - 11.875) * (crr11p9167 - crr11p875), 4)

    # 12.0 hours
    crr12p1 = np.round(1 - crr11p9, 4)

    crr_5min = df_ddf_crr['crr'].loc[5]
    crr_10min = df_ddf_crr['crr'].loc[10]
    crr_6min = crr_5min + 0.2 * (crr_10min - crr_5min)
    crr12 = np.round(crr12p1 - crr_6min, 4)

    # merge 11.6 to 12.0 hours
    df_crr_special = pd.DataFrame(np.array([crr11p6, crr11p7, crr11p8, crr11p9, crr12]))
    df_crr_special.index = [11.6, 11.7, 11.8, 11.9, 12.0]
    df_crr_special.columns = ["crr"]

    df_crrs = pd.concat([df_crrs, df_crr_special], axis=0)
    df_crrs.index = np.round(df_crrs.index, 3)

    # Step 7 —Determine cumulative rain ratios for times from 12.1 to 24 hours
    df_crrs_after = 1 - df_crrs.iloc[:-1]
    df_crrs_after.index = 24 - df_crrs_after.index
    df_crrs_after = df_crrs_after.sort_index()

    # step 8 - merge from 0 to 24 hours with a 0.1 hour step
    df_crrs = pd.concat([df_crrs, df_crrs_after], axis=0)
    df_crrs.index = np.round(df_crrs.index, 1)
    df_crrs.index.name = "hours"

    # return
    return df_crrs


def _interp_gev_pattern_to_hist_duration(raw_gev_pattern: pd.DataFrame,
                                         his_ddfs: pd.DataFrame
                                         ) -> pd.DataFrame:
    r"""produce temporal curve from ddf with the maximum duration = 24 hours.

    parameters
    ----------
         raw_gev_pattern: pandas.dataframe
            gev pattern from AR6 GCMs, unit in %/degC.
            Note: The columns must contain the name of "cf",
                   while index (i.e., durations) must in unit of minutes.

        his_ddfs: pandas.dataframe
            ddf curves with any durations.
            Note: the columns must contain the name of "ddf",
            while index (i.e., durations) must in unit of minutes.

    returns
    --------
        df_intp_cf: pandas.dataframe
            gev patterns at the durations from historical DDFs
    """

    offset = 1000.0  # big enough to avoid minus values
    gev_durs = raw_gev_pattern.index
    gev_cfs = raw_gev_pattern['cf'].values + offset
    his_durs = his_ddfs.index
    his_cfs = _interp_missing(gev_durs, gev_cfs, his_durs) - offset

    df_intp_cf = pd.DataFrame(data=his_cfs, index=his_durs, columns=["cf"]).sort_index()
    df_intp_cf.index.name = his_ddfs.index.name

    return df_intp_cf


def future_ddfs(dgmt: float,
                raw_gev_pattern: pd.DataFrame,
                his_ddfs: pd.DataFrame
                ) -> pd.DataFrame:
    r"""produce future ddfs from gev pattern and historical ddfs.

    parameters
    ----------
         dgmt: float:
            global mean temperature change in degC

         raw_gev_pattern: pandas.dataframe
            normalized gev pattern from AR6 GCMs unit in %/degC.
            Note: The columns must contain the name of "cf",
                   while index (i.e., durations) must be in unit of minutes.

        his_ddfs: pandas.dataframe
            ddf curves with any durations.
            Note: the columns must contain the name of "ddf",
            while index (i.e., durations) must be in unit of minutes.

    returns
    --------
        fut_ddfs: pandas.dataframe
            future DDFs at the durations from historical DDFs
    """

    df_cf_intp = _interp_gev_pattern_to_hist_duration(raw_gev_pattern, his_ddfs)
    fut_ddfs = (his_ddfs['ddf'] * (1.0 + df_cf_intp['cf'] * dgmt / 100)).round(1).to_frame()
    fut_ddfs.columns = ['ddf']
    fut_ddfs.index.name = his_ddfs.index.name

    return fut_ddfs
