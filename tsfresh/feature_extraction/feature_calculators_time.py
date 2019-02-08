# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016
"""
This module contains the feature calculators that take time series as input and calculate the
values of the feature. These feature require the index of the series to be a datetime dtype.
"""

from __future__ import absolute_import, division

import numpy as np
from scipy.stats import linregress
from tsfresh.feature_extraction.feature_calculators import set_property


@set_property("fctype", "combiner")
@set_property("high_comp_cost", True)
def linear_trend_time(x, param):
    """
    Calculate a linear least-squares regression for the values of the time series.

    Possible extracted attributes are "pvalue", "rvalue", "intercept", "slope", "stderr", see the documentation of
    linregress for more information.

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param param: contains dictionaries {"attr": x} with x an string, the attribute name of the regression model
    :type param: list
    :return: the different feature values
    :return type: pandas.Series
    """
    # Get differences in seconds
    times_seconds = (x.index - x.index[0]).total_seconds()
    # Convert to minutes and eshape for linear regression
    times_minutes = np.asarray(times_seconds / 60)

    linReg = linregress(times_minutes, x.values)

    return [("attr_\"{}\"".format(config["attr"]), getattr(linReg, config["attr"]))
            for config in param]
