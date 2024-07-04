"""
Time-series NDVI Analysis Module for Salzburg, Austria

This module contains functions for calculating NDVI, plotting NDVI time-series data,
and analyzing trends in NDVI data over time.

Dependencies:
- numpy
- pandas
- matplotlib
- seaborn
- scipy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def calculate_ndvi(nir, red):
    """
    Calculate the NDVI from NIR and red bands.

    Parameters:
    nir (numpy.ndarray): Near-infrared band data.
    red (numpy.ndarray): Red band data.

    Returns:
    numpy.ndarray: NDVI values.
    """
    ndvi = (nir - red) / (nir + red)
    return ndvi

def plot_ndvi_timeseries(dates, ndvi_values):
    """
    Plot NDVI time-series data.

    Parameters:
    dates (list): List of dates corresponding to the NDVI values.
    ndvi_values (list): List of NDVI values.

    Returns:
    None
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, ndvi_values, marker='o', linestyle='-', color='green')
    plt.xlabel('Date')
    plt.ylabel('NDVI')
    plt.title('NDVI Time-series')
    plt.grid(True)
    plt.show()

def analyze_ndvi_trends(dates, ndvi_values):
    """
    Analyze trends in NDVI data using linear regression.

    Parameters:
    dates (list): List of dates corresponding to the NDVI values.
    ndvi_values (list): List of NDVI values.

    Returns:
    dict: Dictionary containing the slope, intercept, and R-squared value of the trend.
    """
    date_nums = pd.to_datetime(dates).map(pd.Timestamp.toordinal).values
    slope, intercept, r_value, p_value, std_err = stats.linregress(date_nums, ndvi_values)
    trend_info = {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'p_value': p_value,
        'std_err': std_err
    }
    return trend_info
