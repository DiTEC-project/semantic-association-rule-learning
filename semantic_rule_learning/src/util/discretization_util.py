"""
This Python script includes functions relevant to numerical data discretization
"""
import numpy as np


def equal_frequency_discretization(numerical_values, num_bins):
    sorted_values = np.sort(numerical_values)
    boundaries = np.interp(np.linspace(0, len(sorted_values), num_bins + 1),
                           np.arange(len(sorted_values)),
                           sorted_values)
    return boundaries


def equal_width_discretization(numerical_values, num_bins):
    sorted_values = np.sort(numerical_values)
    min = int(np.floor(sorted_values.min()))
    max = int(np.ceil(sorted_values.max()))
    interval = int((max - min) / num_bins)
    boundaries = [i for i in range(min, max + interval, interval)]
    return boundaries
