import numpy as np
from matplotlib import pyplot as plt
from .data_tools import smooth


def extract_slope(arr, plot=True, smooth_order=5):
    yvals, xvals = list(arr.coords.values())

    # Extract peak indices
    max_idxs = []
    for row in arr:
        if any(np.isnan(row)):
            break
        if smooth_order is not None:
            row = smooth(row, smooth_order)
        
        max_idxs.append(np.argmax(row))

    # Get corresponding max xvals 
    max_xvals = np.array([xvals[idx] for idx in max_idxs])

    # Get average slope
    dx_max = max_xvals[1:] - max_xvals[:-1]

    # Remove outliers
    dx_mean = np.mean(dx_max)
    dx_std = np.std(dx_max)
    dx_max = [elem for elem in dx_max if elem < 0]
    plt.plot(dx_max)

    # Plot results
    if plot:
        plt.figure()
        arr.plot()
        plt.plot(max_xvals, yvals[:len(max_xvals)])