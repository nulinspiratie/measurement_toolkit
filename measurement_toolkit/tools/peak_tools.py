import qcodes as qc
from scipy.signal import find_peaks
import numpy as np
from typing import Union
import xarray as xr
import matplotlib.pyplot as plt
from qcodes.dataset.data_set import DataSet
from qcodes.dataset import (
    Measurement,
    experiments,
    initialise_or_create_database_at,
    load_by_run_spec,
    load_or_create_experiment,
)
from measurement_toolkit.tools.data_tools import smooth


def convert_data_to_xarray(run_id=None, plot=False):
    if isinstance(run_id, int):
        data = load_by_run_spec(captured_run_id=run_id)
        data = data.to_xarray_dataset()
    else:
        data = run_id


    # assert len(data.data_vars) == 1, "Cannot handle dataset with multiple arrays"
    name, x_arr = next(iter(data.coords.items()))
    label, y_arr = next(iter(data.data_vars.items()))

    #give values of x,y
    x = x_arr.values
    y = y_arr.values

    #plot if needed
    if plot:
        f = plt.figure()
        f.set_figwidth(10)
        f.set_figheight(4)
        plt.plot(x,y)
        plt.show()
    return x, y

    
def apply_fft(x, y):
    # Perform Fourier transform
    y_fourier = np.fft.fft(y)

    # Take absolute value and trim to only use positive frequencies
    max_idx = len(y)//2
    y_fourier = np.abs(y_fourier)[:max_idx]

    # Create array of corresponding x values
    dx = x[1] - x[0]
    x_fourier = np.fft.fftfreq(y.size, dx)[:max_idx]

    return x_fourier, y_fourier


def estimate_prominence_fft(x, y, smooth_points=None, silent=True):
    # Perform Fourier transform
    x_fourier, y_fourier = apply_fft(x, y)

    # Optionally smooth FFT data
    if smooth_points is not None:
        y_fourier = smooth(y_fourier, smooth_points)

    # Extract dominant peak of FFT
    peak_idxs, _ = find_peaks(y_fourier, prominence=10)
    
    #grab prominence from fft_ysmoothed
    peak_frequencies = x_fourier[peak_idxs]
    peak_amplitudes = y_fourier[peak_idxs]
    prominence = np.abs(1 / peak_frequencies)[1]

    if not silent:
        print(f'Extracted prominence: {prominence:.4f}')

        # Plot Fourier transform
        fig, ax = plt.subplots(figsize=(10,4))
        ax.semilogy(x_fourier, y_fourier)
        ax.plot(peak_frequencies, peak_amplitudes, marker='o', color='r', ms=8, linestyle='')
        ax.set_title('Fourier transform')

    return prominence


def plot_peaks(x ,y ,x_peaks, y_peaks):
    # prominence1 = prominence[0] #komt uit fftfreq
    # distance1 = prominence[1]/dx

    fig, ax = plt.subplots(figsize=(14,4))
    plt.plot(x,y)
    plt.scatter(x_peaks, y_peaks,color = "r");
    return ax


# combined codes
def extract_peaks(
    dataset: Union[int, DataSet], 
    prominence: float = None, 
    min_distance: float = None, 
    smooth_points: int = None, 
    silent: bool = True
):
    # Extract a dict of x and y values
    x, y = convert_data_to_xarray(dataset)

    # Optionally estimate prominence using FFT
    if prominence is None:
        prominence = estimate_prominence_fft(x, y, smooth_points=smooth_points, silent=silent)

    # Perform scipy peak finding algorithm
    idx_peaks, _ = find_peaks(y, prominence=prominence, distance=min_distance)
    x_peaks = x[idx_peaks]
    y_peaks =  y[idx_peaks]

    # Optionally plot data with peaks
    if not silent:
        ax_peaks = plot_peaks(x=x, y=y, x_peaks=x_peaks, y_peaks=y_peaks)

    return {
        'x': x_peaks,
        'y': y_peaks,
        'idxs': idx_peaks,
        'axes': [ax_peaks]
    }