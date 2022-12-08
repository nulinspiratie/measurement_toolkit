from time import sleep
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
from measurement_toolkit.tools.data_tools import smooth, convert_to_dataset


__all__ = ['extract_dataset_peaks', 'goto_next_peak', 'goto_nearest_peak']


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


def extract_dataset_peaks(dataset, measure_param=None, silent=False,  prominence=0.05, **kwargs):
    station = qc.Station.default
    if measure_param is None and hasattr(station, 'default_measure_param'):
        measure_param = station.default_measure_param
        
    dataset = convert_to_dataset(dataset, 'xarray')
        
    if measure_param is not None:
        arr = getattr(dataset, measure_param.name)
    else:    
        arr = next(iter(dataset.data_vars.values()))
    xvals = next(iter(arr.coords.values()))

    peak_idxs, _ = find_peaks(arr.values, prominence=prominence, **kwargs)
    peak_xvals = xvals[peak_idxs].values
    peak_yvals = arr[peak_idxs].values

    if not silent:
        print(f'Found {len(peak_idxs)} peaks at voltages {list(peak_xvals)}')
    
    if not silent:
        arr.plot()
        plt.plot(peak_xvals, peak_yvals, marker='*', ms=12, linestyle='')
    return peak_xvals, peak_yvals


def goto_peak(gate, method='next', around=10e-3, num=101, measure_param=None, silent=False, prominence=0.05, peak_shift=None, **kwargs):
    assert method in ['next', 'nearest']

    station = qc.Station.default
    if measure_param is None:
        measure_param = station.default_measure_param
    dataset = gate.sweep(around=around, num=num, plot=False, measure_params=[measure_param], initial_delay=7*station.t_lockin())

    peak_xvals, peak_yvals = extract_dataset_peaks(dataset, measure_param=measure_param, silent=silent, prominence=prominence, **kwargs)

    V0 = gate()
    # Find peak satisfying method criteria
    if not len(peak_xvals):
        success = False
        peak_xval = None
    elif method == 'nearest':
        peak_idx = np.argmin(np.abs(peak_xvals - V0))
        peak_xval = peak_xvals[peak_idx]
        success = True
    elif method == 'next':
        for peak_idx, peak_xval in enumerate(peak_xvals):
            if peak_xval > V0:
                success = True
                break
        else:
            success = False

    if success:
        print(f'Going to peak {peak_idx+1} at voltage {gate.name}({peak_xval})')
        gate(peak_xval)

        sleep(1)
        peak_yval = measure_param()
        print(f'Conductance at peak: {measure_param.name} = {peak_yval}')
        if not silent:
            plt.plot(peak_xval, peak_yval, marker='*', ms=12, linestyle='', color='g')

        # Optionally also shift from the peak
        if peak_shift is not None:
            shift_xval = peak_xval + peak_shift
            gate(shift_xval)
            sleep(1)
            shift_yval = measure_param()
            print(f'Conductance {peak_shift} mV from peak: {measure_param.name} = {shift_yval}')
            if not silent:
                plt.plot(shift_xval, shift_yval, marker='*', ms=12, linestyle='', color='g')
    else:
        print(f'Did not find a peak past current voltage {gate.name} = {V0} V')

    return {
        'success': success,
        'peak_xval': peak_xval,
        'peak_xvals': peak_xvals,
        'peak_yvals': peak_yvals,
    }

def goto_next_peak(gate, around=10e-3, num=101, measure_param=None, silent=False, prominence=0.05, peak_shift=None, **kwargs):
    return goto_peak(
        gate=gate, 
        method='next', 
        around=around, 
        num=num, 
        measure_param=measure_param,
        silent=False,
        prominence=prominence,
        peak_shift=peak_shift,
        **kwargs
    )

def goto_nearest_peak(gate, around=20e-3, num=151, measure_param=None, silent=False, prominence=0.05, peak_shift=None, **kwargs):
    return goto_peak(
        gate=gate, 
        method='nearest', 
        around=around, 
        num=num, 
        measure_param=measure_param,
        silent=False,
        prominence=prominence,
        peak_shift=peak_shift,
        **kwargs
    )