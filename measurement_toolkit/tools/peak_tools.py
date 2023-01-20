import xarray
from qcodes.dataset.data_set import DataSet
from time import sleep
import qcodes as qc
from scipy.signal import find_peaks
import numpy as np
from typing import Union, List
import xarray as xr
import matplotlib.pyplot as plt
from qcodes.dataset import (
    Measurement,
    experiments,
    initialise_or_create_database_at,
    load_by_run_spec,
    load_or_create_experiment,
    MeasurementLoop,
    Sweep,
)
from measurement_toolkit.tools.data_tools import smooth, convert_to_dataset
from measurement_toolkit.tools.plot_tools import plot_data
from measurement_toolkit.tools.general_tools import args_from_config


__all__ = [
    'extract_dataset_peaks', 
    'analyse_Coulomb_peak_shift',
    'goto_next_peak', 
    'goto_nearest_peak',
    'extract_peaks_2D'
]


def convert_to_array(dataset_or_array=None, array_name=None):
    """Converts dataset or array to an xarray DataArray that includes xvals"""
    station = qc.Station.default
    if array_name is None and hasattr(station, 'default_measure_param'):
        array_name = station.default_measure_param

    if isinstance(dataset_or_array, np.ndarray):  # Convert numpy array to DataArray
        arr = xarray.DataArray(dataset_or_array, coords={'index': range(len(dataset_or_array))})

    if isinstance(dataset_or_array, xarray.DataArray):  # Already an array
        arr = dataset_or_array
    else:  # Convert dataset to DataArray
        dataset = convert_to_dataset(dataset_or_array, 'xarray')
        arrs = list(dataset.data_vars.values())
            
        if array_name is not None:
            arr = next(arr for arr in arrs if arr.name == array_name.name)
        else:    
            if len(arrs) != 1:
                print(f'Found {len(arrs)} arrays, picking first one: {arrs[0].name}')
            arr = arrs[0]

    return arr


def analyse_Coulomb_peak_shift(dataset_or_array, dx, plot=True):
    """Analyses signal difference upon shifting an array by a small amount"""
    array = dataset_or_array

    xvals_name, xvals = list(array.coords.items())[-array.ndim]

    dx_min = xvals[1] - xvals[0]
    dx_scale = dx / dx_min
    idx_shift = max(int(np.floor(dx_scale)), 1)

    array_shift = array[idx_shift:] - array[:-idx_shift].values
    array_shift *= dx_scale / idx_shift
    array_shift = xarray.DataArray(
        array_shift, 
        coords={xvals_name: (xvals[idx_shift:] + xvals[:-idx_shift].values)/2}
    )

    results = {'array_shift': array_shift}

    # Plot results
    if plot:
        if array.ndim == 2:
            fig, axes = plt.subplots(1, 2, figsize=(12,6))

            # Plot signal
            array.plot(ax=axes[0])

            # Plot signal difference
            ax = axes[1]
            array_shift.plot(ax=ax, cmap='bwr')
            cbar = ax.get_colorbar()
            cbar.set_label('Signal difference')
            results['axes'] = axes
        else:
            fig, ax = plt.subplots(figsize=(8, 4))
            array.plot(label=array.name)
            array_shift.plot(label=f'{array.name} shift')
            ax.set_ylabel('Signal')
            ax.legend()
            ax.grid('on')

            results['ax'] = ax

        fig.append_title(f'Signal difference by shifting {xvals.name} by {dx*1e3:.2f} mV')
        fig.tight_layout()
        results['fig'] = fig

    return results


def extract_peaks(
    dataset_or_array, 
    prominence: Union[float, List[float]], 
    array_name=None, 
    detailed_results=False,
    **kwargs
):
    # Convert to array 
    array = convert_to_array(dataset_or_array, array_name=array_name)
    xvals = list(array.coords.values())[-1]

    # Extract peaks by looping over possible prominences
    prominences = [prominence] if isinstance(prominence, float) else prominence
    for prominence in prominences:
        peak_idxs, peak_properties = find_peaks(array.values, prominence=prominence, **kwargs)
        if len(peak_idxs):
            peak_xvals = xvals[peak_idxs].values
            peak_yvals = array[peak_idxs].values
            break
    else:
        peak_xvals, peak_yvals = [], []

    

    if not detailed_results:
        return peak_xvals, peak_yvals
    else:
        return {
            'peak_xvals': peak_xvals,
            'peak_yvals': peak_yvals,
            'peak_idxs': peak_idxs,
            **peak_properties,
        }


def extract_dataset_peaks(dataset_or_array, measure_param=None, silent=False, negative=False, prominence=0.05, **kwargs):
    if isinstance(dataset_or_array, list):  # List of arrays
        dataset_or_array = dataset_or_array[0]
        other_arrs = dataset_or_array[1:]

    arr = convert_to_array(dataset_or_array=dataset_or_array, array_name=measure_param)

    arr_positive = arr * (-1 if negative else 1)

    peak_xvals, peak_yvals = extract_peaks(
        arr_positive, 
        array_name=measure_param,
        prominence=prominence,
    )

    if not silent:
        print(f'Found {len(peak_yvals)} peaks at voltages {list(round(val, 3) for val in peak_xvals)}')
    
    if not silent:
        fig, ax = plt.subplots()
        arr.plot()
        ax.plot(peak_xvals, peak_yvals, marker='*', ms=12, linestyle='')

        try:
            other_arrs = [arr2 for arr2 in arrs if arr2 is not arr]
            if other_arrs:
                ax2 = ax.twinx()
                ax2.color_right_axis('C1')
                for other_arr in other_arrs:
                    other_arr.plot(ax=ax2, color='C1', )
        except Exception:
            print(f"Couldn't plot other data arrs")
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
        print(f'Going to peak {peak_idx+1} at voltage {gate.name}({peak_xval:.5f})')
        gate(peak_xval)

        sleep(1)
        peak_yval = measure_param()
        print(f'Conductance at peak: {measure_param.name} = {peak_yval:.3f}')
        if not silent:
            plt.plot(peak_xval, peak_yval, marker='*', ms=12, linestyle='', color='g')

        # Optionally also shift from the peak
        if peak_shift is not None:
            shift_xval = peak_xval + peak_shift
            gate(shift_xval)
            sleep(1)
            shift_yval = measure_param()
            print(f'Conductance {peak_shift} mV from peak: {measure_param.name} = {shift_yval:.3f}')
            if not silent:
                plt.plot(shift_xval, shift_yval, marker='*', ms=12, linestyle='', color='g')
            plt.show()
    else:
        print(f'Did not find a peak past current voltage {gate.name} = {V0} V')

    return {
        'success': success,
        'peak_xval': peak_xval,
        'peak_xvals': peak_xvals,
        'peak_yvals': peak_yvals,
    }

@args_from_config(
    'goto_nearest_peak', 
    kwargs={'around': 10e-3, 'num': 151, 'prominence': 0.05, 'peak_shift': None}, 
    station_args=['gate']
)
def goto_next_peak(gate, around=None, num=None, measure_param=None, silent=False, prominence=None, peak_shift=None, **kwargs):
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

@args_from_config(
    'goto_nearest_peak', 
    kwargs={'around': 10e-3, 'num': 151, 'prominence': 0.05, 'peak_shift': None}, 
    station_args=['gate']
)
def goto_nearest_peak(gate, around=None, num=None, measure_param=None, silent=False, prominence=None, peak_shift=None, **kwargs):
    return goto_peak(
        gate=gate, 
        method='nearest', 
        around=around, 
        num=num, 
        measure_param=measure_param,
        silent=silent,
        prominence=prominence,
        peak_shift=peak_shift,
        **kwargs
    )


def goto_next_charge_transition(
    charge_transition_parameter,
    compensation_parameter,
    dV_max = 0.1,
    V_step=0.001,
    max_flank_attempts=20,
    target_accuracy=0.1,
    silent=True
):
    delay = qc.Station.default.t_lockin()

    # First go to flank of Coulomb peak
    compensating_gate = compensation_parameter.compensating_gate
    V0 = compensating_gate()
    with MeasurementLoop('goto_flank') as msmt:
        for k in Sweep(range(max_flank_attempts), 'repetition'):
            msmt.measure(compensation_parameter)
            conductance_error = compensation_parameter.results['conductance_error']
            target_conductance = compensation_parameter.target_conductance
            accuracy = np.abs(conductance_error /  target_conductance)
            if accuracy < target_accuracy:
                success = True
                msmt.step_out()
                break
            sleep(2.5*delay)
        else:
            success = False
        msmt.measure(success, 'success')

    if not silent:
        plot_compensated_charge_transition_measurement(msmt.dataset, print_summary=False)
        plt.show()

    if not success:
        print(f'Did not find Coulomb peak flank, reverting {compensating_gate} back to {V0}')
        compensating_gate(V0)
        return

    V0_charge = charge_transition_parameter()
    with MeasurementLoop('goto_charge_transition') as msmt:
        for V in Sweep(charge_transition_parameter, V0_charge, V0_charge + dV_max,  step=V_step):
            msmt.measure(compensation_parameter)

    if not silent:
        plot_compensated_charge_transition_measurement(msmt.dataset, print_summary=False)
        plt.show()
    

def plot_compensated_charge_transition_measurement(data, print_summary=True):
    data = convert_to_dataset(data)
    if len(data.target_voltage) < 2:
        return

    fig, axes = plt.subplots(3, figsize=(6,6), sharex=True)

    s = '0' if hasattr(data, 'success') else ''

    ax = axes[0]
    plot_data(data, f'[0[11]00{s}00]', axes=ax, print_summary=print_summary, marker='o')
    ax.legend(['conductance', 'Conductance error'])
    ax.set_ylabel('Conductance (e^2/h)')

    ax = axes[1]
    plot_data(data, f'[00001{s}00]', axes=ax, print_summary=False, marker='o')
    plot_data(data, f'[00000{s}10]', axes=ax, print_summary=False, marker='o')
    ax.legend(['Initial voltage', 'Target voltage'])
    ax.set_ylabel('Voltage (V)')
    
    ax = axes[2]
    plot_data(data, f'[10000{s}00]', axes=ax, print_summary=False, marker='o')
    plot_data(data, f'[00010{s}00]', axes=ax, print_summary=False, marker='o')
    plot_data(data, f'[00000{s}01]', axes=ax, print_summary=False, marker='o')
    ax.legend(['Capacitive compensation', 'Conductance error compensation', 'Total compensation'])
    ax.set_ylabel('Voltage compensation (V)')

    
    if hasattr(data, 'success'):
        fig.append_title(f'Successfully reached within tolerance: {bool(data.success)}')
        fig.tight_layout()

    return fig, axes


### Extract peak lines from 2D plots
def extract_peak_lines(peaks_array, plot=False):
    import cv2

    peaks_array = np.array(peaks_array, dtype='uint8')
    result = cv2.HoughLines(peaks_array,1,np.pi/180,200)
    if result is None:
        print('No peak lines found')
        return []
    result = result[:,0]  # Somehow we receive a 3D array where second dim has length 1

    N_y, N_x = peaks_array.shape
    # Convert coordinates to x and y
    peak_lines = []
    for rho, theta in result:
        x0 = rho * np.cos(theta)
        y0 = rho * np.sin(theta)

        deviations_x = -(np.arange(N_x) - x0) / np.sin(theta)
        deviations_y = (np.arange(N_y) - y0) / np.cos(theta)
        deviations = np.array([
            max(min(deviations_x), min(deviations_y)),
            min(max(deviations_x), max(deviations_y))
        ])

        xrange_idxs = x0 - deviations * np.sin(theta)
        yrange_idxs = y0 + deviations * np.cos(theta)

        # Convert to actual x and y values
        yvals, xvals = list(elem.values for elem in peaks_array.coords.values())[-2:]
        xrange = xvals[0] + (xvals[1] - xvals[0]) * xrange_idxs
        yrange = yvals[0] + (yvals[1] - yvals[0]) * yrange_idxs
        slope = (yrange[1] - yrange[0]) / (xrange[1] - xrange[0])

        peak_lines.append({
            'xlim': xrange,
            'ylim': yrange,
            'angle': theta + np.pi/2,
            'slope': slope
        })

    # Plot results
    if plot:
        peaks_array.plot()
        for line in peak_lines:
            plt.plot(line['xlim'], line['ylim'])
    
    return peak_lines


def group_peaks_by_lines(peaks_array, peak_lines, max_distance=4e-3):
    for peak_line in peak_lines:
        peak_line['peaks'] = []

    for k, peaks_row in enumerate(peaks_array.values):
        for kk, idx in enumerate(peaks_row):
            if not idx:
                continue

            idx = peaks_array[k, kk]

            # Peak coordinates
            y, x = list(float(elem) for elem in idx.coords.values())[-2:]

            # Check if the peak is close to any of the lines
            peak_distances = []
            for peak_line in peak_lines:
                xlim, ylim = peak_line['xlim'], peak_line['ylim']
                dx = xlim[1] - xlim[0]
                dy = ylim[1] - ylim[0]
                distance = np.abs(dx*(ylim[1]-y) - (xlim[1]-x)*dy) / np.sqrt(dx**2 + dy**2)
                peak_distances.append(distance)

            if np.min(peak_distances) < max_distance:
                closest_line_idx = np.argmin(peak_distances)
                peak_lines[closest_line_idx]['peaks'].append((x, y))
        
    # Transform all peaks to x,y arrays
    for peak_line in peak_lines:
        peak_line['peaks'] = np.array(peak_line['peaks'])


def plot_peaks_2D(peaks_array, peak_lines, **kwargs):
    # Plot results
    peaks_array.plot()

    for k, line in enumerate(peak_lines):
        color = f'C{k%10}'
        plt.plot(line['xlim'], line['ylim'], lw=1, color=color)

        x, y = line['peaks'].transpose()
        plt.plot(x, y, marker='o', ms=1, linestyle='', color=color)


def extract_peaks_2D(array, prominence, max_distance=1e-3, plot=True):
    # Boolean array where 1 indicates that a datapoint is a peak
    peaks_array = xarray.DataArray(np.zeros(shape=array.shape), coords=array.coords)

    # Extract peaks for each row, update peaks_array
    peak_results = []
    for k, row in enumerate(array):
        peak_result = extract_peaks(row, prominence=prominence, detailed_results=True)
        peak_results.append(peak_result)

        peaks_array[k, peak_result['peak_idxs']] = 1

    peak_lines = extract_peak_lines(peaks_array)

    group_peaks_by_lines(peaks_array, peak_lines, max_distance=max_distance)

    if plot:
        plot_peaks_2D(peaks_array=peaks_array, peak_lines=peak_lines)

    # return peak_results
    return {
        'peaks_array': peaks_array,
        'peak_lines': peak_lines,
        'peaks': peak_results,
        'peak_groups': None
    }


### Old code from Tim
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
# def extract_peaks(
#     dataset_or_array: Union[int, DataSet, xarray.DataArray], 
#     prominence: float = None, 
#     min_distance: float = None, 
#     smooth_points: int = None, 
#     silent: bool = True
# ):
#     # Extract a dict of x and y values
#     x, y = convert_data_to_xarray(dataset_or_array)

#     # Optionally estimate prominence using FFT
#     if prominence is None:
#         prominence = estimate_prominence_fft(x, y, smooth_points=smooth_points, silent=silent)

#     # Perform scipy peak finding algorithm
#     idx_peaks, _ = find_peaks(y, prominence=prominence, distance=min_distance)
#     x_peaks = x[idx_peaks]
#     y_peaks =  y[idx_peaks]

#     # Optionally plot data with peaks
#     if not silent:
#         ax_peaks = plot_peaks(x=x, y=y, x_peaks=x_peaks, y_peaks=y_peaks)

#     return {
#         'x': x_peaks,
#         'y': y_peaks,
#         'idxs': idx_peaks,
#         'axes': [ax_peaks]
#     }

