from time import sleep
import numpy as np


import qcodes as qc
from qcodes.utils.dataset.doNd import do1d, do2d, dond, plot, LinSweep, LogSweep

__all__ = ['bias_scan', 'tune_to_peak']

def bias_scan(V_range=600e-6, num=251, show_progress=True, sweep=None, delay=None, plot=True, measure_params=None, V_bias=None):
    station = qc.Station.default
    if V_bias is None:
        assert hasattr(station, 'V_bias'), 'Station.V_bias must be registered'
        V_bias = station.V_bias

    if measure_params is None:
        measure_params = station.measure_params

    if delay is None:
        delay = station.t_lockin.delay

    if sweep is None:
        sweeps = []
    elif isinstance(sweep, LinSweep):
        sweeps = [sweep]
    elif isinstance(sweep, list):
        sweeps = sweep
    sweeps += [LinSweep(V_bias, -V_range, V_range, num, delay=delay)]

    measurement_name = f'{len(sweeps)}D:bias_scan'
    for sweep in sweeps[:-1]:
        measurement_name += f':{sweep._param.name}'

    V_bias(-V_range)
    sleep(0.2)
    return dond(
        *sweeps,
        *measure_params,
        measurement_name=measurement_name,
        show_progress=show_progress,
        do_plot=True
    )


def tune_to_peak(gate, measure_param, voltage_center=None, voltage_window=10e-3, step=.1e-3, reverse=False, delay=True, silent=False,
                 plot=True):
    station = qc.Station.default
    assert station is not None, "No station initialized"

    param_idx = station.measure_params.index(measure_param)

    if voltage_center is None:
        voltage_center = gate()

    if delay is True:
        assert hasattr(station, 't_lockin'), "Station must have function t_lockin"
        delay = station.t_lockin.delay
    elif delay in [False, None]:
        delay = 0

    # Specify sweep parameters
    N = int(round(voltage_window / step)) + 1
    V_start, V_stop = voltage_center - voltage_window / 2, voltage_center + voltage_window / 2
    if reverse:
        V_start, V_stop = V_stop, V_start
    sweep = LinSweep(gate, V_start, V_stop, N, delay=delay)

    # Go to start position
    gate(V_start)
    sleep(2*delay)

    # Perform measurement
    dataset_qcodes, axes, colorbars = dond(
        sweep, *station.measure_params,
        measurement_name=f'1D:tune_to_peak:{gate.name}'
    )
    dataset = dataset_qcodes.to_xarray_dataset()

    # Extract peak voltage
    measure_array = dataset[measure_param.name]
    peak_value = float(measure_array.max())
    peak_idx = np.argmax(measure_array.data)
    peak_voltage = sweep.get_setpoints()[peak_idx]

    # Tune to peak
    gate(peak_voltage)
    sleep(2*delay)
    # for k in range(20):
    #     sleep(0.05)
    #     final_peak_value = measure_param()
    #     print(f'Peak conductance: {final_peak_value:.4g} ({peak_value:.4g} during sweep)')
    # print()
    final_peak_value = measure_param()

    # Plot results
    if plot:
        from ..tools.plot_tools import plot_dual_axis
        fig, axes = plot_dual_axis(dataset)
        ax = axes[param_idx]
        color = f'C{param_idx}'
        ax.plot(peak_voltage, final_peak_value, '*', ms=8, color=color)
        ax.vlines(peak_voltage, *ax.get_ylim(), linestyle='--', color=color)
    else:
        fig = None

    # Print and return results
    if not silent:
        print(f'Peak voltage: {peak_voltage:.6g} V ({peak_idx / (N - 1) * 100:.0f}% of sweep)')
        print(f'Peak conductance: {final_peak_value:.4g} ({peak_value:.4g} during sweep)')

    return {
        'peak_voltage': peak_voltage,
        'peak_conductance': final_peak_value,
        'fig': fig,
        'axes': axes
    }