import numpy as np
from matplotlib import pyplot as plt
from .data_tools import smooth, convert_to_dataset
import xarray


__all__ = ['extract_slope', 'optimize_IQ']

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


def optimize_IQ(data_or_arrs, N_phase=180, phase_min=0, silent=False, plot=False):
    if isinstance(data_or_arrs, tuple):
        I, Q = data_or_arrs
        signal_complex = I + 1.j * Q
    else: 
        data = convert_to_dataset(data_or_arrs, 'xarray')
        if hasattr(data, 'RF_inphase') and hasattr(data, 'RF_quadrature'):
            signal_complex = data.RF_inphase + 1.j * data.RF_quadrature.values
        elif hasattr(data, 'R') and hasattr(data, 'theta'):
            arr_R = data.R
            arr_theta = data.theta
            signal_complex = arr_R * np.exp(1.j * arr_theta/360 * 2*np.pi)
        else:
            raise RuntimeError()

    rotation_angles = np.arange(0, 180, 180/N_phase, dtype=int) + phase_min
    signals_rotated = np.array([
        signal_complex * np.exp(-1.j *angle / 360 * 2*np.pi)
        for angle in rotation_angles
    ])
    signals_I = np.real(signals_rotated)
    signals_Q = np.imag(signals_rotated)
    std_I = np.nanstd(signals_I, axis=tuple(range(1, signals_I.ndim)))
    std_Q = np.nanstd(signals_Q, axis=tuple(range(1, signals_Q.ndim)))
    std_ratios = std_I / std_Q

    std_max_idx = np.argmax(std_ratios)
    angle_max = rotation_angles[std_max_idx]
    signal_complex_max = signal_complex * np.exp(-1.j *angle_max / 360 * 2*np.pi)
    signal_I_rotated = np.real(signal_complex_max)
    signal_Q_rotated = np.imag(signal_complex_max)

    results = {
        'signal_complex': signal_complex,
        'std_ratios': std_ratios,
        'signal': signal_I_rotated,
        'signal_perpendicular': signal_Q_rotated,
        'phase_shift': angle_max
    }

    if not silent:
        print(f'Rotating signal by {angle_max:.0f}° to maximize signal along I')

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14,4))
        ax = axes[0]
        signal_I_rotated.plot(ax=ax)
        ax.set_clim(np.min(signal_I_rotated), np.max(signal_I_rotated))
        ax.append_title(f'Maximal IQ signal (Rotated by {angle_max:.0f}°)')
        ax = axes[1]
        signal_Q_rotated.plot(ax=ax)
        ax.set_clim(np.min(signal_Q_rotated), np.max(signal_Q_rotated))
        ax.append_title(f'Minimal IQ signal (Rotated by {angle_max:.0f}°)')

        results['fig'] = fig
        results['axes'] = axes

    return results
