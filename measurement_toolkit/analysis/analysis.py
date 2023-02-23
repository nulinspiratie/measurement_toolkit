import numpy as np
import functools
import itertools
from matplotlib.axis import Axis
from matplotlib.colors import TwoSlopeNorm
import peakutils
import logging
from typing import Union, Dict, Any, List, Sequence, Iterable, Tuple
from copy import copy
import collections
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

from measurement_toolkit.tools.general_tools import property_ignore_setter

import qcodes as qc
from qcodes.instrument.parameter import Parameter
from qcodes.utils import validators as vals

__all__ = [
    "find_high_low",
    "edge_voltage",
    "find_up_proportion",
    "count_blips",
    "analyse_traces",
]

logger = logging.getLogger(__name__)


if "analysis" not in qc.config.keys():
    qc.config.analysis = {}
analysis_config = qc.config.analysis


def find_high_low(
    traces: Union[np.ndarray, list, dict],
    plot: bool = False,
    threshold_peak: float = 0.02,
    attempts: int = 8,
    threshold_method: str = "config",
    min_voltage_difference: Union[float, str] = "config",
    threshold_requires_high_low: Union[bool, str] = "config",
    min_SNR: Union[float, None] = None,
    skip_pts=0
):
    """ Find high and low voltages of traces using histograms

    This function determines the high and low voltages of traces by binning them
    into 30 bins, and trying to discern two peaks.
    Useful for determining the threshold value for measuring a blip.

    If no two peaks can be discerned after all attempts, None is returned for
    each of the returned dict keys except DC_voltage.

    Args:
        traces: 2D array of acquisition traces
        plot: Whether to plot the histograms
        threshold_peak: Threshold for discerning a peak. Will be varied if too
            many/few peaks are found
        attempts: Maximum number of attempts for discerning two peaks.
            Each attempt the threshold_peak is decreased/increased depending on
            if too many/few peaks were found
        threshold_method: Method used to determine the threshold voltage.
            Allowed methods are:

            * **mean**: average of high and low voltage.
            * **{n}\*std_low**: n standard deviations above mean low voltage,
              where n is a float (ignore slash in raw docstring).
            * **{n}\*std_high**: n standard deviations below mean high voltage,
              where n is a float (ignore slash in raw docstring).
            * **config**: Use threshold method provided in
              ``config.analysis.threshold_method`` (``mean`` if not specified)

        min_voltage_difference: minimum difference between high and low voltage.
            If not satisfied, all results are None.
            Will try to retrieve from config.analysis.min_voltage_difference,
            else defaults to 0.3V
        threshold_requires_high_low: Whether or not both a high and low voltage
            must be discerned before returning a threshold voltage.
            If set to False and threshold_method is not ``mean``, a threshold
            voltage is always determined, even if no two voltage peaks can be
            discerned. In this situation, there usually aren't any blips, or the
            blips are too short-lived to have a proper high current.
            When the threshold_method is ``std_low`` (``std_high``), the top
            (bottom) 20% of voltages are scrapped to ensure any short-lived blips
            with a high (low) current aren't included.
            The average is then taken over the remaining 80% of voltages, which
            is then the average low (high) voltage.
            Default is True.
            Can be set by config.analysis.threshold_requires_high_low
        min_SNR: Minimum SNR between high and low voltages required to determine
            a threshold voltage (default None).
        skip_pts: Optional number of points to skip at the start of each trace

    Returns:
        Dict[str, Any]:
        * **low** (float): Mean low voltage, ``None`` if two peaks cannot be
          discerned
        * **high** (float): Mean high voltage, ``None`` if no two peaks cannot
          be discerned
        * **threshold_voltage** (float): Threshold voltage for a blip. If SNR is
          below ``min_SNR`` or no two peaks can be discerned, returns ``None``.
        * **voltage_difference** (float): Difference between low and high
          voltage. If not two peaks can be discerned, returns ``None``.
        * **DC_voltage** (float): Average voltage of traces.
    """
    if attempts < 1:
        raise ValueError(
            f"Attempts {attempts} to find high and low voltage must be at least 1"
        )

    # Convert traces to list of traces, as traces may contain multiple 2D arrays
    if isinstance(traces, np.ndarray):
        traces = [traces]
    elif isinstance(traces, dict):
        traces = list(traces.values())
        assert isinstance(traces[0], np.ndarray)
    elif isinstance(traces, list):
        pass

    # Optionally remove the first points of each trace
    if skip_pts > 0:
        traces = [trace[:, skip_pts:] for trace in traces]

    # Turn list of 2D traces into a single 2D array
    traces = np.ravel(traces)

    # Retrieve properties from config.analysis
    if threshold_method == "config":
        threshold_method = analysis_config.get("threshold_method", "mean")
    if min_voltage_difference == "config":
        min_voltage_difference = analysis_config.get("min_voltage_difference", 0.3)
    if threshold_requires_high_low == "config":
        threshold_requires_high_low = analysis_config.get(
            "threshold_requires_high_low", True
        )
    if min_SNR is None:
        min_SNR = analysis_config.get("min_SNR", None)

    # Calculate DC (mean) voltage
    DC_voltage = np.mean(traces)

    # Perform a histogram over all voltages in all traces. These bins will be
    # used to determine two peaks, corresponding to low/high voltage
    traces_1D = np.ravel(traces)
    traces_1D = traces_1D[~np.isnan(traces_1D)]
    hist, bin_edges = np.histogram(traces_1D, bins=30)

    # Determine minimum number of bins between successive histogram peaks
    if min_voltage_difference is not None:
        min_dist = int(np.ceil(min_voltage_difference / np.diff(bin_edges)[0]))
    else:
        min_dist = 5

    # Find two peaks by changing the threshold dependent on the number of peaks foudn
    for k in range(attempts):
        peaks_idx = np.sort(
            peakutils.indexes(hist, thres=threshold_peak, min_dist=min_dist)
        )
        if len(peaks_idx) == 2:
            break
        elif len(peaks_idx) == 1:
            print('One peak found instead of two, lowering threshold')
            threshold_peak /= 1.5
        elif len(peaks_idx) > 2:
            print(f'Found {len(peaks_idx)} peaks instead of two, '
                   'increasing threshold')
            threshold_peak *= 1.5
    else:  # Could not identify two peaks after all attempts
        print(f'Could not find two peaks in find_high_low, giving up')
        results = {
            "low": None,
            "high": None,
            "threshold_voltage": np.nan,
            "voltage_difference": np.nan,
            "DC_voltage": DC_voltage,
        }

        if not threshold_requires_high_low and threshold_method != "mean":
            # Still return threshold voltage even though no two peaks were observed
            low_or_high, equation = threshold_method.split(':')
            assert low_or_high in ['low', 'high']

            voltages = sorted(traces.flatten())
            if low_or_high == 'low':
                # Remove top 20 percent (high voltage)
                cutoff_slice = slice(None, int(0.8 * len(voltages)))
                voltages_cutoff = voltages[cutoff_slice]
                mean = results['low'] = np.mean(voltages_cutoff)
            else:
                # Remove bottom 20 percent of voltages (low voltage)
                cutoff_slice = slice(int(0.8 * len(voltages)), None)
                voltages_cutoff = voltages[cutoff_slice]
                mean = results['high'] = np.mean(voltages_cutoff)
            # Mean and std are used when evaluating the equation
            std = results['std'] = np.std(voltages_cutoff)

            threshold_voltage = eval(equation)
            results["threshold_voltage"] = threshold_voltage

        return results

    # Find mean voltage, used to determine which points are low/high
    # Note that this is slightly odd, since we might use another threshold_method
    # later on to distinguish between high and low voltage
    mean_voltage_idx = int(np.round(np.mean(peaks_idx)))
    mean_voltage = bin_edges[mean_voltage_idx]

    # Create dictionaries containing information about the low, high state
    low, high = {}, {}
    low["traces"] = traces[traces < mean_voltage]
    high["traces"] = traces[traces > mean_voltage]
    for signal in [low, high]:
        signal["mean"] = np.mean(signal["traces"])
        signal["std"] = np.std(signal["traces"])
    voltage_difference = high["mean"] - low["mean"]

    if threshold_method == "mean":
        # Threshold_method is midway between low and high mean
        threshold_voltage = (high["mean"] + low["mean"]) / 2
    elif ':' in threshold_method:
        low_or_high, equation = threshold_method.split(':')
        assert low_or_high in ['low', 'high']
        signal = {'low': low, 'high': high}[low_or_high]
        mean = signal["mean"]
        std = signal["std"]
        threshold_voltage = eval(equation)
    else:
        raise RuntimeError(f"Threshold method {threshold_method} not understood")

    SNR = voltage_difference / np.sqrt(high["std"] ** 2 + low["std"] ** 2)

    if min_SNR is not None and SNR < min_SNR:
        logger.info(f"Signal to noise ratio {SNR} is too low")
        threshold_voltage = np.nan

    # Plotting
    if plot is not False:
        if plot is True:
            plt.figure()
        else:
            plt.sca(plot)
        for k, signal in enumerate([low, high]):
            sub_hist, sub_bin_edges = np.histogram(np.ravel(signal["traces"]), bins=10)
            width = np.mean(np.diff(sub_bin_edges))
            plt.bar(sub_bin_edges[:-1], sub_hist, width=width, color="bg"[k])
            if k < len(peaks_idx):
                plt.plot(signal["mean"], hist[peaks_idx[k]], "or", ms=12)

    return {
        "low": low,
        "high": high,
        "threshold_voltage": threshold_voltage,
        "voltage_difference": voltage_difference,
        "SNR": SNR,
        "DC_voltage": DC_voltage,
    }


def edge_voltage(
    traces: np.ndarray,
    edge: str,
    state: str,
    threshold_voltage: Union[float, None] = None,
    points: int = 5,
    start_idx: int = 0,
) -> np.ndarray:
    """ Test traces for having a high/low voltage at begin/end

    Args:
        traces: 2D array of acquisition traces
        edge: which side of traces to test, either ``begin`` or ``end``
        state: voltage that the edge must have, either ``low`` or ``high``
        threshold_voltage: threshold voltage for a ``high`` voltage (blip).
            If not specified, ``find_high_low`` is used to determine threshold
        points: Number of data points to average over to determine state
        start_idx: index of first point to use. Useful if there is some
            capacitive voltage spike occuring at the start.
            Only used if edge is ``begin``.

    Returns:
        1D boolean array, True if the trace has the correct state at the edge
    """
    assert edge in ["begin", "end"], f"Edge {edge} must be `begin` or `end`"
    assert state in ["low", "high"], f"State {state} must be `low` or `high`"

    if edge == "begin":
        if start_idx > 0:
            idx_list = slice(start_idx, start_idx + points)
        else:
            idx_list = slice(None, points)
    else:
        idx_list = slice(-points, None)

    # Determine threshold voltage if not provided
    if threshold_voltage is None or np.isnan(threshold_voltage):
        threshold_voltage = find_high_low(traces)["threshold_voltage"]

    if threshold_voltage is None or np.isnan(threshold_voltage):
        # print('Could not find two peaks for empty and load state')
        success = np.array([False] * len(traces))
    elif state == "low":
        success = [np.mean(trace[idx_list]) < threshold_voltage for trace in traces]
    else:
        success = [np.mean(trace[idx_list]) > threshold_voltage for trace in traces]
    return np.array(success)


def find_up_proportion(
    traces: np.ndarray,
    threshold_voltage: Union[float, None] = None,
    return_array: bool = False,
    start_idx: int = 0,
    filter_window: int = 0,
) -> Union[float, np.ndarray]:
    """ Determine the up proportion of traces (traces that have blips)

    Args:
        traces: 2D array of acquisition traces
        threshold_voltage: threshold voltage for a ``high`` voltage (blip).
            If not specified, ``find_high_low`` is used to determine threshold
        return_array: whether to return the boolean array or the up proportion
        start_idx: index of first point to use. Useful if there is some
            capacitive voltage spike occuring at the start.
            Only used if edge is ``begin``
        filter_window: number of points of smoothing (0 means no smoothing)

    Returns:

        if return_array is False
            (float) The proportion of traces with a blip
        else
            Boolean array, True if the trace has a blip

    """
    # trace has to contain read stage only
    # TODO Change start point to start time (sampling rate independent)
    if threshold_voltage is None or np.isnan(threshold_voltage):
        threshold_voltage = find_high_low(traces)["threshold_voltage"]

    if threshold_voltage is None or np.isnan(threshold_voltage):
        return 0

    if filter_window > 0:
        traces = [
            np.convolve(trace, np.ones(filter_window) / filter_window, mode="valid")
            for trace in traces
        ]

    # Filter out the traces that contain one or more peaks
    traces_up_electron = np.array(
        [np.any(trace[start_idx:] > threshold_voltage) for trace in traces]
    )

    if not return_array:
        return sum(traces_up_electron) / len(traces)
    else:
        return traces_up_electron


def count_blips(
    traces: np.ndarray,
    threshold_voltage: float,
    sample_rate: float,
    t_skip: float,
    ignore_final: bool = False,
):
    """ Count number of blips and durations in high/low state.

    Args:
        traces: 2D array of acquisition traces.
        threshold_voltage: Threshold voltage for a ``high`` voltage (blip).
        sample_rate: Acquisition sample rate (per second).
        t_skip: Initial time to skip for each trace (ms).

    Returns:
        Dict[str, Any]:
        * **blips** (float): Number of blips per trace.
        * **blips_per_second** (float): Number of blips per second.
        * **low_blip_duration** (np.ndarray): Durations in low-voltage state.
        * **high_blip_duration** (np.ndarray): Durations in high-voltage state.
        * **mean_low_blip_duration** (float): Average duration in low state.
        * **mean_high_blip_duration** (float): Average duration in high state.
    """
    low_blip_pts, high_blip_pts = [], []
    start_idx = int(round(t_skip * sample_rate))

    blip_events = [[] for _ in range(len(traces))]
    for k, trace in enumerate(traces):

        idx = start_idx
        trace_above_threshold = trace > threshold_voltage
        trace_below_threshold = ~trace_above_threshold
        while idx < len(trace):
            if trace[idx] < threshold_voltage:
                next_idx = np.argmax(trace_above_threshold[idx:])
                blip_list = low_blip_pts
            else:
                next_idx = np.argmax(trace_below_threshold[idx:])
                blip_list = high_blip_pts

            if next_idx == 0:  # Reached end of trace
                if not ignore_final:
                    next_idx = len(trace) - idx
                    blip_list.append(next_idx)
                    blip_events[k].append(
                        (int(trace[idx] >= threshold_voltage), next_idx)
                    )
                break
            else:
                blip_list.append(next_idx)
                blip_events[k].append((int(trace[idx] >= threshold_voltage), next_idx))
                idx += next_idx

    low_blip_durations = np.array(low_blip_pts) / sample_rate
    high_blip_durations = np.array(high_blip_pts) / sample_rate

    mean_low_blip_duration = (
        np.NaN if not len(low_blip_durations) else np.mean(low_blip_durations)
    )
    mean_high_blip_duration = (
        np.NaN if not len(high_blip_durations) else np.mean(high_blip_durations)
    )

    blips = len(low_blip_durations) / len(traces)

    duration = len(traces[0]) / sample_rate

    return {
        "blips": blips,
        "blip_events": blip_events,
        "blips_per_second": blips / duration,
        "low_blip_durations": low_blip_durations,
        "high_blip_durations": high_blip_durations,
        "mean_low_blip_duration": mean_low_blip_duration,
        "mean_high_blip_duration": mean_high_blip_duration,
    }


def analyse_traces(
    traces: np.ndarray,
    sample_rate: float,
    filtered_shots: np.ndarray = None,
    filter: Union[str, None] = None,
    min_filter_proportion: float = 0.5,
    t_skip: float = 0,
    t_read: Union[float, None] = None,
    t_read_vals: Union[int, None, Sequence] = None,
    segment: str = "begin",
    threshold_voltage: Union[float, None] = None,
    threshold_method: str = "config",
    plot: Union[bool, Axis] = False,
    plot_high_low: Union[bool, Axis] = False,
):
    """ Analyse voltage, up proportions, and blips of acquisition traces

    Args:
        traces: 2D array of acquisition traces.
        sample_rate: acquisition sample rate (per second).
        filter: only use traces that begin in low or high voltage.
            Allowed values are ``low``, ``high`` or ``None`` (do not filter).
        min_filter_proportion: minimum proportion of traces that satisfy filter.
            If below this value, up_proportion etc. are not calculated.
        t_skip: initial time to skip for each trace (ms).
        t_read: duration of each trace to use for calculating up_proportion etc.
            e.g. for a long trace, you want to compare up proportion of start
            and end segments.
        t_read_vals: Optional range of t_read values for which to extract
            up proportion. Can be:
            - an int, indicating that t_read should be uniformly chosen across
              the trace duration.
            - a list of t_read values
        segment: Use beginning or end of trace for ``t_read``.
            Allowed values are ``begin`` and ``end``.
        threshold_voltage: threshold voltage for a ``high`` voltage (blip).
            If not specified, ``find_high_low`` is used to determine threshold.
        threshold_method: Method used to determine the threshold voltage.
            Allowed methods are:

            * **mean**: average of high and low voltage.
            * **{n}\*std_low**: n standard deviations above mean low voltage,
              where n is a float (ignore slash in raw docstring).
            * **{n}\*std_high**: n standard deviations below mean high voltage,
              where n is a float (ignore slash in raw docstring).
            * **config**: Use threshold method provided in
              ``config.analysis.threshold_method`` (``mean`` if not specified)

        plot: Whether to plot traces with results.
            If True, will create a MatPlot object and add results.
            Can also pass a MatPlot axis, in which case that will be used.
            Each trace is preceded by a block that can be green (measured blip
            during start), red (no blip measured), or white (trace was filtered
            out).

    Returns:
        Dict[str, Any]:
        * **up_proportion** (float): proportion of traces that has a blip
        * **end_high** (float): proportion of traces that end with high voltage
        * **end_low** (float): proportion of traces that end with low voltage
        * **num_traces** (int): Number of traces that satisfy filter
        * **filtered_traces_idx** (np.ndarray): 1D bool array,
          True if that trace satisfies filter
        * **voltage_difference** (float): voltage difference between high and
          low voltages
        * **average_voltage** (float): average voltage over all traces
        * **threshold_voltage** (float): threshold voltage for counting a blip
          (high voltage). Is calculated if not provided as input arg.
        * **blips** (float): average blips per trace.
        * **mean_low_blip_duration** (float): average duration in low state
        * **mean_high_blip_duration** (float): average duration in high state
        * **t_read_vals** (list(float)): t_read list if provided as kwarg.
          If t_read_vals was an int, this is converted to a list.
          Not returned if t_read_vals is not set.
        * **up_proportions** (list(float)): up_proportion values for each t_read
          if t_read_vals is provided. Not returned if t_read_vals is not set.

    Note:
        If no threshold voltage is provided, and no two peaks can be discerned,
            all results except average_voltage are set to an initial value
            (either 0 or undefined)
        If the filtered trace proportion is less than min_filter_proportion,
            the results ``up_proportion``, ``end_low``, ``end_high`` are set to an
            initial value
    """
    assert filter in [None, "low", "high"], "filter must be None, `low`, or `high`"

    assert segment in ["begin", "end"], "segment must be either `begin` or `end`"

    # Initialize all results to None
    results = {
        "up_proportion": 0,
        "up_proportion_idxs": np.nan * np.zeros(len(traces)),
        "end_high": 0,
        "end_low": 0,
        "num_traces": 0,
        "filtered_traces_idx": None,
        "voltage_difference": np.nan,
        "average_voltage": np.mean(traces),
        "threshold_voltage": np.nan,
        "blips": None,
        "mean_low_blip_duration": None,
        "mean_high_blip_duration": None,
    }

    # minimum trace idx to include (to discard initial capacitor spike)
    start_idx = int(round(t_skip * sample_rate))

    # Calculate threshold voltage if not provided
    if threshold_voltage is None or np.isnan(threshold_voltage):
        # Histogram trace voltages to find two peaks corresponding to high and low
        high_low_results = find_high_low(
            traces[:, start_idx:], threshold_method=threshold_method, plot=plot_high_low
        )
        results["high_low_results"] = high_low_results
        results["voltage_difference"] = high_low_results["voltage_difference"]
        # Use threshold voltage from high_low_results
        threshold_voltage = high_low_results["threshold_voltage"]

        results["threshold_voltage"] = threshold_voltage

    else:
        # We don't know voltage difference since we skip a high_low measure.
        results["voltage_difference"] = np.nan
        results["threshold_voltage"] = threshold_voltage

    if plot is not False:  # Create plot for traces
        ax = plot
        t_list = np.linspace(0, len(traces[0]) / sample_rate, len(traces[0])) * 1e3

        # Use a diverging colormap that is white at the threshold voltage
        if threshold_voltage:
            divnorm = TwoSlopeNorm(vmin=np.min(traces), vcenter=threshold_voltage, vmax=np.max(traces))
        else:
            divnorm = None

        ax.add(traces, x=t_list, y=np.arange(len(traces), dtype=float), cmap="seismic", norm=divnorm)
        # Modify x-limits to add blips information
        xlim = ax.get_xlim()
        xpadding = 0.05 * (xlim[1] - xlim[0])
        if segment == "begin":
            xpadding_range = [-xpadding + xlim[0], xlim[0]]
            ax.set_xlim(-xpadding + xlim[0], xlim[1])
        else:
            xpadding_range = [xlim[1], xlim[1] + xpadding]
            ax.set_xlim(xlim[0], xlim[1] + xpadding)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Sample')

    if threshold_voltage is None or np.isnan(threshold_voltage):
        logger.debug("Could not determine threshold voltage")
        if plot is not False:
            ax.text(
                np.mean(xlim),
                len(traces) + 0.5,
                "Unknown threshold voltage",
                horizontalalignment="center",
            )
        return results

    # Analyse blips (disabled because it's very slow)
    # blips_results = count_blips(traces=traces,
    #                             sample_rate=sample_rate,
    #                             threshold_voltage=threshold_voltage,
    #                             t_skip=t_skip)
    # results['blips'] = blips_results['blips']
    # results['mean_low_blip_duration'] = blips_results['mean_low_blip_duration']
    # results['mean_high_blip_duration'] = blips_results['mean_high_blip_duration']

    if filter == "low":  # Filter all traces that do not start with low voltage
        filtered_traces_idx = edge_voltage(
            traces,
            edge="begin",
            state="low",
            start_idx=start_idx,
            threshold_voltage=threshold_voltage,
        )
    elif filter == "high":  # Filter all traces that do not start with high voltage
        filtered_traces_idx = edge_voltage(
            traces,
            edge="begin",
            state="high",
            start_idx=start_idx,
            threshold_voltage=threshold_voltage,
        )
    else:  # Do not filter traces
        filtered_traces_idx = np.ones(len(traces), dtype=bool)

    if filtered_shots is not None:
        filtered_traces_idx = filtered_traces_idx & filtered_shots

    results["filtered_traces_idx"] = filtered_traces_idx
    filtered_traces = traces[filtered_traces_idx]
    results["num_traces"] = len(filtered_traces)

    if len(filtered_traces) / len(traces) < min_filter_proportion:
        logger.debug(f"Not enough traces start {filter}")

        if plot is not False:
            ax.pcolormesh(
                xpadding_range,
                np.arange(len(traces) + 1) - 0.5,
                filtered_traces.reshape(1, -1),
                cmap="RdYlGn",
            )
            ax.text(
                np.mean(xlim),
                len(traces) + 0.5,
                f"filtered traces: {len(filtered_traces)} / {len(traces)} = "
                f"{len(filtered_traces) / len(traces):.2f} < {min_filter_proportion}",
                horizontalalignment="center",
            )
        return results

    # Determine all the t_read's for which to determine up proportion
    total_duration = filtered_traces.shape[1] / sample_rate
    if t_read is None:  # Only use a time segment of each trace
        t_read = total_duration

    if isinstance(t_read_vals, int):
        # Choose equidistantly spaced t_read values
        t_read_vals = np.linspace(total_duration/t_read_vals, total_duration, num=t_read_vals)
    elif t_read_vals is None:
        t_read_vals = []
    elif not isinstance(t_read_vals, Sequence):
        raise ValueError('t_read_vals must be an int, Sequence, or None')

    # Determine up_proportion for each t_read
    up_proportions = []
    for k, t_read_val in enumerate(list(t_read_vals) + [t_read]):

        read_pts = int(round(t_read_val * sample_rate))
        if segment == "begin":
            segmented_filtered_traces = filtered_traces[:, :read_pts]
        else:
            segmented_filtered_traces = filtered_traces[:, -read_pts:]

        # Calculate up proportion of traces
        up_proportion_idxs = find_up_proportion(
            segmented_filtered_traces,
            start_idx=start_idx,
            threshold_voltage=threshold_voltage,
            return_array=True,
        )
        up_proportion = sum(up_proportion_idxs) / len(traces)
        if k == len(t_read_vals):
            results["up_proportion"] = up_proportion
            results["up_proportion_idxs"][filtered_traces_idx] = up_proportion_idxs
        else:
            up_proportions.append(up_proportion)

    if t_read_vals is not None:
        results['up_proportions'] = up_proportions
        results['t_read_vals'] = t_read_vals

    # Calculate ratio of traces that end up with low voltage
    idx_end_low = edge_voltage(
        segmented_filtered_traces,
        edge="end",
        state="low",
        threshold_voltage=threshold_voltage,
    )
    results["end_low"] = np.sum(idx_end_low) / len(segmented_filtered_traces)

    # Calculate ratio of traces that end up with high voltage
    idx_end_high = edge_voltage(
        segmented_filtered_traces,
        edge="end",
        state="high",
        threshold_voltage=threshold_voltage,
    )
    results["end_high"] = np.sum(idx_end_high) / len(segmented_filtered_traces)

    if plot is not False:
        # Plot information on up proportion
        up_proportion_arr = 2 * up_proportion_idxs - 1
        up_proportion_arr[~filtered_traces_idx] = 0
        up_proportion_arr = up_proportion_arr.reshape(-1, 1)  # Make array 2D

        mesh = ax.pcolormesh(
            xpadding_range,
            np.arange(len(traces) + 1) - 0.5,
            up_proportion_arr,
            cmap="RdYlGn",
        )
        mesh.set_clim(-1, 1)

        # Add vertical line for t_read
        if t_read is not None:
            ax.vlines(t_read*1e3, -0.5, len(traces + 0.5), lw=2, linestyle="--", color="orange")
            ax.text(
                t_read*1e3,
                len(traces) + 0.5,
                f"t_read={t_read*1e3} ms",
                horizontalalignment="center",
                verticalalignment="bottom",
            )
            ax.text(
                t_skip*1e3,
                len(traces) + 0.5,
                f"t_skip={t_skip*1e6:.0f} us",
                horizontalalignment="center",
                verticalalignment="bottom",
            )

    return results
