import time
import numpy as np
import warnings

import qcodes as qc
from qcodes import Measurement
from qcodes.utils.dataset.doNd import LinSweep, dond
from qcodes.instrument.parameter import _BaseParameter


class TimeParameter(qc.Parameter):
    def __init__(self, name='time', unit='s', **kwargs):
        super().__init__(name=name, unit=unit, **kwargs)
        self.t_start = None
        self.scale = {'s': 1, 'ms': 1000, 'us': 1e6, 'ns': 1e9}[unit]

        self.reset()

    def reset(self):
        self.t_start = time.perf_counter()

    def get_raw(self):
        t_delta = time.perf_counter() - self.t_start
        t_delta * self.scale
        return t_delta


    def set_raw(self):
        pass


def measure_repeatedly(
    duration,
    measure_params=None,
    t_delta=None,
):
    station = qc.Station.default
    assert station is not None, "No station initialized"

    if measure_params is None:
        measure_params = station.measure_params
    elif isinstance(measure_params, _BaseParameter):
        measure_params = [measure_params]
    measure_label = '_'.join(param.name for param in measure_params)

    time_parameter = TimeParameter()

    meas = Measurement(station=station, name=f'repeated_measure_{measure_label}')

    meas.register_parameter(time_parameter)
    for param in measure_params:
        meas.register_parameter(param, setpoints=(time_parameter, ))

    emit_warning = True
    measurement_durations = []
    measurement_intervals = []
    with meas.run() as datasaver:
        time_parameter.reset()

        loop_index = 1
        while time_parameter() < duration:
            t0 = time.perf_counter()

            for param in measure_params:
                datasaver.add_result(
                    (time_parameter, time_parameter()),
                    (param, param())
                )
            t1 = time.perf_counter()
            measurement_durations.append(t1 - t0)
            if t_delta is not None:
                target_duration = t_delta * loop_index
                sleep_duration = target_duration - time_parameter()
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
                elif emit_warning:
                    warnings.warn(f'Measurement time {t1 - t0} longer than wait time t_delta = {t_delta}. Sleep duration {sleep_duration}')
                    emit_warning = False

            measurement_intervals.append(time.perf_counter() - t0)
            loop_index += 1

    return {
        'dataset': datasaver._dataset,
        'average_duration': np.mean(measurement_durations),
        'average_interval': np.mean(measurement_intervals)
    }
