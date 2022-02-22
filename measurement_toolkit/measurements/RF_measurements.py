import time
import numpy as np
from matplotlib import pyplot as plt
from functools import partial
from IPython.display import clear_output

from qcodes.instrument.parameter import ParameterWithSetpoints, Parameter
from qcodes.utils.dataset.doNd import do0d
from qcodes.utils.validators import Numbers, Arrays


class BiasScanRF():
    def __init__(
        self, 
        RF_lockin, 
        bias_channel, 
        demodulator_name='demod0',
        num_points=251, 
        num_traces=1,
        duration=0.2, 
        V_range=1e-3, 
        V_scale=1e3,
        phase_shift=0,
        ):
        self.RF_lockin = RF_lockin
        self.daq = RF_lockin.daq
        self.demodulator_name = demodulator_name
        self.bias_channel = bias_channel

        self.num_points = num_points
        self.num_traces = num_traces
        self.duration = duration
        self.V_range = V_range
        self.V_scale = V_scale
        self.phase_shift = phase_shift

        self.acquisition_signals = []
        self.results = {}
        self._raw_results = None

    @property
    def sweep_values(self):
        if self.V_range is None or self.num_points is None:
            return None
        else:
            return np.linspace(-self.V_range, self.V_range, self.num_points)

    @property
    def voltages(self):
        return self.sweep_values

    def setup_RF_lockin(self, num_points=None, num_traces=None, duration=None):
        if num_points is not None:
            self.num_points = num_points
        if num_traces is not None:
            self.num_traces = num_traces
        if duration is not None:
            self.duration = duration

        self.daq.signals_clear()
        demod_x = self.daq.signals_add(self.demodulator_name, signal_type='X')
        demod_y = self.daq.signals_add(self.demodulator_name, signal_type='Y')
        self.acquisition_signals = [demod_x, demod_y]

        self.daq.trigger(self.demodulator_name, 'trigin3')
        self.daq.type('hardware')  # Wait for trigger
        self.daq.grid_mode('linear')
        self.daq.duration(self.duration)
        self.daq.grid_cols(self.num_points)
        self.daq.grid_rows(self.num_traces)

    def setup_bias_channel(self):
        # Turn on sync trigger for ramp
        self.bias_channel.sync(1)
        self.bias_channel.sync_delay(0e-3)  # The sync pulse delay (s), must be positive
        self.bias_channel.sync_duration(1e-3)  # The sync pulse duration (secs). Default is 10 ms.
        self.bias_channel.v.step = None
        self.bias_channel.v.inter_delay = 0

    def setup(self, num_points=None, num_traces=None, duration=None):
        self.setup_RF_lockin(num_points=num_points, num_traces=num_traces, duration=duration)
        self.setup_bias_channel()

    def start_RF_lockin_acquisition(self):
        self.daq._daq_module._set("endless", 0)
        self.daq._daq_module._set("clearhistory", 1)
        for path in self.daq.signals:
            self.daq._daq_module._module.subscribe(path)
            # print(f'subscribed to {path}')
        self.daq._daq_module._module.execute()

    def ramp_bias(self, V_range=None, V_scale=None, duration=None, V_final=0):
        if V_range is not None:
            self.V_range = V_range
        if V_scale is not None:
            self.V_scale = V_scale
        if duration is not None:
            self.duration = duration

        V_start = -self.V_range*self.V_scale
        V_stop = self.V_range*self.V_scale

        # Verify that voltages are within range
        if self.bias_channel.v.vals is not None:
            min_value = self.bias_channel.v.vals._min_value
            max_value = self.bias_channel.v.vals._max_value
            assert min_value < min(V_start, V_stop) < max(V_start, V_stop) < max_value

        # Go to initial voltage an wait a little bit
        self.bias_channel.sync(0)
        self.bias_channel.v(V_start)
        self.bias_channel.sync(1)

        # Ramp voltage
        self.bias_channel.parent.ramp_voltages(
            channellist=[self.bias_channel.id],
            v_startlist=[V_start],
            v_endlist=[V_stop],
            ramptime=self.duration
        )

        # Sleep ramp duration. Note that the ramp will have finished ~30ms earlier due to overhead)
        time.sleep(self.duration)

        # Ramp ch1 back to 0 V at rate specified by slope.
        # We temporarily disable sending a sync pulse
        if V_final is not None:
            self.bias_channel.sync(0)
            self.bias_channel.v(V_final)
            self.bias_channel.sync(1)

    def wait_acquisition_complete(self, timeout=5, t_start=None, silent=True):
        if t_start is None:
            t_start = time.perf_counter()

        while time.perf_counter() - t_start < timeout:
            if self.daq._daq_module._module.finished():
                if not silent:
                    print(f'Acquisition took {(time.perf_counter() - t_start) * 1e3:.0f} ms')
                break
            time.sleep(0.03)
        else:
            raise RuntimeError(f'Acquisition failed with timeout {timeout} s')

    def process_acquisition(self, simplify=True):
        self._raw_results = self.daq._daq_module._module.read(flat=True)
        self.daq._daq_module._module.finish()
        self.daq._daq_module._module.unsubscribe("*")
        self.daq._daq_module._get_result_from_dict(self._raw_results)
        
        for key in self.acquisition_signals:
            result = self._raw_results[key][0]['value']

            if simplify:
                key = key.split('.')[-2]

            if self.num_traces == 1:
                # Only use the first trace
                result = result[0]
            
            self.results[key] = result

        self.results['voltages'] = self.voltages
        self.results['phase_shifted'] = (self.results['x'] + 1.j*self.results['y']) * np.exp(-1.j*self.phase_shift)
        return self.results

    def acquire(
        self, 
        save_results=True,
        setup=True, 
        finalize=True, 
        num_points=None, 
        num_traces=None, 
        duration=None, 
        V_range=None, 
        V_final=0,
        ):
        if setup:
            self.setup(num_points=num_points, num_traces=num_traces, duration=duration)

        self.start_RF_lockin_acquisition()
        t_start = time.perf_counter()

        self.ramp_bias(V_range=V_range, duration=duration, V_final=V_final)

        if finalize:
            self.wait_acquisition_complete(timeout=5, t_start=t_start)
            
            results = self.process_acquisition()

        if save_results:
            setpoints_V_bias = Parameter(
                'V_bias', unit='V', get_cmd=partial(getattr, self, 'voltages'),
                vals=Arrays(shape=(self.num_points, ))
            )
            measure_parameter = ParameterWithSetpoints(
                'RF_signal', 
                unit='V', 
                setpoints=(setpoints_V_bias,),
                vals=Arrays(shape=(self.num_points, ), valid_types=(np.complex, np.float)),
                get_cmd=lambda: results['phase_shifted']
            )
            do0d(measure_parameter, measurement_name='1D:bias_scan_RF')

        return results

    def plot_1D_results(self, results):
        fig, ax = plt.subplots()
        ax.plot(self.voltages, -results['x'])
        ax2 = ax.twinx()
        ax2.plot(self.voltages, results['y'], color='C1')
        return fig, [ax, ax2]
        
    def plot_2D_results(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        x = self.voltages
        y = np.arange(self.num_traces)
        
        ax = axes[0]
        mesh = ax.pcolormesh(x, y, self.results['x'], shading='auto')
        ax.set_title('Real signal')
        plt.colorbar(mesh, ax=ax)

        ax = axes[1]
        mesh = ax.pcolormesh(x, y, self.results['y'], shading='auto')
        ax.set_title('Imag signal')
        plt.colorbar(mesh, ax=ax)

        return fig, axes

    def sweep_gate(
        self, 
        gate, 
        V_start, 
        V_stop, 
        step=None, 
        num=None, 
        save_results=True, 
        silent=True
    ):
        if num is None:
            num = int(np.ceil(abs((V_stop - V_start) / step))) + 1
        voltages = np.linspace(V_start, V_stop, num)

        self.setup(num_traces=len(voltages))
        self.start_RF_lockin_acquisition()
        t0 = time.perf_counter()

        for V in voltages:
            gate(V)
            self.ramp_bias()

            if not silent:
                clear_output()
                print(f'Finished trace: {self.daq._daq_module._module.finished()} ({self.daq._daq_module._module.progress()[0]*100:.1f}%)')

        self.wait_acquisition_complete(silent=silent)
        results = self.process_acquisition()

        if save_results:
            setpoints_gate = Parameter(
                gate.name, label=gate.label, unit='V', 
                get_cmd=lambda: voltages,
                vals=Arrays(shape=(num, ))
            )
            setpoints_V_bias = Parameter(
                'V_bias', unit='V', get_cmd=partial(getattr, self, 'voltages'),
                vals=Arrays(shape=(self.num_points,))
            )
            measure_parameter = ParameterWithSetpoints(
                'RF_signal', 
                unit='V', 
                setpoints=(setpoints_gate, setpoints_V_bias),
                vals=Arrays(shape=(num, self.num_points), valid_types=(np.complex, np.float)),
                get_cmd=lambda: results['phase_shifted']
            )
            do0d(measure_parameter, measurement_name=f'2D:bias_scan_RF_{gate.name}')

