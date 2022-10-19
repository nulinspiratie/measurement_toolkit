import time
from time import sleep
import numpy as np
from matplotlib import pyplot as plt
from functools import partial
from IPython.display import clear_output

from qcodes.instrument.parameter import ParameterWithSetpoints, Parameter
from qcodes.dataset import MeasurementLoop, Sweep
from qcodes.utils.dataset.doNd import do0d
from qcodes.utils.validators import Numbers, Arrays


def get_slew_rate(qdac_channel):
    if qdac_channel.v.step is None:
        return None
    elif qdac_channel.v.inter_delay == 0:
        return None
    else:
        return qdac_channel.v.step / qdac_channel.v.inter_delay


class QDacSweeper():
    def __init__(self, name, qdac_channel):
        self.name = name
        self.qdac_channel = qdac_channel
        self.qdac = self.qdac_channel.parent

    def setup(self):
        # Turn on sync trigger for ramp
        self.qdac_channel.sync(1)
        self.qdac_channel.sync_delay(0e-3)  # The sync pulse delay (s), must be positive
        self.qdac_channel.sync_duration(1e-3)  # The sync pulse duration (secs). Default is 10 ms.
        self.qdac_channel.v.step = None
        self.qdac_channel.v.inter_delay = 0

    def sweep(self, V_start, V_stop, duration=None, V_final=None):
        # Verify that voltages are within range
        if self.qdac_channel.v.vals is not None:
            min_value = self.qdac_channel.v.vals._min_value
            max_value = self.qdac_channel.v.vals._max_value
            assert min_value <= min(V_start, V_stop) < max(V_start, V_stop) <= max_value

            if V_final is not None:
                assert min_value <= V_final <= max_value

        # Go to initial voltage an wait a little bit
        self.qdac_channel.sync(0)
        self.qdac_channel.v(V_start)
        self.qdac_channel.sync(1)

        # Ramp voltage
        self.qdac.ramp_voltages(
            channellist=[self.qdac_channel.id],
            v_startlist=[V_start],
            v_endlist=[V_stop],
            ramptime=duration
        )

        # Sleep ramp duration. Note that the ramp will have finished ~30ms earlier due to overhead)
        time.sleep(duration)

        # Ramp ch1 back to 0 V at rate specified by slope.
        # We temporarily disable sending a sync pulse
        if V_final is not None:
            self.qdac_channel.sync(0)
            self.qdac_channel.v(V_final)
            self.qdac_channel.sync(1)


class QDac2Sweeper():
    trigger_width = 0.1e-3

    def __init__(self, name, qdac_channel, trigger_channel, max_ramp_rate=0, delay_start=None, scale=None, dV_limit=None):
        self.name = name
        self.qdac_channel = qdac_channel
        self.trigger_channel = trigger_channel
        self.scale = scale or qdac_channel.v.scale
        self.max_ramp_rate = max_ramp_rate
        self.delay_start = delay_start
        self.qdac = self.qdac_channel.parent
        self.dV_limit = None
        self.silent = True

        self.sweep_settings = None

        # Ensure we're using a QDAC2
        from qcodes_contrib_drivers.drivers.QDevil.QDAC2 import QDac2
        assert isinstance(self.qdac, QDac2), "gate DAC instrument must be a QDac-II"

    def setup_ramp(self, V_start, V_stop, duration, repetitions=1, delay_start=None, num=4001, plot=False, silent=True):
        """Ramp QDAC-II gate voltage"""
        dt = duration / (num-1)
        assert dt > 2e-6 - 1e-9, f"Time step {dt*1e6:.1f}us must be >2us, please decrease 'num'"
        
        self.sweep_settings = {
            'V_start': V_start,
            'V_stop': V_stop,
            'duration': duration,
            'repetitions': repetitions,
            'delay_start': delay_start,
            'num': num
        }

        # Record voltage before sweep
        V0 = self.qdac_channel.v() * self.scale

        #Scale V_start, V_stop
        V_start_unscaled, V_start = V_start, V_start * self.scale
        V_stop_unscaled, V_stop = V_stop, V_stop * self.scale

        # Generate sweep voltages
        voltages = []

        # Optionally add sweep from current gate voltage V0 to ramp start V_start
        if self.max_ramp_rate:
            t_ramp_start = abs(V_start - V0) / self.max_ramp_rate
            num_start = int(np.ceil(t_ramp_start / dt)) + 1
            voltages += list(np.linspace(V0, V_start, num_start))

        # Optionally add sweep that remains at V_start for delay_start
        if delay_start is None:
            delay_start = self.delay_start
        if delay_start:
            num_delay_start = int(np.ceil(delay_start / dt))
            voltages += list(np.repeat(V_start, num_delay_start))

        # Add ramp sweep
        trigger_delay = len(voltages) * dt
        voltages += list(np.linspace(V_start, V_stop, num))

        # Optionally add sweep from final gate voltage V_stop to original gate voltage
        if self.max_ramp_rate:
            t_ramp_stop = abs(V_stop - V0) / self.max_ramp_rate
            num_stop = int(np.ceil(t_ramp_stop / dt)) + 1
            voltages += list(np.linspace(V_stop, V0, num_stop))
        else:
            # Return to initial voltage
            voltages += [V0]
        
        voltages = np.array(voltages)

        # Ensure number of points does not exceed limit
        assert len(voltages) < 1e5, f"Number of points {len(voltages)} exceeds limit 100000"

        # Ensure all voltages are within allowed values
        if self.qdac_channel.v.vals is not None:
            scale = self.scale or 1
            V_min = getattr(self.qdac_channel.v.vals, '_min_value', -10) * scale
            V_max = getattr(self.qdac_channel.v.vals, '_max_value', 10) * scale
            V_min = max(V_min, -10)
            V_max = min(V_max, 10)
            assert np.all((V_min <= voltages) & (voltages <= V_max)), f"Voltages are out of range {V_min} <= voltages <= {V_max}"
        if self.dV_limit is not None:
            dV_limit = self.dV_limit * self.scale
            assert np.all((voltages - V0) < dV_limit), f"Voltages are not within {self.dV_limit} of current value {self.qdac_channel.v()}"


        # Optionally plot results
        if plot:
            self.plot_ramp(voltages=voltages, dt=dt, trigger_delay=trigger_delay)
            
        # Program QDac with list of voltages
        self.sweep_object = self.qdac_channel.dc_list(repetitions=repetitions,voltages=voltages,dwell_s=dt)

        # Optionally add trigger
        if self.trigger_channel is not None:
            self.trigger_channel.width_s(self.trigger_width)
            self.trigger_channel.polarity('norm')
            self.qdac_channel.parent.free_all_triggers()
            trigger = self.sweep_object.start_marker()
            self.trigger_channel.delay_s(trigger_delay)
            self.trigger_channel.source_from_trigger(trigger)

        if not silent:
            print(f'Number of points={len(voltages)}, duration={len(voltages)*dt*1e3:.3f} ms, dt={dt*1e6:.1f} us')
            print(f'Voltage range: V_min={min(voltages):.3f}, V_max={max(voltages):.3f}, all within range [{V_min:.3f}, {V_max:.3f}]')

        return self.sweep_object
        
    def plot_ramp(self,voltages, dt, trigger_delay=None):
        fig, ax = plt.subplots(figsize=(10,4))
        t_list = np.arange(len(voltages)) * dt
        ax.plot(t_list, voltages)
        ax.grid('on')

        if trigger_delay is not None:
            ax_trigger = ax.twinx()
            ax_trigger.plot(trigger_delay + np.array([0,0, .2e-3, .2e-3]), [0,5.5,5.5,0], color='r', alpha=0.5)
            
        return fig, ax

    def sweep(self, block=False):
        self.sweep_object.start()

        if block:
            while self.sweep_object.cycles_remaining():
                time.sleep(10e-3)

    def disable_trigger(self):
        trigger_id = int(self.trigger_channel.short_name[-1])
        self.qdac.write(f'outp:trig{trigger_id}:sour hold')


class GateScanRF():
    def __init__(
        self, 
        RF_lockin, 
        sweeper,
        demodulator_idx=0,
        num_points=251, 
        t_int=1e-3,
        num_traces=1,
        phase_shift=0,
    ):
        self.RF_lockin = RF_lockin
        self.daq = RF_lockin.session.modules.daq
        self.daq_raw = RF_lockin.session.modules.daq.raw_module
        self.demodulator_idx = demodulator_idx
        self.demodulator = RF_lockin.demods[demodulator_idx]
        self.sweeper = sweeper

        self.num_points = num_points
        self.num_traces = num_traces
        self.t_int = t_int
        self.phase_shift = phase_shift

        self.acquisition_signals = []
        self.results = {}
        self._raw_results = None
        self._t_acquisition_start = None

        self.RF_parameters = {
            'output_amplitude': self.RF_lockin.sigouts[0].amplitudes[0].value,
            't_int': self.demodulator.timeconstant,  # sets bandwidth to ~ 1/time_constant/4pi
            'output_enabled': self.RF_lockin.sigouts[0].on
        }
        

    @property
    def sweep_values(self):
        V_start = self.sweeper.sweep_settings.get('V_start', None)
        V_stop = self.sweeper.sweep_settings.get('V_stop', None)
        num = self.num_points
        if None not in [V_start, V_stop, num]:
            return np.linspace(V_start, V_stop, num)
        else:
            return None

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

        self.daq_raw.unsubscribe('*')
        self.acquisition_signals = [
            (self.demodulator.sample.zi_node + '.x').lower(),
            (self.demodulator.sample.zi_node + '.y').lower(),
        ]
        for signal in self.acquisition_signals:
            self.daq_raw.subscribe(signal)

        # self.daq.trigger(f'demods{self.demodulator_idx}', 'trigin3')
        self.daq.triggernode(self.demodulator.sample.zi_node + '.TrigIn3')
        # self.daq.type('hardware')  # Wait for trigger
        self.daq.type(6) # 6 apparently means HW trigger  # TODO replace me back
        self.daq.grid.mode('linear')
        self.daq.duration(self.duration)
        self.daq.grid.cols(self.num_points)
        self.daq.grid.rows(self.num_traces)

        t_int = duration / num_points
        self.demodulator.timeconstant(t_int)  # sets bandwidth to ~ 1/time_constant/4pi

    def setup(self, V_start, V_stop, t_int, num_points, num_traces, plot=False):
        duration = t_int * num_points

        # Setup RF lockin
        self.setup_RF_lockin(num_points=num_points, num_traces=num_traces, duration=duration)

        # Setup QDac sweeper
        self.sweeper.setup_ramp(
            V_start=V_start, 
            V_stop=V_stop, 
            duration=duration, 
            repetitions=1, 
            num=max(4001, num_points), 
            plot=plot, 
            silent=True
        )

    def start_RF_lockin_acquisition(self):
        self.daq.endless(0)
        self.daq.clearhistory(1)
        # for path in self.acquisition_signals:
        #     self.daq.subscribe(path)
            # print(f'subscribed to {path}')
        self.daq_raw.execute()
        self._t_acquisition_start = time.perf_counter()

    def wait_acquisition_complete(self, timeout=5, t_start=None, silent=True):
        if t_start is None:
            t_start = self._t_acquisition_start

        t0 = time.perf_counter()

        while time.perf_counter() - t0 < timeout:
            if self.daq_raw.finished():
                if not silent:
                    print(f'Acquisition took {(time.perf_counter() - t_start) * 1e3:.0f} ms')
                break
            time.sleep(0.03)
        else:
            raise RuntimeError(f'Acquisition failed with timeout {timeout} s')

    def process_acquisition(self, simplify=True, remove_singleton=True):
        self._raw_results = self.daq_raw.read(flat=True)
        self.daq_raw.finish()
        # self.daq_raw.unsubscribe("*")  # TODO is this needed?
        # self.daq._daq_module._get_result_from_dict(self._raw_results)
        
        for key in self.acquisition_signals:
            result = self._raw_results[key][0]['value']

            if simplify:
                key = key.split('.')[-1]

            if self.num_traces == 1 and remove_singleton:
                # Only use the first trace
                result = result[0]
            
            self.results[key] = result

        self.results['voltages'] = self.voltages
        self.results['phase_shifted'] = (self.results['x'] + 1.j*self.results['y']) * np.exp(-1.j*self.phase_shift)
        return self.results

    def acquire(
        self, 
        save_results=True,
        finalize=True, 
        delay=0.01,
        silent=True
        ):

        # Start RF acquisition
        duration = self.sweeper.sweep_settings['duration']
        self.start_RF_lockin_acquisition()
        if not silent:
            print('Started RF acquisition')
        time.sleep(delay)

        # Perform sweep
        # for k in range(3):
        self.sweeper.sweep()
        if not silent:
            print('Performed gate sweep')
            # time.sleep(0.3)

        if finalize:
            timeout = max(1.5*duration, 5)
            self.wait_acquisition_complete(timeout=timeout, silent=silent)
            results = self.process_acquisition()
        else:
            results = None

        if save_results:
            # if 
            # with MeasurementLoop('1D:{self.sweeper.name}_scan_RF') as msmt:
            #     for k in Sweep

            setpoints_gate = Parameter(
                self.sweeper.name, unit='V', get_cmd=partial(getattr, self, 'voltages'),
                vals=Arrays(shape=(self.num_points, ))
            )
            measure_parameter = ParameterWithSetpoints(
                'RF_signal', 
                unit='V', 
                setpoints=(setpoints_gate,),
                vals=Arrays(shape=(self.num_points, ), valid_types=(np.complex, np.float)),
                get_cmd=lambda: results['phase_shifted']
            )
            do0d(measure_parameter, measurement_name=f'1D:{self.sweeper.name}_scan_RF')

        return results

    def sweep(
        self,
        *voltages,
        step=None,
        num=None,
        t_int=None,
        duration=None,
        repetitions=1,
        sweep=None,
        setup=True,
        execute=True,
        silent=True,
        plot=False
    ):
        assert repetitions == 1, "Multiple repetitions not yet implemented"

        # Extract args
        if len(voltages) == 0:
            V_start, V_stop = self.V_start, self.V_stop
        elif len(voltages) == 1:
            V_start, V_stop = -voltages[0], voltages[0]
        elif len(voltages) == 2:
            V_start, V_stop = voltages
        elif len(voltages) == 3:
            V_start, V_stop, num = voltages
        else:
            raise ValueError('V_start, V_stop must be defined')

        # Specify number of points / step size
        if step is not None:
            num = np.ceil(abs(V_stop - V_start) / step) + 1
        elif num is None:
            num = self.num_points

        # Specify integration time / total duration
        if duration is not None:
            t_int = duration / num
        elif t_int is None:
            t_int = self.t_int

        if sweep is not None:
            num_traces = repetitions * len(sweep)
        else:
            num_traces = repetitions

        # Ensure we always have an iterable to sweep over
        if sweep is None:
            sequence = [None]
        else:
            sequence = sweep.sequence  # Remove once new measurementLoop is added


        # Setup QDac and RF lockin
        if setup:
            self.setup(
                V_start=V_start, 
                V_stop=V_stop, 
                t_int=t_int, 
                num_points=num, 
                num_traces=num_traces
            )
        duration = self.sweeper.sweep_settings['duration']
        timeout = max(1.5*duration, 5)

        # Perform measurement        
        if execute:
            self.start_RF_lockin_acquisition()
            time.sleep(0.05)

            for val in sequence:
                if sweep is not None:
                    if sweep.parameter is not None:
                        sweep.parameter(val)
                    if sweep.delay:
                        sleep(sweep.delay)

                self.sweeper.sweep(block=True)

                if not silent:
                    # clear_output()
                    print(f'Finished trace: {self.daq_raw.finished()} ({self.daq_raw.progress()[0]*100:.1f}%)')

            self.wait_acquisition_complete(timeout=timeout, silent=silent)
            results = self.process_acquisition(remove_singleton=False)

            # Store data
            setpoints_slow_gate = Parameter(
                sweep.parameter.name, label=sweep.parameter.name, unit='V', 
                get_cmd=lambda: sweep.sequence,
                vals=Arrays(shape=(len(sweep), ))
            )
            setpoints_fast_gate = Parameter(
                self.sweeper.name, unit='V', get_cmd=partial(getattr, self, 'voltages'),
                vals=Arrays(shape=(self.num_points,))
            )
            measure_parameter = ParameterWithSetpoints(
                'RF_signal', 
                unit='V', 
                setpoints=(setpoints_slow_gate, setpoints_fast_gate),
                vals=Arrays(shape=(len(sweep), self.num_points), valid_types=(np.complex, np.float)),
                get_cmd=lambda: results['phase_shifted']
            )
            data_qcodes, _, _ = do0d(measure_parameter, measurement_name=f'2D:{sweep.parameter.name}_{self.sweeper.name}_scan_RF')
            data = data_qcodes.to_xarray_dataset()

            # with MeasurementLoop('RF_measurement') as msmt:
            #     self.start_RF_lockin_acquisition()
            #     time.sleep(0.05)

            #     for val in sweep:
            #         self.sweeper.sweep(block=True)

            #         if not silent:
            #             # clear_output()
            #             print(f'Finished trace: {self.daq_raw.finished()} ({self.daq_raw.progress()[0]*100:.1f}%)')

            #     self.wait_acquisition_complete(silent=silent)
            #     results = self.process_acquisition(remove_singleton=False)
            #     signal = results['phase_shifted']

            #     fake_sweep = Sweep(sweep.sequence, name='bla') if isinstance(sweep, Sweep) else [None]
            #     for k, _ in enumerate(fake_sweep):
            #         for kk, V in enumerate(Sweep(self.voltages, name=self.sweeper.name, unit='V')):
            #             msmt.measure(np.real(signal[k,kk]), 'RF_signal_I', unit='V')
            #             msmt.measure(np.imag(signal[k,kk]), 'RF_signal_Q', unit='V')
                
            if not silent:
                print(f'Finished trace: {self.daq_raw.finished()} ({self.daq_raw.progress()[0]*100:.1f}%)')

            if plot:
                fig, axes = plt.subplots(1, 2, figsize=(14,5))
                arr = data.RF_signal.transpose()
                ax = axes[0]
                np.real(arr).plot(ax=ax)
                ax = axes[1]
                np.imag(arr).plot(ax=ax)
                fig.tight_layout()

            return data

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

    # def sweep_gate(
    #     self, 
    #     gate, 
    #     *voltages,
    #     step=None,
    #     num=251,
    #     duration=0.2, 
    #     save_results=True, 
    #     silent=True
    # ):
    #     if num is None:
    #         num = int(np.ceil(abs((V_stop - V_start) / step))) + 1
    #     voltages = np.linspace(V_start, V_stop, num)

    #     self.setup(num_traces=len(voltages))
    #     self.start_RF_lockin_acquisition()
    #     t0 = time.perf_counter()

    #     for V in voltages:
    #         gate(V)
    #         self.sweeper.sweep(V_start=self.V_start, V_stop=self.V_stop, duration=self.duration)

    #         if not silent:
    #             clear_output()
    #             print(f'Finished trace: {self.daq._daq_module._module.finished()} ({self.daq._daq_module._module.progress()[0]*100:.1f}%)')

    #     self.wait_acquisition_complete(silent=silent)
    #     results = self.process_acquisition()

    #     if save_results:
    #         setpoints_slow_gate = Parameter(
    #             gate.name, label=gate.label, unit='V', 
    #             get_cmd=lambda: voltages,
    #             vals=Arrays(shape=(num, ))
    #         )
    #         setpoints_fast_gate = Parameter(
    #             self.sweeper.name, unit='V', get_cmd=partial(getattr, self, 'voltages'),
    #             vals=Arrays(shape=(self.num_points,))
    #         )
    #         measure_parameter = ParameterWithSetpoints(
    #             'RF_signal', 
    #             unit='V', 
    #             setpoints=(setpoints_slow_gate, setpoints_fast_gate),
    #             vals=Arrays(shape=(num, self.num_points), valid_types=(np.complex, np.float)),
    #             get_cmd=lambda: results['phase_shifted']
    #         )
    #         do0d(measure_parameter, measurement_name=f'2D:{gate.name}_{self.sweeper.name}_scan_RF')
