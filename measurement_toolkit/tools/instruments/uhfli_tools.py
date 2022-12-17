from matplotlib import pyplot as plt
import numpy as np
from functools import partial
from time import sleep, perf_counter
from typing import Iterable
from contextlib import contextmanager

import qcodes as qc
from qcodes.instrument import InstrumentBase
from qcodes.parameters import Parameter, ManualParameter
from qcodes.dataset.measurement_loop import running_measurement, MeasurementLoop, Sweep

from measurement_toolkit.tools.plot_tools import plot_data
from measurement_toolkit.tools.data_tools import load_data
from measurement_toolkit.tools.analysis_tools import optimize_IQ


from contextlib import contextmanager

@contextmanager
def disable_lockin_outputs(activate=True, delay=0.2):
    if not activate:
        yield
        return
    station = qc.Station.default
    amplitudes = []
    for lockin in station.lockins.values():
        amplitudes.append(lockin.amplitude())
        lockin.amplitude(lockin.amplitude.vals._min_value)
    
    sleep(delay)

    try:
        yield
    finally:
        for amplitude, lockin in zip(amplitudes, station.lockins.values()):
            lockin.amplitude(amplitude)
_disable_lockin_outputs = disable_lockin_outputs


class UHFLI_Interface(InstrumentBase):
    _acquisition_durations = []
    def __init__(
        self, 
        uhfli, 
        name='RF_lockin', 
        oscillator_idx=0, 
        demodulator_idx=0, 
        input_idx=0, 
        output_idx=0,
        phase_shift=0
    ):
        super().__init__(name=name)

        self.uhfli = uhfli
        self.daq = uhfli.session.modules.daq
        self.daq_raw = uhfli.session.modules.daq.raw_module

        self.oscillator_idx = oscillator_idx
        self.oscillator = self.uhfli.oscs[oscillator_idx]

        self.demodulator_idx = demodulator_idx
        self.demodulator = self.uhfli.demods[demodulator_idx]

        self.input_idx = input_idx
        self.input_channel = self.uhfli.sigins[input_idx]

        self.output_idx = output_idx
        self.output_channel = self.uhfli.sigouts[output_idx]

        self._signal = None  # Updated by self.get_signal()

        self.output_amplitude = self.output_channel.amplitudes[oscillator_idx].value
        self.frequency = self.oscillator.freq
        self.time_constant = self.demodulator.timeconstant
        for key in ['output_amplitude', 'frequency', 'time_constant']:
            self.parameters[key] = getattr(self, key)

        # Trace-related attributes
        self._trace_context = False
        self._acquisition_signals = []
        self._timeout = None

        self.measurement_params = {}
        self.measurement_params['I'] = Parameter(
            name='RF_inphase',
            label='RF signal In-phase',
            unit='V',
            get_cmd=lambda: self._signal['I']
        )
        self.measurement_params['Q'] = Parameter(
            name='RF_quadrature',
            label='RF signal Quadrature',
            unit='V',
            get_cmd=lambda: self._signal['Q']
        )
        self.measurement_params['R'] = Parameter(
            name='RF_magnitude',
            label='RF signal magnitude',
            unit='V',
            get_cmd=lambda: self._signal['R']
        )
        self.measurement_params['theta'] = Parameter(
            name='RF_magnitude',
            label='RF signal phase',
            unit='deg',
            get_cmd=lambda: self._signal['theta']
        )
        self.measurement_params['amplitude_mean'] = Parameter(
            name='RF_amplitude_mean',
            label='RF amplitude mean',
            unit='V',
            get_cmd=lambda: self._signal['amplitude_mean']
        )
        self.measurement_params['signal_mean'] = Parameter(
            name='RF_signal_I_mean',
            label='RF signal Inphase mean',
            unit='V',
            get_cmd=lambda: self._signal['signal_mean']
        )
        self.measurement_params['signal_std'] = Parameter(
            name='RF_signal_std',
            label='RF signal std',
            unit='V',
            get_cmd=lambda: self._signal['signal_std']
        )

        self.phase_shift = ManualParameter(
            name='phase_shift',
            label='RF demodulator phase shift',
            unit='deg',
            initial_value=phase_shift
        )
        self.parameters['phase_shift'] = self.phase_shift

    def initialize(self):
        self.disable_channels()
        self.configure_output()
        self.configure_demodulator()

    def disable_channels(self):
        # Disable all input channels
        for input_channel in self.sigins:
            input_channel.on(False)
            for demodulator in self.demods:
                demodulator.enable(False)

        # Disable all output channels
        for output_channel in self.sigouts:
            # Turn channel output off
            output_channel.on(False)

            for oscillator_idx in range(8):
                # Turn off all oscillators
                oscillator_enable_parameter = output_channel.parameters[f'enables{oscillator_idx}']
                oscillator_enable_parameter(False)

    def configure_output(self, output_amplitude=10e-3, output_idx=None, oscillator_idx=None):
        if output_idx is None:
            output_idx = self.output_idx
        output_channel = self.uhfli.sigouts[output_idx]

        if oscillator_idx is None:
            oscillator_idx = self.oscillator_idx

        # Configure output channel
        output_channel.range(0.15)  # Can be 150 mV or 1.5 V
        output_channel.offset(0)  # No DC signal offset
        output_channel.imp50(False)  # DUT is not 50-ohm matched. Doubles amplitude if set to True

        # Set oscillator amplitude
        oscillator_amplitude = output_channel.parameters[f'amplitudes{oscillator_idx}']
        oscillator_amplitude(output_amplitude)  # Vpk, max set by channel range

        # Enable oscillator
        oscillator_enable = output_channel.parameters[f'enables{oscillator_idx}']
        oscillator_enable(True)

    def configure_demodulator(self, demodulator_idx=None, input_idx=None, oscillator_idx=None):
        if input_idx is None:
            input_idx = self.input_idx
        
        if oscillator_idx is None:
            oscillator_idx = self.oscillator_idx

        if demodulator_idx is None:
            demodulator_idx = self.demodulator_idx
        demodulator = self.uhfli.demods[demodulator_idx]

        demodulator.timeconstant(1e-3)  # sets bandwidth to ~ 1/time_constant/4pi
        demodulator.phaseshift(0)  # Apply phase shift
        demodulator.sinc(0)  # Apply sinc filter (globally applied)
        demodulator.trigger(0)  # Output on trigger. If true, will wait on trigger before updating value
        demodulator.harmonic(1)  # Use first harmonic
        demodulator.bypass(False)  # Bypass filters
        demodulator.order(3)  # Third order filter
        demodulator.rate(50e3)  # Data transfer rate (sets maximum sampling rate)
        demodulator.adcselect(input_idx)  # Use channel 1 (idx 0)
        demodulator.oscselect(oscillator_idx)  # Use oscillator 1 (idx 0)

        demodulator.enable(True)  # Turn on demodulator

        return demodulator

    # Acquisitions
    def get_signal(self):
        val = self.demodulator.sample()
        signal = (val['x'][0] + 1.j * val['y'][0]) * np.exp(-1.j*self.phase_shift()/360*2*np.pi)
        components = {
            'complex': signal,
            'I': np.real(signal), 
            'Q': np.imag(signal), 
            'R': np.abs(signal), 
            'theta': np.angle(signal, deg=True)
        }
        self._signal = components
        return components

    def _measure_components(self, components, measure=True):
        signal = self.get_signal() # Updates self._signal, needed for measurement

        msmt = running_measurement()
        if msmt is not None:
            for component in components:
                param = self.measurement_params[component]
                msmt.measure(param)
        
        return {signal[component] for component in components}

    def measure_I_Q(self):
        return self._measure_components(['I', 'Q'])

    def measure_R_theta(self):
        return self._measure_components(['R', 'theta'])

    def measure_I_Q_R_theta(self):
        return self._measure_components(['I', 'Q', 'R', 'theta'])

    # Measurements
    def measure_resonance(self, f_min=None, f_max=None, num=401, df=20e6):
        if f_min is None:
            f_min = self.frequency() - df
            f_max = self.frequency() + df
        self.frequency.sweep(f_min, f_max, num, measure_params=[self.measure_R_theta], revert=True)

    def measure_frequency_SNR_across_peak(self, voltage_sweep, df=2e6, num=31, disable_lockin_outputs=True):
        with MeasurementLoop('measure_RF_noise_across_Coulomb_peak') as msmt:
            self.phase_shift(0)
            with _disable_lockin_outputs(disable_lockin_outputs):
                V0 = voltage_sweep.parameter()
                for V in Sweep(self.frequency, around=df, num=num):
                    voltage_sweep.parameter(V0)
                    for V in voltage_sweep:
                        self.measure_I_Q()
        plot_data(msmt.dataset)

    # Trace functions
    @contextmanager
    def setup_traces(self, num, traces=1, time_constant=None, delay_scale=1, disable_lockin_outputs=True):
        with _disable_lockin_outputs(activate=disable_lockin_outputs):
            try:
                self._trace_context = True

                original_time_constant = self.time_constant()

                if time_constant is None:
                    time_constant = original_time_constant

                # Subscribe to signals
                self.daq_raw.unsubscribe('*')
                self._acquisition_signals = [
                    f'{self.demodulator.sample.zi_node}.{signal}'.lower() for signal in 'xy'
                ]
                for signal in self._acquisition_signals:
                    self.daq_raw.subscribe(signal)

                self.daq.triggernode(self.demodulator.sample.zi_node + '.TrigIn3')
                self.daq.type('continuous')
                self.daq.grid.cols(num)  # Points per trace
                self.daq.grid.rows(traces)  # Points per trace
                self.daq.grid.mode('linear')  # Not sure what this does
                self.time_constant(time_constant)
                self.daq.duration(time_constant * delay_scale * num)
                
                self.daq.endless(0)
                self.daq.clearhistory(1)

                yield
            finally:
                self._trace_context = False
                self.time_constant(original_time_constant)

    def acquire_trace(self, timeout=None, num=512, traces=1, time_constant=None, delay_scale=1, plot=False, disable_lockin_outputs=True):
        # Ensure we're set up for trace acquisitions
        if not self._trace_context:
            with self.setup_traces(
                num=num,
                traces=traces,
                time_constant=time_constant,
                delay_scale=delay_scale,
                disable_lockin_outputs=disable_lockin_outputs
            ):
                return self.acquire_trace(plot=plot)

        # Calculate timeout
        if timeout is None:
            timeout = max(3, 3 * self.daq.grid.cols.get_latest() * self.time_constant.get_latest())

        # Perform acquisition
        self.daq_raw.execute()
        sleep(0.1)
        t0 = perf_counter()
        while not self.daq_raw.finished():
            if perf_counter() - t0 > timeout:
                raise RuntimeError('Could not finish acquisition within timeout')
            sleep(0.1)
        # Record the 10 last acquisition durations for performance metrics
        self._acquisition_durations = self._acquisition_durations[-9:] + [perf_counter() - t0]

        raw_results = self.daq_raw.read(flat=True)
        self.daq_raw.finish()

        results = self.analyse_trace_results(raw_results, plot=plot)
        return results

    def measure_trace(
        self, 
        timeout=None, 
        num=512, 
        traces=1,
        time_constant=None, 
        delay_scale=1,
        plot=False, 
        disable_lockin_outputs=True,  
        signals=('I', 'Q', 'signal_std', 'signal_mean')
    ):
        if time_constant is None:
            time_constant = self.time_constant()

        sweep = Sweep(np.arange(num) * time_constant * delay_scale, 'time', unit='s')
        results = self.acquire_trace(
            timeout=timeout, 
            num=num, 
            traces=traces,
            time_constant=time_constant, 
            delay_scale=delay_scale,
            plot=plot,
            disable_lockin_outputs=disable_lockin_outputs
        )
        with MeasurementLoop('RF_trace') as msmt:
            for signal in signals:
                msmt.measure(
                    results[signal], 
                    name=self.measurement_params[signal].name, 
                    label=self.measurement_params[signal].label,
                    unit=self.measurement_params[signal].unit, 
                    setpoints=sweep
                )
        return results

    def analyse_trace_results(self, raw_results, plot=False):
        results = {}

        for key in self._acquisition_signals:
            result = raw_results[key][0]['value']
            if len(result) == 1:
                result = result[0]

            if True:
                key = key.split('.')[-1]

            results[key] = result

        arr_complex = results['x'] + 1.j*results['y']
        arr_complex *= np.exp(-1.j*self.phase_shift() / 360 * 2*np.pi)

        results = {
            'I': np.real(arr_complex), 
            'Q': np.imag(arr_complex), 
            'R': np.abs(arr_complex), 
            'theta': np.angle(arr_complex, deg=True)
        }
        results['signal_std'] = np.nanstd(results['I'] + 1.j * results['Q'])
        results['signal_mean'] = np.nanmean(results['I'])
        results['amplitude_mean'] = np.abs(np.nanmean(results['I'] + 1.j * results['Q']))

        if plot:
            results_plot = {key: val for key, val in results.items() if np.ndim(val)}
            fig, axes = plt.subplots(len(results_plot), 1, sharex=True, figsize=(8, 3*len(results_plot)))
            if not isinstance(axes, Iterable):
                axes = [axes]
            for k, (key, val) in enumerate(results_plot.items()):
                ax = axes[k]
                
                if np.ndim(val) == 1:
                    ax.plot(val)
                else:
                    ax.pcolormesh(val)

                ax.set_ylabel(key)
            plt.show()

        return results

    # Analysis
    @staticmethod
    def analyse_SNR_across_peak(data_idx):
        data = load_data(data_idx, print_summary=False)
        frequencies = data.zi_uhfli_dev2235_oscs0_freq.values
        voltages = data.qdac1_V_sensor_plunger

        plot_data(data, negative_clim=True)

        # Analyse results
        shifts = []
        results = []
        for I_row, Q_row in zip(data.RF_inphase, data.RF_quadrature):
            result = optimize_IQ((I_row, Q_row), silent=True)
            if result['signal'][0] > 0:
                result['signal'] *= -1
            results.append(result)
            shifts.append(np.percentile(result['signal'], 97) - np.percentile(result['signal'], 3))

        max_idx = np.argmax(shifts)
        max_frequency = frequencies[max_idx]
        max_shift = shifts[max_idx]
        phase_shift = results[max_idx]['phase_shift']
        print(
            f'Optimal frequency: {max_frequency/1e6:.2f} MHz, '
            f'SNR = {max_shift*1e3:.2f} mV, '
            f'Phase shift = {phase_shift} degrees'
        )

        signals = np.array([result['signal'] for result in results])
        signals_perpendicular = np.array([result['signal_perpendicular'] for result in results])

        fig, axes = plt.subplots(1, 2, figsize=(12,4))
        ax = axes[0]
        mesh=ax.pcolormesh(voltages, frequencies/1e6, signals * 1e3)
        ax.set_xlabel(voltages.name)
        ax.set_ylabel(f'RF frequency (MHz)')
        plt.colorbar(mesh)
        ax = axes[1]
        mesh = ax.pcolormesh(voltages, frequencies/1e6, signals_perpendicular * 1e3)
        plt.colorbar(mesh)
        ax.set_xlabel(voltages.name)
        ax.set_ylabel(f'RF frequency (MHz)')

        fig, ax = plt.subplots(figsize=(4,3))
        ax.plot(np.array(frequencies)/1e6, np.array(shifts)*1e6)
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Voltage shift (uV)')