import numpy as np
from functools import partial

from qcodes.instrument import Instrument


class UHFLI_Interface(Instrument):
    def __init__(self, RF_lockin, name='RF_lockin', oscillator_idx=1, output_idx=1):
        super().__init__(name=name)

        self.RF_lockin = RF_lockin

        self.oscillator_idx = oscillator_idx
        self.oscillator = self.RF_lockin.oscs[oscillator_idx]

        self.output_idx = output_idx

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

    


def configure_uhfli_output(RF_lockin, output_idx, oscillator_idx, output_amplitude):
    # Configure output channel
    output_channel = RF_lockin.sigouts[output_idx]
    output_channel.range(0.15)  # Can be 150 mV or 1.5 V
    output_channel.offset(0)  # No DC signal offset
    output_channel.imp50(False)  # DUT is not 50-ohm matched. Doubles amplitude if set to True

    # Set oscillator amplitude
    oscillator_amplitude = output_channel.parameters[f'amplitudes{oscillator_idx}']
    oscillator_amplitude(output_amplitude)  # Vpk, max set by channel range

    # Enable oscillator
    oscillator_enable = output_channel.parameters[f'enables{oscillator_idx}']
    oscillator_enable(True)

    return output_channel


def configure_uhfli_demodulator(RF_lockin, demodulator_idx, input_idx, oscillator_idx):
    demodulator = RF_lockin.demods[demodulator_idx]
    demodulator_name = 'demod'+demodulator.short_name[-1]

    demodulator.timeconstant(1e-3)  # sets bandwidth to ~ 1/time_constant/4pi
    demodulator.phaseshift(0)  # Apply phase shift
    demodulator.sinc(0)  # Apply sinc filter (globally applied)
    demodulator.trigger(0)  # Output on trigger. If true, will wait on trigger before updating value
    demodulator.harmonic(1)  # Use first harmonic
    demodulator.bypass(False)  # Bypass filters
    demodulator.order(3)  # Third order filter
    demodulator.rate(1716.6)  # Data transfer rate (not sure what it does)
    demodulator.adcselect(input_idx)  # Use channel 1 (idx 0)
    demodulator.oscselect(oscillator_idx)  # Use oscillator 1 (idx 0)

    demodulator.enable(True)  # Turn on demodulator

    return demodulator


def add_uhfli_acquisition_params(RF_lockin, demodulator_idx):
    # Add common commands to acquire a single sample
    def get_sample(component):
        sample = RF_lockin.demods[demodulator_idx].sample()
        if component == 'sample':
            return sample
        if component == 'X':
            return np.real(sample)
        elif component == 'Y':
            return np.imag(sample)
        elif component == 'R':
            return np.abs(sample)
        elif component == 'theta':
            return np.angle(sample, deg=True)
        else:
            raise SyntaxError('kwarg "component" must be sample, X, Y, R, or theta')
    for component in ['sample', 'X', 'Y', 'R', 'theta']:
        RF_lockin.parameters.pop(component, None)
        RF_lockin.add_parameter(component, unit='V', get_cmd=partial(get_sample, component=component))


def configure_uhfli(
    RF_lockin, 
    output_amplitude=1e-3,
    oscillator_idx=0, 
    input_idx=0,
    output_idx=0,
    demodulator_idx=0,
):
    add_uhfli_acquisition_params(RF_lockin, demodulator_idx=demodulator_idx)

    disable_uhfli_channels(RF_lockin)
    output_channel = configure_uhfli_output(RF_lockin, output_idx=output_idx, oscillator_idx=oscillator_idx, output_amplitude=output_amplitude)
    demodulator = configure_uhfli_demodulator(RF_lockin, demodulator_idx=demodulator_idx, input_idx=input_idx, oscillator_idx=oscillator_idx)
    output_channel.on(True)