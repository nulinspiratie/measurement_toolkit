import numpy as np
from functools import reduce
from warnings import warn
from dataclasses import dataclass
from typing import ClassVar
from matplotlib import pyplot as plt

import qcodes as qc

import broadbean as bb
from broadbean.plotting import plotter

ramp = bb.PulseAtoms.ramp  # args: start, stop
sine = bb.PulseAtoms.sine  # args: freq, ampl, off, phase


__all__ = ['Pulse', 'DCPulse', 'RampPulse', 'SinePulse', 'create_sequence', 'add_final_compensation_pulse']


class PulseSequence(qc.Parameter):
    pulses = None

    def snapshot_base(self, update=False):
        if self.pulses is None:
            return {}

        snapshot = {
            'pulses': [pulse.snapshot_base() for pulse in self.pulses]
        }
        return snapshot


class Pulse:
    # sampling_rate: ClassVar[float] = 1e9
    default_marker = [(0, 2e-6)]  # (t_start, duration)

    def create_blueprint(self, amplitude_scale):
        raise NotImplementedError

    def create_element(self, amplitude_scales: dict):
        element = bb.Element()
        for channel, amplitude_scale in amplitude_scales.items():
            blueprint = self.create_blueprint(amplitude_scale=amplitude_scale)
            element.addBluePrint(channel, blueprint)
        return element

    def snapshot_base(self):
        raise NotImplementedError

@dataclass
class DCPulse(Pulse):
    name: str
    amplitude: float
    duration: float

    def create_blueprint(self, amplitude_scale, channel_idx=None):
        assert amplitude_scale < 1

        blueprint = bb.BluePrint()
        # blueprint.setSR(Pulse.sampling_rate)

        amplitude = self.amplitude  
        if isinstance(self.amplitude, (tuple, list)):
            amplitude = amplitude[channel_idx]
        amplitude /= amplitude_scale

        blueprint.insertSegment(0, ramp, (amplitude, amplitude), dur=self.duration, name=self.name)

        return blueprint

    def snapshot_base(self):
        return {
            'name': self.name, 
            'amplitude': self.amplitude, 
            'duration': self.duration,
            'class': 'DCPulse',
        }   


@dataclass
class RampPulse(Pulse):
    name: str
    amplitude_start: float
    amplitude_stop: float
    duration: float

    def create_blueprint(self, amplitude_scale, channel_idx=None):
        assert amplitude_scale < 1

        blueprint = bb.BluePrint()
        # blueprint.setSR(Pulse.sampling_rate)
        
        amplitude_start = self.amplitude_start  
        if isinstance(self.amplitude_start, (tuple, list)):
            amplitude_start = amplitude_start[channel_idx]
        amplitude_start /= amplitude_scale
        
        amplitude_stop = self.amplitude_stop  
        if isinstance(self.amplitude_stop, (tuple, list)):
            amplitude_stop = amplitude_stop[channel_idx]
        amplitude_stop /= amplitude_scale

        blueprint.insertSegment(0, ramp, (amplitude_start, amplitude_stop), dur=self.duration, name=self.name)

        return blueprint

    def snapshot_base(self):
        return {
            'name': self.name, 
            'amplitude_start': self.amplitude_start, 
            'amplitude_stop': self.amplitude_stop, 
            'duration': self.duration,
            'class': 'RampPulse',
        }

@dataclass
class SinePulse(Pulse):
    name: str
    frequency: float
    amplitude: float
    duration: float
    offset: float = 0
    phase: float = 0

    def create_blueprint(self, amplitude_scale, channel_idx=None):
        assert amplitude_scale < 1

        blueprint = bb.BluePrint()
        # blueprint.setSR(Pulse.sampling_rate)

        amplitude = self.amplitude  
        if isinstance(self.amplitude, (tuple, list)):
            amplitude = amplitude[channel_idx]
        amplitude /= amplitude_scale
        
        blueprint.insertSegment(0, sine, (self.frequency, amplitude, self.offset, self.phase), dur=self.duration, name=self.name)

        return blueprint

    def snapshot_base(self):
        return {
            'name': self.name, 
            'amplitude': self.amplitude, 
            'frequency': self.frequency,
            'offset': self.offset,
            'phase': self.phase,
            'duration': self.duration,
            'class': 'SinePulse',
        }

def _apply_sweep_to_elem(element, sweeps, amplitude_scales):
    names, args, channels, iters = [], [], [], []
    for channel, amplitude_scale in amplitude_scales.items():
        for key, sequence in sweeps.items():
            name, arg = key.split('.')

            if arg == 'amplitude':
                amplitude_sequence = np.array(sequence) / amplitude_scale
                for pulse_keyword in ['start', 'stop']:
                    names.append(name)
                    args.append(pulse_keyword)
                    channels.append(channel)
                    iters.append(amplitude_sequence)
            else:
                names.append(name)
                args.append(arg)
                channels.append(channel)
                iters.append(amplitude_sequence)
        
    sequence = bb.makeVaryingSequence(
        baseelement=element,
        names=names,
        args=args,
        channels=channels,
        iters=iters,
    )
    return sequence
    
def create_sequence(
    pulses, 
    amplitude_scales, 
    marker=True, 
    upload=False, 
    start=False, 
    plot=True, 
    sweeps=None,
    frequency_cutoff=None,
    cutoff_order=1,
    silent=False
):
    AWG = getattr(qc.Station.default, 'AWG', None)
    if AWG is None:
        sampling_rate = 1e7
        if not silent:
            warn('AWG not found in station')
    else:
        sampling_rate = AWG.clock_freq()

    # Create element (we only need one for our purposes)
    element = bb.Element()

    # Create blueprint for each channel
    blueprints = []
    for channel_idx, (channel, amplitude_scale) in enumerate(amplitude_scales.items()):
        channel_blueprints = [
            pulse.create_blueprint(amplitude_scale=amplitude_scale, channel_idx=channel_idx) 
            for pulse in pulses
        ]
        blueprint = reduce(bb.BluePrint.__add__, channel_blueprints)

        if marker:
            blueprint.marker1 = Pulse.default_marker

        blueprints.append(blueprint)
        for blueprint in blueprints:
            blueprint.setSR(sampling_rate)
        element.addBluePrint(channel, blueprint)

    elements = [element]

    if sweeps is None:
        # Add elements to sequence
        sequence = bb.Sequence()
        for k, element in enumerate(elements, start=1):
            sequence.addElement(k, element)
    else:
        assert len(elements) == 1
        sequence = _apply_sweep_to_elem(element, sweeps, amplitude_scales)
        elements = list(sequence._data.values())
    
    # Set sequence sample rate
    sequence.setSR(sampling_rate)

    # Configure amplitudes and offsets
    for channel in amplitude_scales:
        if AWG is not None:
            amplitude = getattr(AWG, f'ch{channel}_amp')()
        else:
            amplitude = 4 
            if not silent:
                warn('Could not get channel amplitudes from AWG, please set station.AWG')
        sequence.setChannelAmplitude(channel, amplitude)  # Call signature: channel, amplitude (peak-to-peak)
        sequence.setChannelOffset(channel, 0)

        if frequency_cutoff is not None:
            f_cut = frequency_cutoff
            if isinstance(f_cut, dict):
                f_cut = f_cut[channel]
            sequence.setChannelFilterCompensation(channel, 'HP', order=cutoff_order, f_cut=f_cut)

    # Configure sequencing of elements
    for k, element in enumerate(elements, start=1):
        sequence.setSequencingTriggerWait(k, 0)
        sequence.setSequencingNumberOfRepetitions(k, 1)
        sequence.setSequencingEventJumpTarget(k, 0)
        next_idx = k + 1 if k < len(elements) else 1
        sequence.setSequencingGoto(k, next_idx)

    sequence.checkConsistency()  # returns True if all is well, raises errors if not

    # Package results for AWG
    package = sequence.outputForAWGFile()

    # Upload to AWG
    if upload:
        assert AWG is not None
        AWG.stop()
        AWG.all_channels_off()
        AWG.make_send_and_load_awg_file(*package[:], channels=package.channels)

    if start:
        assert upload
        AWG.all_channels_on()
        AWG.start()

    if plot:
        plotter(sequence)
        plt.show()

    if not hasattr(qc.Station.default, 'pulse_sequence'):
        pulse_sequence = PulseSequence('pulse_sequence')
        qc.Station.default.add_component(pulse_sequence)
    pulse_sequence = qc.Station.default.pulse_sequence
    pulse_sequence.pulses = pulses
    
    return {
        'elements': elements,
        'sequence': sequence,
        'package': package,
    }


def add_final_compensation_pulse(pulses, duration, channels=2):
    amplitude_duration = [0] * channels
    for pulse in pulses:
        if isinstance(pulse, DCPulse):
            amplitude = pulse.amplitude
            if isinstance(amplitude, (int, float)):
                amplitude = (amplitude,) * channels

            for k in range(channels):
                amplitude_duration[k] += amplitude[k] * pulse.duration
        elif isinstance(pulse, RampPulse):
            amplitude_start = pulse.amplitude_start
            if isinstance(amplitude_start, (int, float)):
                amplitude_start = (amplitude_start,) * channels
            amplitude_stop = pulse.amplitude_stop
            if isinstance(amplitude_stop, (int, float)):
                amplitude_stop = (amplitude_stop,) * channels

            for k in range(channels):
                amplitude_mean = (amplitude_start[k] + amplitude_stop[k]) / 2
                amplitude_duration[k] += amplitude_mean * pulse.duration
        elif isinstance(pulse, SinePulse):
            offset = pulse.offset
            if isinstance(offset, (int, float)):
                offset = (offset,) * channels
            for k in range(channels):
                amplitude_duration[k] += offset[k] * pulse.duration
        else:
            raise NotImplementedError('Cannot determine compensation pulse')

    compensation_amplitude = [-A_t / duration for A_t in amplitude_duration]
    return DCPulse('compensation', duration=duration, amplitude=compensation_amplitude)