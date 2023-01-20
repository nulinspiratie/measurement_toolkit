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


__all__ = ['Pulse', 'DCPulse', 'SinePulse', 'create_sequence']

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

@dataclass
class DCPulse(Pulse):
    name: str
    amplitude: float
    duration: float
    marker: bool = False

    def create_blueprint(self, amplitude_scale):
        assert amplitude_scale < 1

        blueprint = bb.BluePrint()
        # blueprint.setSR(Pulse.sampling_rate)
        amplitude = self.amplitude / amplitude_scale
        blueprint.insertSegment(0, ramp, (amplitude, amplitude), dur=self.duration, name=self.name)

        if self.marker:
            blueprint.marker1 = self.default_marker

        return blueprint

@dataclass
class SinePulse(Pulse):
    name: str
    frequency: float
    amplitude: float
    duration: float
    offset: float = 0
    phase: float = 0
    marker: bool = False

    def create_blueprint(self, amplitude_scale):
        assert amplitude_scale < 1

        blueprint = bb.BluePrint()
        # blueprint.setSR(Pulse.sampling_rate)
        amplitude = self.amplitude / amplitude_scale
        
        blueprint.insertSegment(0, sine, (self.frequency, amplitude, self.offset, self.phase), dur=self.duration, name=self.name)

        if self.marker:
            blueprint.marker1 = self.default_marker

        return blueprint


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
    for channel, amplitude_scale in amplitude_scales.items():
        channel_blueprints = [pulse.create_blueprint(amplitude_scale=amplitude_scale) for pulse in pulses]
        blueprint = reduce(bb.BluePrint.__add__, channel_blueprints)
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
            sequence.setChannelFilterCompensation(channel, 'HP', order=1, f_cut=frequency_cutoff)

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
    
    return {
        'elements': elements,
        'sequence': sequence,
        'package': package,
    }
