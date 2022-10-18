"""Functions and tools related to the QDevil QDac"""
import numbers
from functools import partial

import numpy as np
import qcodes as qc
from measurement_toolkit.parameters.general_parameters import CombinedParameter
from measurement_toolkit.parameters.DC_line_parameter import sweep_gate_to
from qcodes.dataset.data_set import load_by_run_spec
from qcodes.utils import validators as vals

__all__ = [
    'ramp_voltages_zero',
    'ramp_voltages',
    'configure_qdac',
    'configure_qdac2'
]

def ramp_voltages_zero():
    """Ramps voltages to zero of all qdacs"""
    # QDac should be accessible from Station
    for instrument_name, instrument in qc.Station.default.components.items():
        if not instrument_name.startswith('qdac'):
            continue

        for ch in instrument.channels:
            ch.v(0)


def qdac_gate_voltages(qdac=None, show_zero=False):
    # QDac should be accessible from Station
    station = qc.Station.default

    if qdac is None:
        qdac = station.qdac

    qdac_nonzero_voltages = {ch.id: round(ch.v(), 5) for ch in qdac.channels if abs(ch.v()) > 1e-4}

    voltages = {}
    for gate in station.gates:
        if gate.DAC_channel is not None:
            voltage = round(gate.v(), 5)
            if voltage > 1e-4:
                qdac_nonzero_voltages.pop(gate.DAC_channel)
                voltages[gate.DC_line].append(voltage)
            elif show_zero:
                voltages[gate.DC_line].append(voltage)

    if qdac_nonzero_voltages:
        print("QDac has nonzero voltages that don't belong to a gate:", qdac_nonzero_voltages)
    return voltages


def configure_qdac(qdac, set_vhigh_ilow=False, inter_delay=30e-3, step=10e-3):
    from qcodes.instrument_drivers.QDevil.QDevil_QDAC import Mode as QDac_mode

    # Set channel ids
    for ch_id, channel in enumerate(qdac.channels, start=1):
        channel.id = ch_id

        # Update voltage
        channel.v()

        # Set channel ranges
        # All channels set to high voltage (+-10V) with 2 mV precision (1V has 400 uV precision)
        # All channels set to low current (1 uA), with resolution ~0.2 nA (100 uA has 5 nA resolution)
        mode = channel.mode()
        if not mode == QDac_mode.vhigh_ilow:
            print(
                f'QDac channel {ch_id:02} not set to {mode.name} mode instead of high voltage / low current.'
                f'When at 0V, run: qdac.ch{ch_id:02}.mode(QDac_mode.vhigh_ilow)'\
            )
        # Set all gate voltages to high voltage / low current
        if set_vhigh_ilow:
            channel.mode(QDac_mode.vhigh_ilow)

        # Set ramping
        channel.v.inter_delay = inter_delay
        channel.v.step = step


def configure_qdac2(qdac, inter_delay=30e-3, step=10e-3):
    for ch_id, channel in enumerate(qdac.channels, start=1):
        channel.id = ch_id

        channel.parameters.pop('i', None)
        channel.add_parameter(
            name='i',
            label=f'Channel {channel.id} current',
            unit='A',
            get_cmd=partial(partial(lambda ch: ch.read_current_A()[0]), channel)
        )

        channel.parameters.pop('v', None)
        channel.add_parameter(
            name='v',
            label=f'Channel {channel.id} voltage',
            unit='V',
            set_cmd=channel.dc_constant_V,
            get_cmd=channel.dc_constant_V,
            vals=vals.Numbers(-9.99, 9.99)
        )

        # Set ramping
        channel.v.inter_delay = inter_delay
        channel.v.step = step
        
        # Raise warning if not in low-current measurement range
        if channel.measurement_range() != 'LOW':
            print(
                f'QDac channel {ch_id:02} not set to low current measurement mode.'
                f'When at 0V, run: qdac.ch{ch_id:02}.measurement_range("low")'\
            )
