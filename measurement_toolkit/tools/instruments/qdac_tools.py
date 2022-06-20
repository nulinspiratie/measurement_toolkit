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
    'combine_gates',
    'check_leakages'
]

def ramp_voltages_zero():
    # QDac should be accessible from Station
    for instrument_name, instrument in qc.Station.default.components.items():
        if not instrument_name.startswith('qdac'):
            continue

        for ch in instrument.channels:
            ch.v(0)


def ramp_voltages(target_voltages, other_gates_zero=True, silent=True):
    # QDac should be accessible from Station
    station = qc.Station.default
    qdac = station.qdac
    yoko = station.components['yoko']
    yoko.voltage(0)

    assert all(isinstance(channel_id, int) for channel_id in target_voltages)

    # Convert DC lines to DAC channels
    qdac_target_voltages = {}
    for key, target_voltage in target_voltages.items():
        gate = next(g for g in gates.values() if g.DC_line == key)
        assert gate.DAC_channel is not None
        qdac_target_voltages[gate.DAC_channel] = target_voltage

    for ch in qdac.channels:
        if ch.id in qdac_target_voltages:
            target_voltage = qdac_target_voltages[ch.id]
        elif other_gates_zero:
            target_voltage = 0
        else:
            continue

        voltage_difference = target_voltage - ch.v()
        if not silent and abs(voltage_difference) > 1e-4:
            print(f'Ramping ch{ch.id} from {round(ch.v(), 4):2.3g} V to {round(target_voltage, 4):.3g} V...', end=' ')
            ch.v(target_voltage)
            print('Done')
        else:
            ch.v(target_voltage)


def qdac_gate_voltages(show_zero=False):
    # QDac should be accessible from Station
    station = qc.Station.default
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


def combine_gates(
        *gates,
        max_difference=2e-3,
        precision=3,
        scales=None,
        offsets=None
):
    station = qc.Station.default

    assert len(set(gates)) == len(gates), f"{gates=} have duplicates"
    
    combined_gates = []
    for gate in gates:
        if isinstance(gate, int):
            combine_gates.append(station.gates[gate])
        else:
            combined_gates.append(gate)

    name = 'gate_' + '_'.join(f'DC{gate.DC_line}' for gate in gates)
    # label = ' & '.join(f'{gate.name}:DC{gate.DC_line}' for gate in combined_gates)

    parameter = CombinedParameter(
        name=name,
        # label=label,
        parameters=[gate for gate in combined_gates],
        unit='V',
        max_difference=max_difference,
        precision=precision,
        scales=scales,
        offsets=offsets
    )
    parameter.sweep_to = partial(sweep_gate_to, parameter)
    return parameter


def check_leakages(current_limit=3e-9):
    global gates
    leaking_gates = {}
    for gate in gates.values():
        if hasattr(gate, 'i'):
            current = gate.i()

            if current > current_limit:
                leaking_gates[gate] = current

    if leaking_gates:
        for gate, current in leaking_gates.items():
            print(f'{current * 1e9:.0f} nA leakage from gate {gate}')
    else:
        print('No gate leakage')


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
            label=f'Channel {id} current',
            unit='A',
            get_cmd=partial(partial(lambda ch: ch.read_current_A()[0]), channel)
        )

        channel.parameters.pop('v', None)
        channel.add_parameter(
            name='v',
            label=f'Channel {id} voltage',
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



### In progress
def load_qdac_snapshot_voltages(dataset, qdac_snapshot):
    voltages = {}

    for key, value in qdac_snapshot['parameters'].items():
        if not key.startswith('V'):
            continue
        elif value['value'] is None or abs(value['value']) < 1e-4:
            continue
        elif f'qdac_{key}' in dataset.paramspecs:
            # Gate is being swept
            continue

        voltages[key] = value['value']
    return voltages

def load_qdac2_snapshot_voltages(dataset, qdac_snapshot):
    return {}

def gate_voltages(dataset=None, silent=False, pretty=False):
    voltages = {}

    if dataset is not None:
        if isinstance(dataset, numbers.Integral):
            dataset = load_by_run_spec(captured_run_id=dataset)
        for instrument_name, instrument_snapshot in dataset.snapshot['station']['instruments'].items():
            if instrument_snapshot['__class__'] == 'qcodes.instrument_drivers.QDevil.QDevil_QDAC.QDac':
                voltages.update(load_qdac_snapshot_voltages(dataset, instrument_snapshot))
            elif instrument_snapshot['__class__'] == 'qcodes_contrib_drivers.drivers.QDevil.QDAC2.QDac2':
                # voltages.update(load_qdac2_snapshot_voltages(dataset, instrument_snapshot))
                voltages.update(load_qdac_snapshot_voltages(dataset, instrument_snapshot))
                # return instrument_snapshot
        V_bias = np.NaN
    else:
        station = qc.Station.default
        V_threshold = 1e-4

        for DC_line, gate in station.gates.items():
            if not hasattr(gate, 'v'):
                continue

            voltage = gate()
            if abs(voltage) > V_threshold:
                voltages[gate] = voltage

        V_bias = station.V_bias()

    if silent:
        return voltages
    else:
        if voltages:
            station = qc.Station.default
            # Determine maximum length
            max_length = max(len(f'{V:.3g}') for V in voltages.values())
            for gate, voltage in voltages.items():
                if isinstance(gate, str):
                    gate = next(g for g in station.gates.values() if g.name == gate)

                if not pretty:
                    unformatted_str = f"{{:}}({{:{max_length}.3g}})  # DC{{:}}"
                else:
                    unformatted_str = f"{{:}} = {{:.3g}} (DC{{:}})"
                formatted_str = unformatted_str.format(gate.name, voltage, gate.DC_line)
                print(formatted_str)

        if not np.isnan(V_bias):
            print(f'V_bias({V_bias:.4g})')




def get_dataset_voltages(dataset, print_nonzero=True):
    qdac_snapshot = dataset.snapshot['station']['instruments']['qdac']['parameters']
    gates = {key: val['value'] for key, val in qdac_snapshot.items() if key.startswith('V_')}
    if print_nonzero:
        for key, val in gates.items():
            if abs(val) > 0.001:
                print(f"{key}: {val:.3f} V")
    return gates

