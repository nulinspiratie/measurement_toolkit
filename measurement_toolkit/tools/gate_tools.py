from functools import partial
from time import sleep
import pandas as pd

import qcodes as qc
from qcodes.utils import validators as vals

from measurement_toolkit.parameters import DCLine, CombinedParameter


__all__ = [
    'iterate_gates',
    'initialize_DC_lines',
    'combine_gates',
    'check_gate_leakages'
]

def iterate_gates(gates, sort=True, silent=False, active_switch=None, inactive_switch='float'):
    assert all(gate.DC_line is not None for gate in gates.values())

    if sort:
        sorted_gates = sorted(
            gates, key=lambda gate: gate.DC_line
        )  # TODO verify this line is correct
    else:
        sorted_gates = gates

    previous_gate = None
    breakout_box = None
    for gate in sorted_gates:
        if not silent:
            # Check if we should emit notification to switch breakout box
            if gate.breakout_box != breakout_box:
                message = f"Switch to breakout box {gate.breakout_box}."
                print(message, flush=True)
                sleep(0.01)
                if input("{message}\t Press 'q' to cancel") == 'q': raise KeyboardInterrupt
                breakout_box = gate.breakout_box

            message = f"Connect breakout box {gate.breakout_box} idx {gate.breakout_idx}"
            if active_switch is not None:
                message += f' to {active_switch}.'
            for breakout_idx in gate.breakout_idxs[1:]:
                message += f"\n  Float breakout box {gate.breakout_box} idx {gate.breakout_idx}."
            if previous_gate is not None:
                if inactive_switch == 'float':
                    message += f"\nFloat breakout box {previous_gate.breakout_box} idx {previous_gate.breakout_idx}."
                else:
                    message += f"\nConnect breakout box {previous_gate.breakout_box} idx {previous_gate.breakout_idx} to {inactive_switch}."
            print(message, flush=True)
            sleep(0.01)
            if input("{message}\t Press 'q' to cancel") == 'q': raise KeyboardInterrupt

            yield gate
    print("Finished iterating over gates")


def initialize_DC_lines(
    gates_excel_file, 
    sample_holder='QDevil',
    qdac=None, 
    populate_namespace=True,
    update_monitor=True,
    parameter_container=None,
    silent=False
):
    assert sample_holder in ['QDevil', 'Sydney']
    gates_table = pd.read_excel(gates_excel_file, skiprows=[0])
    
    lines = {}
    for idx, row in gates_table.iterrows():
        if row['skip']:
            continue
        line = DCLine(**row, sample_holder=sample_holder)
        lines[row['name']] = line

    DC_gates = {name: line for name, line in lines.items() if line.line_type == 'gate'}
    DC_line_groups = {
        'lines': lines,
        'DC_gates': DC_gates,
        'gates': [gate for gate in DC_gates.values() if gate.DC_line is not None],
        'ohmics': [line for line in lines.values() if line.line_type == 'ohmic']
    }
    
    # Add gate lists to station
    station = qc.Station.default or qc.Station()
    station.lines = DC_line_groups['lines']
    station.gates = DC_line_groups['gates']
    station.ohmics = DC_line_groups['ohmics']
    for name, line in lines.items():
        setattr(station, name, line)
    
    # Populate namespace
    if populate_namespace:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is not None:
            # Remove any pre-existing DC lines from namespace
            if hasattr(station, 'lines'):
                for line in station.lines:
                    shell.user_ns.pop(line, None)

            # Add DC lines to namespace
            for line in lines.values():
                shell.user_ns[line.name] = line
                
            # Add line groups to namespace
            for key, line_group in DC_line_groups.items():
                shell.user_ns[key] = line_group

    # Perform these operations if qdac is passed
    if qdac is not None:
        for name, line in lines.items():
            if not line.DAC_channel:
                continue
            
            line.attach_QDac(qdac, line._V_min, line._V_max, line.voltage_scale)
            qdac.parameters[name] = line

    # Add to monitor
    if update_monitor:
        for param in station._monitor_parameters.copy():    
            if isinstance(param, DCLine):
                station._monitor_parameters.remove(param)
                
        for name, line in lines.items():
            if hasattr(line, 'v') and line.v is not None:
                station._monitor_parameters.append(line)
                
        for gate in DC_line_groups['gates']:
            qc.config.monitor.parameters_metadata[gate.name] = {'formatter': '{:.4g}'}

    # Update function showing gate voltages
    if parameter_container is not None:
        for gate in DC_line_groups['lines'].values():
            if getattr(gate, 'DAC', None) is not None:
                parameter_container.add_parameter(
                    gate, 
                    formatter='.5g', 
                    comment=f'DC{gate.DC_line}', 
                    value_lower_bound=1e-5
                )

    # Print number of gates and ohmics
    if not silent:
        print(f'Number of gates: {len(DC_line_groups["gates"])}, ohmics: {len(DC_line_groups["ohmics"])}')

    return DC_line_groups


def combine_gates(
        *gates,
        max_difference=2e-3,
        precision=3,
        scales=None,
        offsets=None,
        name=None,
):
    station = qc.Station.default

    assert len(set(gates)) == len(gates), f"{gates=} have duplicates"
    
    combined_gates = []
    for gate in gates:
        if isinstance(gate, int):
            combine_gates.append(station.gates[gate])
        else:
            combined_gates.append(gate)

    if name is None:
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
    parameter.sweep_defaults = gates[0].sweep_defaults
    return parameter


def check_gate_leakages(current_threshold=3e-9):
    gates = qc.Station.default.gates
    leaking_gates = {}
    for gate in gates.values():
        if hasattr(gate, 'i'):
            current = gate.i()

            if current > current_threshold:
                leaking_gates[gate] = current

    if leaking_gates:
        for gate, current in leaking_gates.items():
            print(f'{current * 1e9:.0f} nA leakage from gate {gate}')
    else:
        print('No gate leakage')


def print_gate_leakages(indent_level=0, current_limit=5e-9):
    station = qc.Station.default
    indent = ' ' * (4 *indent_level)
    is_leaking = False
    for gate in station.gates:
        current = gate.i()
        print(f'{indent}{gate.short_name:30} = {current*1e9:4.1f} nA', end='')
        if abs(current) > current_limit:
            print(' LEAKING!')
            is_leaking = True
        else:
            print()

    text = 'GATES ARE LEAKING!!!' if is_leaking else 'NO GATE LEAKAGE'
    print(indent + '*' * (len(text)+8))
    print(indent + '*** ' + text + ' ***')
    print(indent + '*' * (len(text)+8))