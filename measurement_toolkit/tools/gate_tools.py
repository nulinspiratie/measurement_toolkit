from functools import partial
from time import sleep
import pandas as pd

import qcodes as qc
from qcodes.utils import validators as vals

from measurement_toolkit.parameters import DCLine, CombinedParameter
from measurement_toolkit.measurements.DC_measurements import bias_scan


__all__ = [
    'iterate_gates',
    'initialize_DC_lines',
    'configure_DC_bias_line',
    'combine_gates',
    'check_gate_leakages'
]

def iterate_gates(gates, sort=True, silent=False):
    assert all(gate.DC_line is not None for gate in gates.values())

    if sort:
        sorted_gates = sorted(
            gates.values(), key=lambda gate: gate.DC_line
        )  # TODO verify this line is correct
    else:
        sorted_gates = gates

    breakout_box = None
    for gate in sorted_gates:
        if not silent:
            # Check if we should emit notification to switch breakout box
            if gate.breakout_box != breakout_box:
                print(f"Switch to breakout box {gate.breakout_box}", flush=True)
                input()
                breakout_box = gate.breakout_box

            print(f"Connect breakout box {gate.breakout_box} idx {gate.breakout_idx}")
            for breakout_idx in gate.breakout_idxs[1:]:
                print(f"Float breakout box {gate.breakout_box} idx {gate.breakout_idx}", flush=True)
            input()

            yield gate
    print("Finished iterating over gates")


def initialize_DC_lines(
    gates_excel_file, 
    qdac=None, 
    populate_namespace=True,
    update_monitor=True,
    parameter_container=None
):
    gates_table = pd.read_excel(gates_excel_file, skiprows=[0])
    
    lines = {}
    for idx, row in gates_table.iterrows():
        if row['skip']:
            continue
        line = DCLine(**row)
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
    
    # Populate namespace
    if populate_namespace:
        from IPython import get_ipython
        shell = get_ipython
        if shell is not None:
            # Remove any pre-existing DC lines from namespace
            if hasattr(station, lines):
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
            if hasattr(line, 'v'):
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

    return DC_line_groups


def configure_DC_bias_line(voltage_parameter, scale=1e3, update_monitor=True): 
    station = qc.Station.default or qc.Station()

    V_bias = qc.DelegateParameter(
        'V_bias',
        source=voltage_parameter,
        scale=scale,
        vals=vals.Numbers(-1.1e-3, 1.1e-3),
        unit='V'
    )
    # Add method to update ohmics
    def update_ohmics(V_bias, *ohmics):
        V_bias.ohmics = ohmics
        ohmics_str = '&'.join([f'{station.ohmics[ohmic].name}:DC{ohmic}' for ohmic in ohmics])
        V_bias.label = f"{ohmics_str} bias voltage"
    V_bias.update_ohmics = partial(update_ohmics, V_bias)

    # Add to Station
    try:
        station.remove_component('V_bias')
    except KeyError:
        pass
    station.add_component(V_bias, 'V_bias')

    V_bias.sweep = partial(bias_scan, V_bias=V_bias)

    # Remove any pre-existing parameter
    if update_monitor:
        for param in station._monitor_parameters.copy():
            if param.name == 'V_bias':
                station._monitor_parameters.remove(param)
        station._monitor_parameters.insert(0, V_bias)

    return V_bias


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