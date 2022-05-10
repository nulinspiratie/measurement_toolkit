from functools import partial
import pandas as pd

import qcodes as qc
from qcodes.utils import validators as vals

from measurement_toolkit.parameters.DC_line_parameter import DCLine
from measurement_toolkit.measurements.DC_measurements import bias_scan


def initialize_DC_lines(
    gates_excel_file, 
    attach_to_qdac=False, 
    namespace=None,
    update_monitor=True,
    parameter_container=None
):
    gates_table = pd.read_excel(gates_excel_file, skiprows=[0])
    
    lines = {}
    for idx, row in gates_table.iterrows():
        line = DCLine(**row)
        lines[row['name']] = line

    DC_gates = {name: line for name, line in lines.items() if line.line_type == 'gate'}
    DC_line_groups = {
        'lines': lines,
        'DC_gates': DC_gates,
        'gates': {gate.DC_line: gate for gate in DC_gates.values() if gate.DC_line is not None},
        'ohmics': {line.DC_line: line for name, line in lines.items() if line.line_type == 'ohmic'}
    }
    
    # Populate namespace
    if namespace is not None:
        # Add DC lines to namespace
        for line in lines.values():
            setattr(namespace, line.name, line)
            
        # Add line groups to namespace
        for key, line_group in DC_line_groups.items():
            setattr(namespace, key, line_group)
    
    # Add gate lists to station
    station = qc.Station.default or qc.Station()
    station.lines = DC_line_groups['lines']
    station.gates = DC_line_groups['gates']
    station.ohmics = DC_line_groups['ohmics']

    # Perform these operations if the station contains relevant DAC instruments
    if attach_to_qdac:
        for instrument_name, instrument in station.components.items():
            if not instrument_name.startswith('qdac'):
                continue

            # Add all lines to QDac
            for name, line in lines.items():
                instrument.parameters[name] = line
                line._instrument = instrument

    # Add to monitor
    if update_monitor:
        for param in station._monitor_parameters.copy():    
            if isinstance(param, DCLine):
                station._monitor_parameters.remove(param)
                
        for name, line in lines.items():
            if hasattr(line, 'v'):
                station._monitor_parameters.append(line)
                
        for gate in DC_line_groups['gates'].values():
            qc.config.monitor.parameters_metadata[gate.name] = {'formatter': '{:.4g}'}

    # Update function showing gate voltages
    if parameter_container is not None:
        for gate in DC_line_groups['lines'].values():
            if hasattr(gate, 'DAC'):
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