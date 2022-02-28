import pandas as pd
from measurement_toolkit.parameters.DC_line_parameter import DCLine


def initialize_DC_lines(gates_excel_file, attach_to_qdac=False, namespace=None):
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
        if namespace:
            setattr(namespace, line.name, line)
            
        # Add line groups to namespace
        for key, line_group in DC_line_groups.items():
            setattr(namespace, key, line_group)
    
    # Add gate lists to station
    station = qc.Station.default or qc.Station()
    station.lines = lines
    station.gates = gates
    station.ohmics = ohmics

    # Perform these operations if the station contains relevant instruments
    if attach_to_qdac:
        for param in station._monitor_parameters.copy():    
            if isinstance(param, DCLine):
                station._monitor_parameters.remove(param)

        for instrument_name, instrument in station.components.items():
            if not instrument_name.startswith('qdac'):
                continue

            # Add all lines to QDac
            for name, line in lines.items():
                instrument.parameters[name] = line
                line._instrument = instrument

                if hasattr(line, 'v'):
                    station._monitor_parameters.append(line)

    return DC_line_groups