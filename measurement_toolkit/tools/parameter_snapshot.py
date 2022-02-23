import warnings
from qcodes.instrument.parameter import _BaseParameter


class ParameterSnapshot:
    """Collection of parameters

    Allowed optional parameter dict keys:

    - formatter: how to format value (e.g. ".2f")
    - comment: comment to place after value, e.g. "param(val)  # {comment}
    - name (optional): Name of parameter. Overrides parameter.name
    - unit (optional): Unit of parameter. Overrides parameter.unit
    """
    def __init__(self, parameters=()):
        self.parameters = []
        for parameter in parameters:
            if isinstance(parameter, dict):
                self.add_parameter(**parameter)
            elif isinstance(parameter, _BaseParameter):
                self.add_parameter(parameter)

    def update(self):
        for parameter_info in self.parameters:
            parameter = parameter_info['parameter']

            if parameter_info.get('snapshot_get', True):
                parameter_info['value'] = parameter()
            else:
                parameter_info['value'] = parameter.get_latest()

    def __call__(self, verbose=True, pretty=False, comment=True, **kwargs):
        self.update()

        if verbose:
            self.print(pretty=pretty, comment=comment)
        else:
            return {elem['name']: elem['value'] for elem in self.parameters}

    def print(self, pretty=False, comment=True):
        for parameter_info in self.parameters:
            name = parameter_info.get('name', parameter_info['parameter'].name)
            unit = parameter_info.get('unit', parameter_info['parameter'].unit)
            formatter = parameter_info.get('formatter', '')

            # Must be a better way to do this
            value_str = ('{:'+formatter+'}').format(parameter_info['value'])

            if pretty:
                parameter_string = f'{name} = {value_str}'
                if unit:
                    parameter_string += f' {parameter_info["unit"]}'
            else:
                parameter_string = f'{name}({value_str})'
                if comment and 'comment' in parameter_info:
                    parameter_string += f'  # {parameter_info["comment"]}'

            print(parameter_string)

    def add_parameter(self, parameter, name=None, comment=''):
        if parameter in self.parameters:
            warnings.warn(f'Parameter {parameter} already added to gate_voltages')
        else:
            self.parameters.append({
                'parameter': parameter,
                'name': name,
                'comment': comment
            })
