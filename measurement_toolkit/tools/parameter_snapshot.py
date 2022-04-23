import warnings
from typing import Union
from qcodes.instrument.parameter import _BaseParameter


class ParameterSnapshot:
    """Collection of parameters

    Allowed optional parameter dict keys:

    - formatter: how to format value (e.g. ".2f")
    - comment: comment to place after value, e.g. "param(val)  # {comment}
    - name (optional): Name of parameter. Overrides parameter.name
    - unit (optional): Unit of parameter. Overrides parameter.unit
    """
    def __init__(self, parameters: dict = {}):
        self.parameters = {}
        for parameter_name, parameter in parameters.values():
            if isinstance(parameter, dict):
                self.add_parameter(**parameter)
            elif isinstance(parameter, _BaseParameter):
                self.add_parameter(parameter)

    def update(self):
        for parameter_info in self.parameters.values():
            parameter = parameter_info['parameter']

            if parameter_info.get('snapshot_get', True):
                parameter_info['value'] = parameter()
            else:
                parameter_info['value'] = parameter.get_latest()

    def __call__(self, *args, **kwargs):
        default_kwargs = dict(verbose=True, pretty=False, comment=True)
        kwargs = {**default_kwargs, **kwargs}

        if args:
            return self.call_with_args(*args, **kwargs)

        self.update()

        if kwargs['verbose']:
            self.print(pretty=kwargs['pretty'], comment=kwargs['comment'])
        else:
            return {name: elem['value'] for name, elem in self.parameters.items()}

    def call_with_args(self, *args, **kwargs):
        pass

    def print(self, pretty=False, comment=True):
        for name, parameter_info in self.parameters.items():
            unit = parameter_info.get('unit', parameter_info['parameter'].unit)
            formatter = parameter_info.get('formatter', '')

            if parameter_info['value_lower_bound'] is not None:
                if abs(parameter_info['value']) < parameter_info['value_lower_bound']:
                    continue

            # Must be a better way to do this
            value_str = ('{:'+formatter+'}').format(parameter_info['value'])

            if pretty:
                parameter_string = f'{name} = {value_str}'
                if unit:
                    parameter_string += f' {parameter_info["unit"]}'
            else:
                parameter_string = f'{name}({value_str})'
                if comment and parameter_info.get('comment'):
                    parameter_string += f'  # {parameter_info["comment"]}'

            print(parameter_string)

    def add_parameter(self, parameter, name=None, comment='', overwrite=True, value_lower_bound=None, **kwargs):
        if name is None:
            name = parameter.name

        if name in self.parameters and not overwrite:
            raise SyntaxError(f'Parameter {name} already registered in snapshot')

        self.parameters[name] = {
            'parameter': parameter,
            'comment': comment,
            'value_lower_bound': value_lower_bound,
            **kwargs
        }
