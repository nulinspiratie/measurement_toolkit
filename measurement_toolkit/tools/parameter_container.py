from typing import Optional, Sequence, Dict, Any

from qcodes.instrument.parameter import _BaseParameter
from qcodes.utils.metadata import Metadatable


class ParameterContainer(Metadatable):
    """Collection of parameters

    Allowed optional parameter dict keys:

    - formatter: how to format value (e.g. ".2f")
    - comment: comment to place after value, e.g. "param(val)  # {comment}
    - name (optional): Name of parameter. Overrides parameter.name
    - unit (optional): Unit of parameter. Overrides parameter.unit
    """
    def __init__(
        self, 
        name='', 
        parameters: dict = {},
        parameter_containers: dict = {},
    ):
        self.name = name

        self.parameters = {}
        self.nested_containers = parameter_containers

        for name, parameter in parameters.values():
            if isinstance(parameter, dict):
                self.add_parameter(**parameter, name=name)
            elif isinstance(parameter, _BaseParameter):
                self.add_parameter(parameter, name=name)

    def snapshot_base(self, update: Optional[bool] = False, params_to_skip_update: Optional[Sequence[str]] = None) -> Dict[Any, Any]:
        return self(verbose=False)

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
            return self.get_parameter_values()

    def call_with_args(self, *args, **kwargs):
        pass

    def get_parameter_values(self):
        parameter_values = {}
        for parameter_snapshot in self.nested_containers.values():
            parameter_values.update(**parameter_snapshot(verbose=False))
        for name, parameter in self.parameters.items():
            parameter_values[name] == parameter['value']

        return parameter_values

    def print(self, pretty=False, comment=True):
        # Print nested parameter snapshots
        for parameter_snapshot in self.nested_containers.values():
            parameter_snapshot.print()

        # Print parameters
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
