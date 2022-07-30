from typing import Optional, Sequence, Dict, Any
from qcodes.instrument.parameter import _BaseParameter
from qcodes.utils.metadata import Metadatable


def print_parameters_from_container(parameters, evaluatable=True, comment=True, newline=True):
    parameter_strings = []

    for name, parameter_info in parameters.items():
        if 'unit' in parameter_info:
            unit = parameter_info['unit']
        elif 'parameter' in parameter_info:
            unit = parameter_info['parameter'].unit
        else:
            unit = ''
        formatter = parameter_info.get('formatter', '')

        if parameter_info['value_lower_bound'] is not None:
            if abs(parameter_info['value']) < parameter_info['value_lower_bound']:
                continue

        # Must be a better way to do this
        value_str = ('{:'+formatter+'}').format(parameter_info['value'])

        if evaluatable:
            parameter_string = f'{name}({value_str})'
            if comment and newline and parameter_info.get('comment'):
                parameter_string += f'  # {parameter_info["comment"]}'
        else:
            parameter_string = f'{name} = {value_str}'
            if unit:
                parameter_string += f' {parameter_info["unit"]}'
        parameter_strings.append(parameter_string)
    
    if newline:
        result = '\n'.join(parameter_strings)
    else:
        result = ', '.join(parameter_strings)
    
    print(result)


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
        super().__init__(None)
        self.name = name

        self.parameters = {}
        self.nested_containers = parameter_containers

        for name, parameter in parameters.values():
            if isinstance(parameter, dict):
                self.add_parameter(**parameter, name=name)
            elif isinstance(parameter, _BaseParameter):
                self.add_parameter(parameter, name=name)

    def snapshot_base(self, update: Optional[bool] = False, params_to_skip_update: Optional[Sequence[str]] = None) -> Dict[Any, Any]:
        snapshot = {}
        for parameter_container in self.nested_containers.values():
            snapshot.update(parameter_container.snapshot())
        self.update()
        snapshot.update(self.parameters)

        for name, parameter_info in snapshot.items():
            if 'parameter' in parameter_info:
                if 'unit' not in parameter_info:
                    parameter_info['unit'] = parameter_info['parameter'].unit
                snapshot[name] = {key: val for key, val in parameter_info.items() if key != 'parameter'}

        return snapshot

    def update(self):
        for parameter_container in self.nested_containers.values():
            parameter_container.update()
            
        for parameter_info in self.parameters.values():
            parameter = parameter_info['parameter']

            if parameter_info.get('snapshot_get', True):
                parameter_info['value'] = parameter()
            else:
                parameter_info['value'] = parameter.get_latest()

    def __call__(self, *args, **kwargs):
        if args:
            return self.call_with_args(*args, **kwargs)

        self.update()

        if kwargs.get('return_dict', False):
            self.print(evaluatable=kwargs['evaluatable'], comment=kwargs['comment'])
        else:
            self.update()
            return self.get_parameter_values()

    def call_with_args(self, *args, **kwargs):
        pass

    def get_parameter_values(self):
        parameter_values = {}
        for parameter_snapshot in self.nested_containers.values():
            parameter_values.update(**parameter_snapshot(verbose=False))
        for name, parameter in self.parameters.items():
            value = parameter['value']
            if parameter.get('formatter', '').startswith('.'):
                value = float(('{:'+parameter['formatter']+'}').format(value))
            parameter_values[name] = value

        return parameter_values

    def print(self, evaluatable=False, comment=True):
        # Print nested parameter snapshots
        for parameter_snapshot in self.nested_containers.values():
            parameter_snapshot.print(evaluatable=evaluatable, comment=comment)

        print_parameters_from_container(self.parameters, evaluatable=evaluatable, comment=comment)
        

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
