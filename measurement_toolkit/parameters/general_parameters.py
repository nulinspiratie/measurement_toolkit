from typing import List
from time import sleep
import numpy as np

import qcodes as qc
from qcodes.instrument.parameter import Parameter

from qcodes import config

properties_config = config.get('properties', {})
pulse_config = config.get('pulses', {})
__all__ = [
    'CombinedParameter', 
    'AttributeParameter',
    'RepetitionParameter'
]

class _OffsetParameter(Parameter):
    def __init__(self, parameter, combined_parameter):
        super().__init__(
            name=f'{parameter.name}_offset', 
            label=f'{parameter.label} offset',
            unit=parameter.unit
        )
        self.combined_parameter = combined_parameter
        self.parameter = parameter

        self._value = 0

    def set_raw(self, value):
        combined_val = self.combined_parameter.get_latest()
        self._value = value
        self.combined_parameter(combined_val)
        
    def get_raw(self):
        return self._value



class CombinedParameter(Parameter):
    delay_between_set = None
    """Combines multiple parameters into a single parameter.
    Setting this parameter sets all underlying parameters to this value, after
    applying possible scale and offset in that order.
    Getting this parameter gets the value of the first parameter, and applies
    offset and scale in that order.
    Args:
        parameters: Parameters to be combined.
        name: Name of ``CombinedParameter``, by default equal to the names of
            the composed parameters separated by underscores.
        label: Label of ``CombinedParameter``, by default equal to the labels of
            the composed parameters separated by ``and``. Also includes any
            scale and offset.
        unit: Parameter unit.
        offsets: Optional offset for parameters. If set, must have equal number
            of elements as parameters
        scales: Optional scale for parameters. If set, must have equal number
            of elements as parameters.
        full_label: Add scales and offsets for all parameters to the label
        **kwargs: Additional kwargs passed to ``Parameter``.
    Note:
        * All args are also attributes.
        * While QCoDeS already has a ``CombinedParameter``, it has some
          shortcomings which are addressed here. Maybe in the future this will
          be PR'ed to the main QCoDeS repository.
    """
    def __init__(self,
                 parameters: List[Parameter],
                 name: str = None,
                 label: str = '',
                 unit: str = None,
                 offsets: List[float] = None,
                 scales: List[float] = None,
                 full_label: bool = True,
                 max_difference: float = None,
                 precision: float = None,
                 **kwargs):
        if name is None:
            name = '_'.join([parameter.name for parameter in parameters])

        self.label = None
        if unit is None:
            unit = parameters[0].unit

        self.parameters = parameters
        self.offsets = offsets
        self.scales = scales
        self.max_difference = max_difference
        self.precision = precision

        self.offset_parameters = {
            parameter.name: _OffsetParameter(parameter=parameter, combined_parameter=self)
            for parameter in parameters
        }

        self.full_label = full_label

        super().__init__(name, label=label, unit=unit, **kwargs)

        self._meta_attrs += ['offsets', 'scales']

    @property
    def label(self):
        if self._label:
            return self._label

        if self.scales is None and self.offsets is None:
            return ' and '.join([parameter.label for parameter in self.parameters])
        else:
            labels = []
            for k, parameter in enumerate(self.parameters):
                if not self.full_label:
                    labels.append(parameter.name)
                    continue

                label = parameter.name

                if self.offsets is not None and self.offsets[k] != 0:
                    sign = '+' if self.offsets[k] < 0 else '-'
                    label += f' {sign} {abs(self.offsets[k]):.4g}'
                    
                if self.scales is not None and self.scales[k] != 1:
                    label = f'({label}) / {self.scales[k]:.3g}'

                labels.append(label)

            return f'{", ".join(labels)}'

    @label.setter
    def label(self, label):
        self._label = label

    def zero_offset(self, offset=0):
        """Use current values of parameters as offsets."""
        if self.scales is not None:
            self.offsets = [param() - offset * scale for param, scale in
                       zip(self.parameters, self.scales)]
        else:
            self.offsets = [param() for param in self.parameters]

        # Also set the offsets to zero
        for parameter in self.offset_parameters.values():
            parameter._value = 0
        return self.offsets

    def calculate_individual_values(self, value):
        """Calulate values of parameters from a combined value
        Args:
            value: combined value
        Returns:
            list of values for each parameter
        """
        vals = []
        for k, parameter in enumerate(self.parameters):
            val = value
            if self.scales is not None:
                val *= self.scales[k]
            if self.offsets is not None:
                val += self.offsets[k]
            val += self.offset_parameters[parameter.name]()
            vals.append(val)

        return vals

    def get_raw(self):
        raw_values = values = [param() for param in self.parameters]

        values = [value - offset_parameter() for value, offset_parameter in zip(values, self.offset_parameters.values())]

        if self.offsets is not None:
            values = [value - offset for value, offset in zip(values, self.offsets)]

        if self.scales is not None:
            values = [value / scale for value, scale in zip(values, self.scales)]

        if self.max_difference is not None:
            assert max(values) - min(values) < self.max_difference, f"Too much spread in param values: {values}, raw values: {raw_values}"

        mean_value = np.mean(values)

        if self.precision is not None:
            mean_value = round(mean_value, self.precision)

        return mean_value

    def set_raw(self, value):
        individual_values = self.calculate_individual_values(value)
        for parameter, val in zip(self.parameters, individual_values):
            parameter(val)
            if self.delay_between_set is not None:
                sleep(self.delay_between_set)


class AttributeParameter(Parameter):
    """Creates a parameter that can set/get an attribute from an object.
    Args:
        object: Object whose attribute to set/get.
        attribute: Attribute to set/get
        is_key: whether the attribute is a key in a dictionary. If not
            specified, it will check if ``AttributeParameter.object`` is a dict.
        **kwargs: Additional kwargs passed to ``Parameter``.
    """
    def __init__(self,
                 object: object,
                 attribute: str,
                 name: str = None,
                 is_key: bool = None,
                 **kwargs):
        name = name if name is not None else attribute
        super().__init__(name=name, **kwargs)

        self.object = object
        self.attribute = attribute
        self.is_key = isinstance(object, dict) if is_key is None else is_key

    def set_raw(self, value):
        if not self.is_key:
            setattr(self.object, self.attribute, value)
        else:
            self.object[self.attribute] = value

    def get_raw(self):
        if not self.is_key:
            value = getattr(self.object, self.attribute)
        else:
            value = self.object[self.attribute]
        return value


class RepetitionParameter(Parameter):
    def __init__(
        self, 
        target_parameter, 
        repetitions=1, 
        delay=0,
        **kwargs
        ):
        for key in ['name', 'label', 'unit']:
            kwargs.setdefault(key, getattr(target_parameter, key))
        super().__init__(**kwargs)

        self.target_parameter = target_parameter
        self.repetitions = repetitions
        self.delay = delay

    def get_raw(self):
        self.values = []
        for k in range(self.repetitions):
            self.values.append(self.target_parameter())
            sleep(self.delay)

        self.mean_value = np.mean(self.values)
        self.std_value = np.std(self.values)

        return self.mean_value
        
