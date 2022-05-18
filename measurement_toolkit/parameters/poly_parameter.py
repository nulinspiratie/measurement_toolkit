from qcodes.instrument.parameter import MultiParameter
from measurement_toolkit.tools.general_tools import property_ignore_setter


class PolyParameter(MultiParameter):
    def __init__(self, name, measure_parameters):
        self.measure_parameters = measure_parameters
        super().__init__(
            name=name, 
            names=self.names,
            shapes=self.shapes
        )

        # Add property enabled to all parameters
        for parameter in self.measure_parameters:
            parameter.enabled = True

    @property
    def enabled_parameters(self):
        return tuple([parameter for parameter in self.measure_parameters if getattr(parameter, 'enabled', True)])

    @property_ignore_setter
    def names(self):
        return tuple([parameter.name for parameter in self.enabled_parameters])

    @property_ignore_setter
    def labels(self):
        return tuple([parameter.label for parameter in self.enabled_parameters])

    @property_ignore_setter
    def units(self):
        return tuple([parameter.unit for parameter in self.enabled_parameters])

    @property_ignore_setter
    def setpoints(self):
        return tuple([() for parameter in self.enabled_parameters])

    @property_ignore_setter
    def shapes(self):
        return tuple([getattr(parameter, 'shape', ()) for parameter in self.enabled_parameters])