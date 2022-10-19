import numpy as np
from functools import partial

import qcodes as qc
from qcodes import Parameter
from qcodes.utils import validators as vals
from measurement_toolkit.parameters import ConductanceParameter
from measurement_toolkit.tools.general_tools import property_ignore_setter


class LockinTimeParameter(Parameter):
    def __init__(self, lockins, name='t_lockin', unit='s', **kwargs):
        super().__init__(name=name, unit=unit, **kwargs)
        self.lockins = lockins
        self._delay = None
        self.delay_scale = 1

    def iterate_lockins(self):
        if isinstance(self.lockins, dict):
            return list(self.lockins.values())
        else:
            return self.lockins

    def get_raw(self):
        time_constants = [lockin.time_constant() for lockin in self.iterate_lockins()]
        assert len(set(time_constants)) == 1, f"Lockin time constants not equal: {time_constants}"
        return time_constants[0]

    def set_raw(self, time_constant):
        for lockin in self.iterate_lockins():
            lockin.time_constant(time_constant)

        if self._delay is not None:
            self._delay = None
            print(f'Resetting lockin delay to {self.delay_scale*time_constant} s ({self.delay_scale}*t_lockin)')

    @property
    def delay(self):
        if self._delay is not None:
            return self._delay
        else:
            return self() * self.delay_scale

    @delay.setter
    def delay(self, delay):
        self._delay = delay


def configure_lockins(*lockins):asdas
    assert station is not None
    
    # Add Xnoise and Ynoise parameters
    def get_lockin_noise(lockin, quadrature='R'):
        assert quadrature in 'XYR'

        Xnoise, Ynoise = lockin.get_values('Xnoise', 'Ynoise')
        if quadrature == 'X':
            return Xnoise
        elif quadrature == 'Y':
            return Ynoise
        else: ## quadrature == 'R'
            return np.sqrt(Xnoise**2 + Ynoise**2)

    for lockin in lockins:
        for axis in 'XYR':
            lockin.parameters.pop(f'{axis}noise', None)
            lockin.add_parameter(f'{axis}noise', unit='V', get_cmd=partial(get_lockin_noise, lockin, axis))

    # Add parameter to set all lockin time constants
    from measurement_toolkit.tools.instruments.lockin_tools import LockinTimeParameter
    t_lockin = LockinTimeParameter(lockins=lockins)
    station.add_component(t_lockin)
    
    return t_lockin


def adapt_lockins_to_conductance_paths(lockins, conductance_parameters, lockin_dependencies=(),):
    def get_master_lockin(lockin):
        try:
            return next(
                master_lockin for slave_lockin, master_lockin in lockin_dependencies
                if slave_lockin == lockin
            )
        except StopIteration:
            return None

    excitation_lockins = {
        conductance_parameter.excitation_lockin
        for conductance_parameter in conductance_parameters
    }

    if isinstance(lockins, dict):
        lockins = list(lockins.values())

    # Setup amplitudes and phases
    for conductance_parameter in conductance_parameters:
        # TODO Ideally shouldn't be hardcoded
        conductance_parameter.excitation_line.V_AC(5e-6)
        if conductance_parameter.excitation_line == conductance_parameter.measure_line:
            conductance_parameter.measure_lockin.phase(180)
        else:
            conductance_parameter.measure_lockin.phase(0)
    for lockin in lockins:
        if lockin not in excitation_lockins:
            lockin.amplitude(0)

    # First set all lockins that measures itself to internal reference
    for conductance_parameter in conductance_parameters:
        if conductance_parameter.excitation_lockin == conductance_parameter.measure_lockin:
            conductance_parameter.excitation_lockin.reference_source('INT')
    
    # Then set relevant ext references depending on reference direction
    # Note that this can overwrite previous internal reference sources
    for conductance_parameter in conductance_parameters:
        if conductance_parameter.excitation_lockin != conductance_parameter.measure_lockin:
            measure_master_lockin = get_master_lockin(conductance_parameter.measure_lockin)
            excitation_master_lockin = get_master_lockin(conductance_parameter.excitation_lockin)
            
            if measure_master_lockin == conductance_parameter.excitation_lockin:
                conductance_parameter.measure_lockin.reference_source('EXT')
                conductance_parameter.excitation_lockin.reference_source('INT')
            elif excitation_master_lockin == conductance_parameter.measure_lockin:
                conductance_parameter.measure_lockin.reference_source('INT')
                conductance_parameter.excitation_lockin.reference_source('EXT')
            else:
                raise ValueError('Cannot configure lockins: wrong dependency')