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


def configure_lockins(
        lockins, 
        source_lockin, 
        excitation_scale=1e5,
        namespace=None, 
        update_monitor=True
):
    station = qc.Station.default
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

    # AC excitation parameter
    AC_excitation = qc.DelegateParameter(
        'AC_excitation',
        source=source_lockin.amplitude,
        scale=excitation_scale,
        vals=vals.Numbers(0, 20e-6),
        unit='V'
    )
    AC_excitation.ohmics = []
    def update_ohmics(AC_excitation, *ohmics):
        AC_excitation.ohmics = ohmics
        ohmics_str = '&'.join([f'{station.ohmics[ohmic].name}:DC{ohmic}' for ohmic in ohmics])
        AC_excitation.label = f"{ohmics_str} AC excitation"
    AC_excitation.update_ohmics = partial(update_ohmics, AC_excitation)
    # Update value
    AC_excitation()

    # Lockin current and conductance
    current_parameters = []
    conductance_parameters = []
    
    if isinstance(lockins, list):
        lockins = dict(enumerate(lockins, start=1))
        
    # Create current and conductance parameters
    for idx, lockin in lockins.items():
        I_lockin = qc.DelegateParameter(
            f'I_lockin{idx}',
            source=lockin.X,
            scale=1e8,
            unit='A'
        )
        current_parameters.append(I_lockin)
        
        conductance_parameter = ConductanceParameter(
            name=f'G{idx}',
            source_parameter=AC_excitation,
            I_lockin_parameter=I_lockin
        )
        conductance_parameters.append(conductance_parameter)

    # Populate namespace
    if namespace is not None:
        setattr(namespace, 'AC_excitation', AC_excitation)
        setattr(namespace, 'current_parameters', current_parameters)
        setattr(namespace, 'conductance_parameters', conductance_parameters)
        for conductance_parameter in conductance_parameters:
            setattr(namespace, conductance_parameter.name, conductance_parameter)
            
    # Populate monitor
    if update_monitor:
        # Remove any pre-existing conductance parameters
        for param in station._monitor_parameters.copy():
            if isinstance(param, ConductanceParameter):
                station._monitor_parameters.remove(param)
        for conductance_parameter in conductance_parameters:
            station._monitor_parameters.insert(0, conductance_parameter)
    
    return {
        'AC_excitation': AC_excitation,
        'current_parameters': current_parameters,
        'conductance_parameters': conductance_parameters
    }