
from functools import partial

import qcodes as qc
from qcodes import Parameter
from qcodes.utils import validators as vals
from measurement_toolkit.parameters import ConductanceParameter


class LockinTimeParameter(Parameter):
    def __init__(self, lockins, name='t_lockin', unit='s', **kwargs):
        super().__init__(name=name, unit=unit, **kwargs)
        self.lockins = lockins

    def get_raw(self):
        time_constants = [lockin.time_constant() for lockin in self.lockins]
        assert len(set(time_constants)) == 1
        return time_constants[0]

    def set_raw(self, time_constant):
        for lockin in self.lockins:
            lockin.time_constant(time_constant)


def configure_lockins(
        lockins, 
        source_lockin, 
        excitation_scale=1e5,
        namespace=None, 
        update_monitor=True
):
    station = qc.Station.default
    assert station is not None
    
    # AC excitation
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

    # Lockin current and conductance
    current_parameters = []
    conductance_parameters = []
    
    if isinstance(lockins, list):
        lockins = enumerate(lockins, start=1)
        
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