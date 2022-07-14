import warnings
import numpy as np
import qcodes as qc
from measurement_toolkit.tools.general_tools import property_ignore_setter

__all__ = ['ConductanceParameter', 'create_conductance_parameters']


class ConductanceParameter(qc.ManualParameter):
    G0 = 1 / 25813

    _label = None  # Need to define to avoid autoreload issues

    def __init__(self,
                 name,
                 excitation_line,
                 measure_line,
                 label=None,
                 **kwargs
                 ):
        self._label = label

        self.excitation_line = excitation_line
        self.measure_line = measure_line

        self.excitation_lockin = self.excitation_line.V_AC.source.instrument
        self.measure_lockin = self.measure_line.I_AC.source.instrument

        super().__init__(
            name=name,
            unit='$e^2/h$',
            **kwargs
        )

        self.values = {}

    def __repr__(self):
        source_ohmics_str = '&'.join([
            f'DC{ohmic}' for ohmic in self.excitation_line.DC_lines
        ])
        drain_ohmics_str = '&'.join([
            f'DC{ohmic}' for ohmic in self.measure_line.DC_lines
        ])
        return f'G({source_ohmics_str} → {drain_ohmics_str})'

    @property_ignore_setter
    def label(self):
        if self._label is not None:
            return self._label
        else:
            source_ohmics_str = '&'.join([
                f'DC{ohmic}' for ohmic in self.excitation_line.DC_lines
            ])
            drain_ohmics_str = '&'.join([
                f'DC{ohmic}' for ohmic in self.measure_line.DC_lines
            ])
            return f'Conductance {source_ohmics_str} → {drain_ohmics_str}'

    @property
    def drain_conductance(self):
        station = qc.Station.default
        drain_conductance = 0
        for drain_ohmic_idx in self.ohmics:
            drain_ohmic = station.ohmics[drain_ohmic_idx]
            drain_conductance += 1 / drain_ohmic.line_resistance
        return drain_conductance

    @property
    def drain_resistance(self):
        # Calculate line resistance of drain ohmics
        if self.drain_conductance > 0:
            drain_resistance = 1 / self.drain_conductance
        else:
            drain_resistance = np.NaN
        return drain_resistance

    @property
    def line_resistance(self):
        if np.isnan(self.excitation_line.line_resistance + self.measure_line.line_resistance):
            warnings.warn('No line resistance provided, please update spreadsheet.')
        return self.excitation_line.line_resistance + self.measure_line.line_resistance

    def measure(self):
        V_AC = self.excitation_line.V_AC.get_latest()
        if V_AC < 1e-7:
            warnings.warn(f'Lockin excitation {V_AC=} too low')

        I_sd = self.measure_line.I_AC()

        if I_sd != 0:
            R_total = V_AC / I_sd
            R_device = R_total - self.line_resistance 

            V_device = I_sd * R_device
            G_device = 1 / R_device / self.G0
        else:
            R_total = np.inf
            R_device = np.inf
            V_device = V_AC
            G_device = 0

        self.values = {
            'I_sd': I_sd,
            'R_total': R_total,
            'R_device': R_device,
            'V_device': V_device,
            'G_device': G_device
        }

        return self.values

    def get_raw(self):
        values = self.measure()
        return values['G_device']


def create_conductance_parameters(ohmics):
    source_ohmics = [ohmic for ohmic in ohmics if hasattr(ohmic, 'V_AC')]
    drain_ohmics = [ohmic for ohmic in ohmics if hasattr(ohmic, 'I_AC')]

    conductance_parameters = []
    for source_ohmic in source_ohmics:
        source_name = source_ohmic.name.split('Vo_')[-1]
        for drain_ohmic in drain_ohmics:
            drain_name = drain_ohmic.name.split('Vo_')[-1]
            G = ConductanceParameter(
                f'G_{source_name}_{drain_name}', 
                excitation_line=source_ohmic,
                measure_line=drain_ohmic,
                label=f'G({source_name}→{drain_name})'
            )
            conductance_parameters.append(G)
    return conductance_parameters