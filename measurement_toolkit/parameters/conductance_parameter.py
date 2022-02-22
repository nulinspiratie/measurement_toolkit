import numpy as np
import qcodes as qc
from measurement_toolkit.tools.general_tools import property_ignore_setter

__all__ = ['ConductanceParameter']


class ConductanceParameter(qc.ManualParameter):
    G0 = 1 / 25813 / 2

    def __init__(self,
                 name,
                 source_parameter,
                 I_lockin_parameter,
                 ohmics=(),
                 **kwargs
                 ):
        self.source_parameter = source_parameter
        self.ohmics = ohmics
        self.I_lockin_parameter = I_lockin_parameter

        super().__init__(
            name=name,
            unit='$e^2/h$',
            **kwargs
        )

        self.values = {}

    def __repr__(self):
        source_ohmics_str = '&'.join([
            f'DC{ohmic}' for ohmic in self.source_parameter.ohmics
        ])
        drain_ohmics_str = '&'.join([
            f'DC{ohmic}' for ohmic in self.ohmics
        ])
        return f'G({source_ohmics_str} → {drain_ohmics_str})'

    @property_ignore_setter
    def label(self):
        source_ohmics_str = '&'.join([
            f'DC{ohmic}' for ohmic in self.source_parameter.ohmics
        ])
        drain_ohmics_str = '&'.join([
            f'DC{ohmic}' for ohmic in self.ohmics
        ])
        return f'Conductance {source_ohmics_str} → {drain_ohmics_str}'

    @property
    def source_conductance(self):
        station = qc.Station.default
        source_conductance = 0
        for source_ohmic_idx in self.source_parameter.ohmics:
            source_ohmic = station.ohmics[source_ohmic_idx]
            source_conductance += 1 / source_ohmic.line_resistance
        return source_conductance

    @property
    def source_resistance(self):
        # Calculate line resistance of source ohmics
        if self.source_conductance > 0:
            source_resistance = 1 / self.source_conductance
        else:
            source_resistance = np.NaN
        return source_resistance

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
        return self.source_resistance + self.drain_resistance

    def update_ohmics(self, *ohmics):
        self.ohmics = ohmics

    def measure(self):
        R_line = self.line_resistance

        V_lockin = self.source_parameter.get_latest()
        I_sd = self.I_lockin_parameter()
        if I_sd != 0:
            R_total = V_lockin / I_sd
        else:
            R_total = np.nan
        R_device = R_total - R_line

        V_device = I_sd * R_device
        G_device = 1/R_device / self.G0

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