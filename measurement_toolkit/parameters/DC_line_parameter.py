import warnings
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys
from time import sleep
from typing import Tuple, Union

import qcodes as qc
from qcodes import Parameter
from qcodes.utils.dataset.doNd import LinSweep, dond, AbstractSweep
from qcodes.dataset.plotting import plot_dataset
from qcodes.station import Station
from qcodes.utils import validators as vals


__all__ = [
    'DCLine'
]

this = sys.modules[__name__]  # Allows setting of variables via 'setattr(this, name, value)'


def convert_DC_line_to_breakout(DC_line):
    assert 0 < DC_line < 100
    # DC lines 25, 26, 51 are skipped
    DC_breakout_idx = DC_line
    if DC_line > 24:
        DC_breakout_idx -= 2
    if DC_line > 50:
        DC_breakout_idx -= 1
    assert 0 < DC_breakout_idx < 100
    DC_breakout_box = 1 + (DC_breakout_idx - 1) // 24
    DC_breakout_idx = 1 + (DC_breakout_idx - 1) % 48
    return int(DC_breakout_box), int(DC_breakout_idx)


def convert_breakout_to_DC_line(breakout_box, breakout_idx):
    DC_line = breakout_idx
    if breakout_box > 2:
        DC_line += 3 + 48
    elif breakout_box == 2:
        DC_line += 2
    return DC_line

def sweep_gate_to(
        gate,
        target_voltage,
        initial_voltage=None,
        step=None,
        num=None,
        delay=None,
        sweep=None,
        measure=True,
        show_progress=True,
        plot=True,
        **kwargs
):
    station = qc.Station.default
    assert station is not None, "No station initialized"

    if not measure:
        gate.ramp_voltage(target_voltage)

    if delay is None:
        if hasattr(station, 't_lockin'):
            delay = station.t_lockin.delay
        else:
            print('station.t_lockin not defined, using delay 200 ms')
            delay = 0.2

    if initial_voltage is None:
        initial_voltage = gate()
    else:
        # Go to initial voltage, then wait for system to settle
        gate(initial_voltage)
        sleep(delay)

    if num is None:
        if step is not None:
            num = int(np.ceil(abs((target_voltage - initial_voltage) / step))) + 1
        else:
            # Choose minimum of 101 steps, more if each step would be more than 10 mV
            step = 10e-3
            step_num = int(np.ceil(abs((target_voltage - initial_voltage) / step))) + 1
            num = max(step_num, 101)

    if sweep is None:
        sweeps = [LinSweep(gate, initial_voltage, target_voltage, num, delay)]
    elif isinstance(sweep, AbstractSweep):
        sweeps = [sweep, LinSweep(gate, initial_voltage, target_voltage, num, delay)]
    else:
        sweeps = [*sweep, LinSweep(gate, initial_voltage, target_voltage, num, delay)]
    dataset, _, _ = dond(
        *sweeps,
        *station.measure_params,
        show_progress=show_progress,
        measurement_name=f'1D_gate_sweep_{gate.name}',
        **kwargs
    )

    if plot:
        num_measurements = len(station.measure_params)
        fig, ax = plt.subplots()
        plot_dataset(dataset, axes=[ax] * num_measurements)
        labels = [param.label for param in station.measure_params]
        ax.legend(labels)
    return dataset


class DCLine(Parameter):
    current_limit = 3e-9

    def __init__(
        self,
        name: str,
        line_type: str,
        DC_line: Union[int, Tuple[int]],
        AC_line: int,
        RF_line: int,
        DAC_channel: int,
        breakout_box: int,
        breakout_idx: int,
        V_min: float,
        V_max: float,
        lockin_out: int = None,
        lockin_in: int = None,
        voltage_scale: float = None,
        side: str = None,
        additional_DC_lines: str = None,
        skip: bool = False,
        leakage: bool = False,
        line_resistance: float = None,
        notes: str = '',
        tex_label: str = '',
        verify_no_leakage: bool = False
    ):
        super().__init__(name=name, unit='V')

        self.line_type = line_type
        self.side = side

        if pd.isna(DC_line):
            self.DC_lines = []
        elif isinstance(DC_line, (int, float)):
            self.DC_lines = [int(DC_line)]
        elif isinstance(DC_line, (tuple, list)):
            self.DC_lines = list(DC_line)
        else:
            self.DC_lines = []

        if not pd.isna(additional_DC_lines):
            self.DC_lines += [int(float(line)) for line in str(additional_DC_lines).replace(' ', '').split(',')]
        
        self.DC_line = self.DC_lines[0] if self.DC_lines else None

        self.voltage_scale = 1 if (pd.isna(voltage_scale) or voltage_scale is None) else voltage_scale
        self.AC_line = None if pd.isna(AC_line) else int(AC_line)
        self.RF_line = None if pd.isna(RF_line) else RF_line
        self.skip = None if pd.isna(skip) else bool(skip)
        self.leakage = None if pd.isna(leakage) else bool(leakage)
        self.notes = notes
        self.tex_label = tex_label
        self.verify_no_leakage = verify_no_leakage
        self.lockin_out = None if pd.isna(lockin_out) else int(lockin_out)
        self.lockin_in = None if pd.isna(lockin_in) else int(lockin_in)
        self._V_min = V_min
        self._V_max = V_max

        if self.line_type == 'ohmic':
            self.line_resistance = line_resistance

        # Determine DC breakout idxs from DC lines
        if breakout_idx is not None and breakout_idx != ' ' and not np.isnan(breakout_idx):
            self.breakout_idx = int(breakout_idx)
            self.breakout_box = int(breakout_box)
            # self.breakout_idxs = []
        elif self.DC_line is None:
            self.breakout_idx = None
        else:
            self.breakout_box, self.breakout_idx = convert_DC_line_to_breakout(self.DC_line)
        self.breakout_idxs = [convert_DC_line_to_breakout(DC_line) for DC_line in self.DC_lines]

        # Set QDac DAC channel
        if pd.isna(DAC_channel):
            self.DAC_channel = None
            qdac_idx = None
        elif isinstance(DAC_channel, (float, int)): 
            self.DAC_channel = int(DAC_channel)
            qdac_idx = None
        elif isinstance(DAC_channel, str) and ',' in DAC_channel:
            qdac_idx, DAC_channel = DAC_channel.split(',')
            self.DAC_channel = int(DAC_channel)

        # Attach qdac
        station = qc.Station.default
        self.DAC = None
        self.V = self.v = None
        self.I = self.i = None

        if getattr(station, 'instruments_loaded', False) and self.DAC_channel:
            qdac_name = 'qdac' if qdac_idx is None else f'qdac{qdac_idx}'
            qdac = self._instrument = getattr(station, qdac_name, None)

            if qdac is not None:
                self.DAC, self.V, self.I = self.attach_QDac(
                    qdac, self._V_min, self._V_max, self.voltage_scale
                )
                qdac.parameters[name] = self
            else:
                warnings.warn(f'Could not attach {qdac_name} to {name}')

        # Attach lockin controls if there are connected lockins
        if self.lockin_out is not None:
            # Attach AC excitation parameter line.V_AC
            self.V_AC = self.attach_lockin_out(self.lockin_out)
        if self.lockin_in is not None:
            # Attach current parameter line.I_AC
            self.I_AC = self.attach_lockin_in(self.lockin_in)

        # Generate label
        self.label = self.name
        if self.line_type == 'ohmic':
            self.label += ' bias voltage'
        if self.DC_lines:
            self.label += ": " + "&".join([f"DC{line}" for line in self.DC_lines])

    def __repr__(self):
        repr_str = ''
        try:
            repr_str = self.line_type.upper()
            properties = [
                f'DC_line: {self.DC_line}',
                f'breakout_idx: {self.breakout_box}.{self.breakout_idx}',
            ]
            if self.DAC_channel is not None:
                properties.append(f'DAC: {self.DAC_channel}')

            properties_str = ', '.join(properties)

            repr_str += f'("{self.name}" {properties_str})'
        except Exception:
            warnings.warn('Could not represent DC line')
            
        return repr_str

    def attach_QDac(self, qdac, V_min, V_max, voltage_scale):
        channel = qdac.channels[self.DAC_channel - 1]

        # Set voltage limits
        dV = 30e-6  # Add a small offset so that min/max is still allowed
        channel.v.vals = qc.validators.Numbers(V_min - dV, V_max + dV)
        channel.v.scale = voltage_scale

        # Attach current, creating a separate parameter
        def get_DAC_current():
            try:
                return channel.i()
            except ValueError:
                print(f'Error retrieving {self.name} current, trying again')
                return channel.i()

        current_parameter = Parameter(
            name=f"{self.name}_DC_current",
            unit='A',
            get_cmd=get_DAC_current,
            set_cmd=channel.i
        )

        self.DAC = channel
        self.V = channel.v
        self.I = current_parameter
        self._instrument = qdac
        # Deprecated parameters
        self.v = self.V
        self.i = self.I

        return channel, channel.v, current_parameter

    def attach_lockin_out(self, lockin_out, excitation_scale=1e5):
        station = Station.default
        if station is None or not getattr(station, 'instruments_loaded', False):
            return
        elif not hasattr(station, 'lockins'):
            warnings.warn(
                f'Could not attach AC excitation to ohmic {self}. '
                f'Please attach lockins to station.lockins'
            )
            return
        else:
            lockin = station.lockins[lockin_out]

            V_AC = qc.DelegateParameter(
                f'V_AC_{self.name}',
                source=lockin.amplitude,
                scale=excitation_scale,
                vals=vals.Numbers(0, 20e-6),
                unit='V'
            )
            # Attach corresponding ohmic
            V_AC.DC_line = self

            # Update AC excitation value
            V_AC()
            
            return V_AC

    def attach_lockin_in(self, lockin_in, amplification_scale=1e8):
        station = Station.default
        if station is None or not getattr(station, 'instruments_loaded', False):
            return
        elif not hasattr(station, 'lockins'):
            warnings.warn(
                f'Could not attach AC excitation to ohmic {self}. '
                f'Please attach lockins to station.lockins'
            )
            return
        else:
            lockin = station.lockins[lockin_in]
            I_lockin = qc.DelegateParameter(
                f'I_AC_{self.name}',
                source=lockin.X,
                scale=amplification_scale,
                unit='A'
            )
            I_lockin.DC_line = self

            return I_lockin

    def get_raw(self):
        try:
            value = self.v()
        except ValueError:
            # print(f'Error retrieving {self.name} voltage, trying again')
            value = self.v()
        self.cache._update_with(value=value, raw_value=value)
        return value

    def set_raw(self, val, force=False, silent=True):
        self.v.validate(val)

        if not self.verify_no_leakage or force:
            self.v(val)
        else:
            step = self.step or self.v.step or None
            assert step is not None, "Cannot check leakage while ramping without a step"
            result = self.ramp_voltage(target_voltage=val, delay=0, step=step, silent=silent)
            if result['leakage']:
                raise RuntimeError(f'{self.label} gate leakage at {result["voltages"][-1]}V: {result["currents"][-1]}')

        self.cache._update_with(value=val, raw_value=val)

    def sweep_to(
            self,
            target_voltage,
            initial_voltage=None,
            step=None,
            num=None,
            delay=None,
            sweep=None,
            measure=True,
            show_progress=True,
            plot=True,
            **kwargs
    ):
        return sweep_gate_to(
            gate=self,
            target_voltage=target_voltage,
            initial_voltage=initial_voltage,
            step=step,
            num=num,
            delay=delay,
            sweep=sweep,
            measure=measure,
            show_progress=show_progress,
            plot=plot,
            **kwargs
        )

    def sweep_around(
            self,
            dV,
            step=None,
            num=None,
            delay=None,
            sweep=None,
            measure=True,
            show_progress=True,
            plot=True,
            **kwargs
    ):
        # Record initial voltage, also to reset later on
        V0 = self()
        
        try:
            result = sweep_gate_to(
                gate=self,
                target_voltage=V0 + dV,
                initial_voltage=V0 - dV,
            step=step,
            num=num,
            delay=delay,
            sweep=sweep,
            measure=measure,
            show_progress=show_progress,
            plot=plot,
            **kwargs
            )
        finally:
            # Reset voltage to original
            self(V0)

        return result


    def bias_scan(
            self,
            max_voltage=600e-6,
            step=None,
            num=201,
            delay=None,
            sweep=None,
            measure=True,
            show_progress=True,
            plot=True,
            **kwargs
    ):
        assert self.lockin_out is not None
        return sweep_gate_to(
            gate=self,
            target_voltage=max_voltage,
            initial_voltage=-max_voltage,
            step=step,
            num=num,
            delay=delay,
            sweep=sweep,
            measure=measure,
            show_progress=show_progress,
            plot=plot,
            **kwargs
        )

    def ramp_voltage(self, target_voltage, current_limit=None, delay=100e-3, step=10e-3, silent=True):
        if current_limit is None:
            current_limit = self.current_limit

        V0 = self()
        N_steps = int(max(abs(target_voltage - V0) / step, 3))
        voltages = np.linspace(V0, target_voltage, N_steps)
        currents = []
        for k, voltage in enumerate(voltages):
            self.v(voltage)
            sleep(delay)
            current = self.i()
            currents.append(current)

            if not silent:
                print(f"Voltage {voltage:.3g} V, current {current * 1e9:.1f} nA")

            if abs(current) >= current_limit - 0.1e-9:
                print(f'Gate started leaking at {voltage:.3g} V. '
                      f'Leakage current {current*1e9:.1f} nA > {current_limit*1e9:.1f} nA')
                return {
                    'voltages': voltages[:len(currents)],
                    'currents': np.array(currents),
                    'leakage': True
                }

        return {
            'voltages': voltages,
            'currents': np.array(currents),
            'leakage': False
        }