from qcodes import Parameter
from measurement_toolkit.parameters.DC_line_parameter import sweep_gate_to


class VoltageCombinerParameter(Parameter):
    precision = 6
    def __init__(self, name, coarse_voltage_parameter, fine_voltage_parameter, unit='V', **kwargs):
        super().__init__(name=name, unit=unit, **kwargs)

        self.coarse_voltage_parameter = coarse_voltage_parameter
        self.fine_voltage_parameter = fine_voltage_parameter

        self.use_fine = False

    @property
    def V_coarse(self):
        return self.coarse_voltage_parameter()

    @property
    def V_fine(self):
        return self.fine_voltage_parameter()

    def get_raw(self):
        return round(self.V_coarse + self.V_fine, self.precision)

    def set_raw(self, voltage):
        if self.use_fine:
            V_fine = voltage - self.V_coarse
            self.fine_voltage_parameter(V_fine)
        else:
            self.fine_voltage_parameter(0)
            self.coarse_voltage_parameter(voltage)

    def print_voltages(self):
        print(f'{self.name}({self():.8g})')
        print(f'{self.name}.coarse_voltage_parameter({self.coarse_voltage_parameter():.8g})  # raw {float(self.coarse_voltage_parameter.raw_value):.8g}')
        print(f'{self.name}.fine_voltage_parameter({self.fine_voltage_parameter():.8g})  # raw {float(self.fine_voltage_parameter.raw_value):.8g}')


    def sweep_to(
            self,
            target_voltage,
            initial_voltage=None,
            step=10e-3,
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
