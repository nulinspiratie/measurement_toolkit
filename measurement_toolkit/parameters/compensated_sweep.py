from qcodes import Parameter
from measurement_toolkit.parameters.poly_parameter import PolyParameter
from measurement_toolkit.parameters.DC_line_parameter import DCLine
from measurement_toolkit.parameters.conductance_parameter import ConductanceParameter


class CompensatedSweep(PolyParameter):
    def __init__(
        self, 
        compensating_gate: DCLine, 
        conductance_parameter: ConductanceParameter,
        target_conductance: float,
        cross_capacitances: dict[DCLine, float],
        conductance_coefficient=0,
        max_voltage_compensation=10e-3,
        **kwargs
    ):
        measure_parameters = [
            Parameter('conductance', unit='2e^2/h'),
            Parameter('conductance_error', unit='2e^2/h'),
            Parameter('capacitance_compensation', unit='V'),
            Parameter('conductance_error_compensation', unit='V'),
            Parameter('voltage_compensation', unit='V'),
            Parameter('initial_voltage', unit='V'),
            Parameter('target_voltage', unit='V'),
        ]

        super().__init__(name=compensating_gate.name, measure_parameters=measure_parameters, **kwargs)

        assert all(isinstance(key, DCLine) for key in cross_capacitances)

        self.compensating_gate = compensating_gate
        self.conductance_parameter = conductance_parameter
        self.target_conductance = target_conductance
        self.cross_capacitances = cross_capacitances
        self.conductance_coefficient = conductance_coefficient
        self.max_voltage_compensation = max_voltage_compensation

        self._latest_voltages = {
            gate: gate.get_latest() for gate in self.cross_capacitances
        }
        self.results = {}

    def calculate_compensation(self, execute=False):
        voltage_compensation = 0

        # Calculate compensation due to cross-capacitance to other gates
        capacitance_compensation = 0
        for gate, alpha in self.cross_capacitances.items():
            voltage_difference = gate.get_latest() - self._latest_voltages[gate]
            capacitance_compensation -= alpha * voltage_difference
        voltage_compensation += capacitance_compensation

        # Calculate compensation shift due to change in conductance
        # Measure how much the conductance is off
        conductance = self.conductance_parameter()
        conductance_error = conductance - self.target_conductance
        conductance_error_compensation = self.conductance_coefficient * conductance_error
        voltage_compensation += conductance_error_compensation

        # Upper bound the voltage compensation
        voltage_compensation = min(voltage_compensation, self.max_voltage_compensation)
        voltage_compensation = max(voltage_compensation, -self.max_voltage_compensation)

        # Calculate target voltage
        initial_voltage = self.compensating_gate.get_latest()
        target_voltage = initial_voltage + voltage_compensation

        if execute:
            self.compensating_gate(target_voltage)
            for gate in self.cross_capacitances:
                self._latest_voltages[gate] = gate.get_latest()

        self.results = {
            'conductance': conductance,
            'conductance_error': conductance_error,
            'capacitance_compensation': capacitance_compensation,
            'conductance_error_compensation': conductance_error_compensation,
            'voltage_compensation': voltage_compensation,
            'initial_voltage': initial_voltage,
            'target_voltage': target_voltage
        }

        return self.results

    def get_raw(self):
        results = self.calculate_compensation(execute=True)

        return_results = []
        # TODO This part should be improved
        for parameter in self.measure_parameters:
            if parameter.enabled:
                return_results.append(results[parameter.name])

        return tuple(return_results)