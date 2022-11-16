from time import sleep
import ast
import sys
import numpy as np
from matplotlib import pyplot as plt

from ..tools.gate_tools import iterate_gates
from ..tools.data_tools import load_data

"""
Example:

for gate in iterate_gates(gates, sort=True):
    measure_gate_leakage(gate, voltages=voltages, current_limit=1e-9)
"""

__all__ = [
    'measure_leakage',
    'measure_gate_leakage',
    'measure_gate_leakages',
    'analyse_gate_leakages',
]


def measure_leakage(
    measure_current_function,
    current_limit,
    num_attempts,
    message=None,
    delay=0,
    wait_for_input=False,
    verbose=False,
):
    """Measure if a gate is leaking current"""
    for k in range(num_attempts):
        current = measure_current_function()

        # measure_current_function can also return a list, so we make current a list
        if isinstance(current, (int, float)):
            current_list = [current]
            current_str = round(current * 1e9, 2)
        else:
            current_list = current
            current_str = str([round(I * 1e9, 2) for I in current_list])

        if verbose:
            print(f"Measured current: {current_str} nA")

        if np.max(np.abs(current_list)) < current_limit:
            return {"leakage": False, "current": current}
        else:
            if message is not None:
                print(
                    f"Current {current_str} nA above limit {current_limit*1e9:.1f} nA. "
                    + message
                    + f" Trying {num_attempts - k-1} more time(s)"
                )
            if wait_for_input:
                sys.stdout.flush()
                input()
            elif delay:
                sleep(delay)
    else:
        return {"leakage": True, "current": current}


def measure_gate_leakage(
    gate,
    voltages,
    current_limit,
    measure_current_function=None,
    verbose=False,
    delay=0.05,
    reset_voltage=True,
):
    """
    Sweep a gate and check if it starts leaking.
    Ramps back at the end.
    """
    from qcodes.dataset.measurement_loop import MeasurementLoop, Sweep, RepetitionSweep
    if measure_current_function is None:
        measure_current_function = gate.i

    original_voltage = gate()
    voltage_start, *sweep_voltages = voltages

    # Measure if the start voltage is already leaking
    gate(voltage_start)
    sleep(delay)
    results = measure_leakage(
        measure_current_function=measure_current_function,
        current_limit=current_limit,
        num_attempts=2,
        message="Are you sure you're ungrounded?",
        wait_for_input=True,
        verbose=verbose,
    )
    if results["leakage"]:
        print("Not starting gate sweep because of leakage")
        return False

    # Sweep voltages and check for leakage
    currents = []
    with MeasurementLoop(f"gate_leakage_{gate.name}") as msmt:
        for V in Sweep(gate, sweep_voltages):
            if verbose:
                print(f"Gate {gate} set to {V:.4g} V")

            sleep(delay)

            # Measure leakage current at this voltage
            results = measure_leakage(
                measure_current_function=measure_current_function,
                current_limit=current_limit,
                num_attempts=3,
                message="",
                wait_for_input=False,
                delay=0.2,
                verbose=verbose,
            )
            if isinstance(results['current'], (list, np.ndarray)):
                for k in RepetitionSweep(len(results['current']), name='idx'):
                    msmt.measure(results["current"][k], "leakage_current", unit="A")
            else:
                msmt.measure(results["current"], "leakage_current", unit="A")
            currents.append(results["current"])

            if results["leakage"]:
                print(f"Gate {gate} is leaking at {V:.4f} V. Stopping.")
                msmt.step_out()
                leakage = msmt.measure(True, "leakage")

                # Ensure voltages and currents are the same size
                voltages = voltages[: len(currents)]
                break
        else:
            leakage = msmt.measure(False, "leakage")

    # Reset voltage back to original value
    if reset_voltage:
        if verbose:
            print(f"Setting gate {gate} back to {original_voltage:.4g} V")
        gate(original_voltage)

    return {
        "leakage": leakage,
        "final_voltage": V,
        "voltages": np.array(voltages),
        "currents": np.array(currents),
    }


def measure_gate_leakages(
    gates,
    voltages,
    measure_current_function=None,
    measurement_name="gate_leakage_matrix",
    current_limit=5e-9,
    delay=100e-3,
    verbose=True,
):
    from qcodes.dataset.measurement_loop import MeasurementLoop, Sweep, RepetitionSweep
    
    # Ensure all voltages are within range
    for gate in gates:
        assert np.min(voltages) > gate.v.vals._min_value
        assert np.max(voltages) < gate.v.vals._max_value

    # Set default measure_current_function
    if measure_current_function is None:
        measure_current_function = lambda: [g.i() for g in gates]

    with MeasurementLoop(measurement_name) as msmt:
        # Add gates to metadata
        msmt.dataset.add_metadata('gates', str([g.name for g in gates]))

        for gate_idx in RepetitionSweep(len(gates), name="gate_idx"):
            gate = gates[gate_idx]

            if verbose:
                print(f'Measuring {gate=:}')

            measure_gate_leakage(
                gate=gates[gate_idx],
                voltages=voltages,
                current_limit=current_limit,
                measure_current_function=measure_current_function,
                verbose=verbose,
                delay=delay,
                reset_voltage=True,
            )
            
    print(f'Finished measuring leakages, run id #{msmt.dataset.run_id}')
    return msmt


def analyse_gate_leakages(data_idx, gate_names=None):
    data = load_data(data_idx)

    if gate_names is None:
        if 'gates' in data.attrs:
            gate_names = ast.literal_eval(data.attrs['gates'])
        else:
            raise SyntaxError('Cannot determine gate names, please add manually')

    for gate_name, is_leaking in zip(gate_names, data.leakage):
        print(f'Gate {gate_name} is {"not " if not is_leaking else ""}leaking')
        
    for k, arr in enumerate(data.leakage_current):
        fig, ax = plt.subplots()
        arr.plot(cmap='bwr')
        ax.set_title(f'Sweep gate {gate_names[k]}')
        ax.set_xlabel('Gate current')
        ax.set_xticks(range(len(gate_names)))
        ax.set_xticklabels(gate_names, rotation=35)

    return fig, ax
