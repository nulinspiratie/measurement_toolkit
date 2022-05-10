from time import sleep
import sys
import numpy as np
from qcodes.dataset.measurement_loop import MeasurementLoop, Sweep
from qcodes.utils.dataset.doNd import ArraySweep

"""
Example:

for gate in iterate_gates(gates, sort=True):
    measure_gate_leakage(gate, voltages=voltages, current_limit=1e-9)
"""

def iterate_gates(gates, sort=True, silent=False):
    assert all(gate.DC_line is not None for gate in gates.values())

    if sort:
        sorted_gates = sorted(gates.values(), key=lambda gate: gate.DC_line)  # TODO verify this line is correct
    else:
        sorted_gates = gates

    breakout_box = None
    for gate in sorted_gates:
        if not silent:
            # Check if we should emit notification to switch breakout box
            if gate.breakout_box != breakout_box:
                print(f'Switch to breakout box {gate.breakout_box}')
                sys.stdout.flush()
                input()
                breakout_box = gate.breakout_box

            print(f'Connect breakout box {gate.breakout_box} idx {gate.breakout_idx}')
            for breakout_idx in gate.breakout_idxs[1:]:
                print(f'Float breakout box {gate.breakout_box} idx {gate.breakout_idx}')
            sys.stdout.flush()
            input()

            yield gate
    print('Finished measuring all gates')


def measure_leakage(measure_current_function, current_limit, num_attempts, message=None, delay=0, wait_for_input=False, verbose=False):
    """Measure if a gate is leaking current"""
    for k in range(num_attempts):
        current = measure_current_function()  # TODO

        if verbose:
            print(f'Measured current: {current*1e9:.2f} nA')

        if abs(current) < current_limit:
            return {
                'leakage': False,
                'current': current
            }
        else:
            if message is not None:
                print(f"Current {current*1e9:.1f} nA above limit {current_limit*1e9:.1f} nA. "
                      + message + f" Trying {num_attempts - k-1} more time(s)")
            if wait_for_input:
                sys.stdout.flush()
                input()
            elif delay:
                sleep(delay)
    else:
        return {
            'leakage': True,
            'current': current
        }


def measure_gate_leakage(gate, voltages, current_limit, measure_current_function=None, verbose=False, delay=0.05, reset_voltage=True):
    """
    Sweep a gate and check if it starts leaking.
    Ramps back at the end.
    """
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
        verbose=verbose
    )
    if results['leakage']:
        print('Not starting gate sweep because of leakage')
        return False

    # Sweep voltages and check for leakage
    currents = []
    with MeasurementLoop(f'gate_leakage_{gate.name}') as msmt:
        for V in Sweep(ArraySweep(gate, sweep_voltages)):
            if verbose:
                print(f'Gate {gate} set to {V:.4g} V')

            sleep(delay)

            # Measure leakage current at this voltage
            results = measure_leakage(
                measure_current_function=measure_current_function,
                current_limit=current_limit, 
                num_attempts=3,
                message="",
                wait_for_input=False,
                delay=0.2,
                verbose=verbose
            )
            msmt.measure(results['current'], 'leakage_current', unit='A')
            currents.append(results['current'])

            if results['leakage']:
                print(f'Gate {gate} is leaking at {V:.4f} V. Stopping.')
                msmt.step_out()
                leakage = msmt.measure(True, 'leakage')
                break
        else:
            leakage = msmt.measure(False, 'leakage')

    # Ensure voltages and currents are the same size
    voltages = voltages[:len(currents)]
    
    if reset_voltage:
        if verbose:
            print(f'Setting gate {gate} back to {original_voltage:.4g} V')
        gate(original_voltage)


    return {
        'leakage': leakage,
        'final_voltage': V,
        'voltages': np.array(voltages),
        'currents': np.array(currents)
    }
