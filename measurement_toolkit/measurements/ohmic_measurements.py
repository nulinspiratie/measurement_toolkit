import numpy as np
from time import sleep

from qcodes.dataset import MeasurementLoop, Sweep
from measurement_toolkit.tools.gate_tools import iterate_gates


def measure_ohmic_combinations(
    ohmics, 
    voltages, 
    voltage_set_parameter,
    current_get_parameter,
    initial_actions=(), 
    final_actions=(),
    delay=0.1,
):
    # Measure ohmic matrix using a singular voltage source and current measurement

    connected_ohmics = [ohmic for ohmic in ohmics if ohmic.DC_line is not None]
    connected_ohmics = sorted(connected_ohmics, key=lambda ohmic: ohmic.DC_line)
    print(f'Testing {len(connected_ohmics)} ohmics in different pair configurations')

    # Ensure all ohmics are ungrounded
    breakout_idxs = [
        f'{ohmic.breakout_box}.{ohmic.breakout_idx}'
        for ohmic in connected_ohmics
    ]
    print(f'Please unground all ohmics with breakout idxs:', breakout_idxs) 

    with MeasurementLoop('Ohmic_tests') as msmt:
        # Add ohmics to metadata
        msmt.dataset.add_metadata('ohmics', str([o.name for o in ohmics]))

        # Perform initial actions
        for action, args in initial_actions:
            action(*args)
        # Register final actions
        for action, args in final_actions:
            msmt.final_actions.append(action)

        # Create iterator for source ohmics, handles print statements
        source_ohmic_iterator = iterate_gates(
            connected_ohmics, sort=False, active_switch='bus'
        )

        for k1, source_ohmic in zip(
            Sweep(range(len(connected_ohmics)), 'source_ohmic_idx'),
            source_ohmic_iterator
        ):
            # Create iterator for drain ohmics, handles print statements
            drain_ohmic_iterator = iterate_gates(
                connected_ohmics, sort=False, active_switch='ground'
            )

            for k2, drain_ohmic in zip(
                Sweep(range(len(connected_ohmics)), 'drain_ohmic_idx'),
                drain_ohmic_iterator
            ):
                if k1 == k2:  # Skip if ohmics are the same
                    continue

                # Sweep source ohmic
                currents = np.zeros(len(voltages))
                for k, V in enumerate(Sweep(voltage_set_parameter, voltages, delay=delay)):
                    currents[k] = msmt.measure(current_get_parameter)
                voltage_set_parameter(0)

                conductance, offset = np.polyfit(voltages, currents, deg=1)
                resistance = 1 / conductance
                msmt.measure(resistance, 'resistance', unit='ohm')
    print(f'Measurement finished')