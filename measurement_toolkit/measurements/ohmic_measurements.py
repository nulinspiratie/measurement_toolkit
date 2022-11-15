import numpy as np
from time import sleep

from qcodes.dataset import MeasurementLoop, Sweep
from measurement_toolkit.tools.gate_tools import iterate_gates
from measurement_toolkit.tools.data_tools import load_data

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
        line_resistances = {ohmic.name: ohmic.line_resistance for ohmic in ohmics}
        msmt.dataset.add_metadata('line_resistances', str(line_resistances))

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


def analyse_ohmic_combinations(data_idx, resistance_max=100e3):
    data = load_data(data_idx)
    
    ohmic_labels = data.attrs['ohmics']

    resistance_capped = data.resistance.copy()
    if resistance_max is not None:
        resistance_capped = np.abs(resistance_capped)
        resistance_capped = resistance_capped.clip(max=resistance_max)

    fig, ax = plt.subplots()
    resistance_capped.plot()

    ax.set_xlabel('Source ohmic')
    ax.set_ylabel('Drain ohmic')
    ax.set_xticks(range(len(ohmic_labels)))
    ax.set_xticklabels(ohmic_labels, rotation=65)
    ax.set_yticks(range(len(ohmic_labels)))
    ax.set_yticklabels(ohmic_labels);

    for i in range(len(ohmic_labels)):
        for j in range(len(ohmic_labels)):
            resistance = resistance_capped[i, j].values
            if resistance == resistance_max:
                resistance_str = f'>{resistance_max/1e3:.0f}'
            else:
                resistance_str = str(np.round(resistance / 1e3, 1))
            text = ax.text(j, i, resistance_str,
                        ha="center", va="center", color="r")
    
    return fig, ax