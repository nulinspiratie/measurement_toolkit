from time import sleep

"""
Example:

for gate in iterate_gates(gates, sort=True):
	measure_gate_leakage(gate, voltages=voltages, current_limit=1e-9)
"""


def iterate_gates(gates, sort=True, silent=False):
	assert all(gate.DC_line is not None for gate in gates)

	if sort:
		sorted_gates = sorted(gates, key=lambda gate: gate.DC_line)  # TODO verify this line is correct
	else:
		sorted_gates = gates

	breakout_box = None
	for gate in sorted_gates:
		if not silent:
			# Check if we should emit notification to switch breakout box
			if gate.breakout_box != breakout_box:
				print(f'Switch to breakout box {breakout_box}')
				input()
				breakout_box = gate.breakout_box

			print(f'Connect breakout box {gate.breakout_box} idx {gate.breakout_idx}')
			for breakout_idx in gate.breakout_idxs[1:]:
				print(f'Float breakout box {gate.breakout_box} idx {gate.breakout_idx}')
			input()

		yield gate
	print('Finished measuring all gates')


def measure_leakage(current_limit, num_attempts, message=None, delay=0, wait_for_input=False):
	for k in range(num_attempts):
		current = measure_current()  # TODO

		if abs(current) < current_limit:
			return {
				'leakage': False,
				'current': current
			}
		else:
			if message is not None:
				print(f"Gate {gate} current {current*1e9:.1f} nA above limit {current_limit*1e9:.1f} nA. "
				      + message + f" Trying {num_attempts - k-1} more time")
			if wait_for_input:
				input()
			elif delay:
				sleep(delay)
	else:
		return {
			'leakage': True,
			'current': current
		}


def measure_gate_leakage(gate, voltages, current_limit):
	voltage_start, *sweep_voltages = voltages

	# Measure if the start voltage is already leaking
	gate(voltage_start)
	results = measure_leakage(
		current_limit=current_limit, 
		num_attempts=2,
		message="Are you sure you're ungrounded?",
		wait_for_input=True
	)
	if results['leakage']:
		print('Not starting gate sweep because of leakage')
		return False

	# Sweep voltages and check for leakage
	currents = []
	for k, V in enumerate(sweep_voltages):
		gate(V)

		results = measure_leakage(
			current_limit=current_limit, 
			num_attempts=3,
			message="",
			wait_for_input=False,
			delay=0.2
		)

		if results['leakage']:
			print(f'Gate {gate} is leaking at {V:.4f} V. Stopping.')
			leakage = True
			break
	else:
		leakage = False

	return {
		'leakage': leakage,
		'final_voltage': V,
		'voltages': np.array(voltages),
		'currents': np.array(currents)
	}
