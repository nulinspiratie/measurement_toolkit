from qcodes import (
    load_by_run_spec,
)
from qcodes.utils.dataset.doNd import do1d, do2d, dond, plot, LinSweep, LogSweep

from measurement_toolkit.parameters.general_parameters import RepetitionParameter

import matplotlib.ticker as ticker
from matplotlib import pyplot as plt


def measure_noise_spectrum(
        measure_lockin,
        source_lockin=None,
        time_constant=0.1,
        f_min=1,
        f_max=500,
        df=1,
        repetitions=1,
        show_progress=True,
        plot=True
):
    mask_params = {
        measure_lockin.reference_source: 'INT',
        measure_lockin.time_constant: time_constant,
    }

    if source_lockin is not None:
        mask_params[source_lockin.amplitude] = 0

    initial_params = {param: param() for param in mask_params}

    num_f = int(round((f_max - f_min + 1) / df))
    frequency_sweep = LinSweep(measure_lockin.frequency, f_min, f_max, num_f, delay=time_constant * 2)

    metadata = {
        'initial_signal': measure_lockin.R(),
        'source_lockin': source_lockin.name or None,
        # 'initial_params': {param.name: val for param, val in initial_params.items()},
        # 'mask_params': {param.name: val for param, val in mask_params.items()}
    }

    measure_param = RepetitionParameter(
        target_parameter=measure_lockin.R,
        repetitions=repetitions
    )

    try:
        for param, val in mask_params.items():
            param(val)

        dataset, _, _ = dond(frequency_sweep, measure_param, show_progress=show_progress)
        # dataset.add_metadata('measurement_info', metadata)
        # dataset.save_metadata()
    finally:
        for param, val in initial_params.items():
            param(val)


    if plot:
        plot_noise_spectrum(dataset)
    return dataset


def plot_noise_spectrum(dataset, ax=None, label=None):
    if isinstance(dataset, int):
        dataset = load_by_run_spec(captured_run_id=dataset)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    arrs = dataset.to_xarray_dataarray_dict()
    arr = next(iter(arrs.values()))
    arr.plot(ax=ax, label=label)

    ax.set_ylabel('Lockin signal R (mV)')

    yscale = 1e3
    ticks = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y * yscale))
    ax.yaxis.set_major_formatter(ticks)

    ax.grid('on')
    ax.set_yscale('log')
    return fig, ax


def measure_lockin_50Hz_noise(lockin, source_lockin=None, time_constant=0.1, repetitions=5, return_mean=True,
                              delay_scale=7):
    frequencies = [50, 150, 250]

    mask_params = {
        lockin.reference_source: 'INT',
        lockin.time_constant: time_constant,
        lockin.frequency: lockin.frequency(),
        lockin.amplitude: 0,
    }

    if source_lockin is not None:
        mask_params[source_lockin.amplitude] = 0
        mask_params[source_lockin.frequency] = 16

    initial_params = {param: param() for param in mask_params}

    results = {frequency: [] for frequency in frequencies}

    try:
        for param, val in mask_params.items():
            param(val)

        for k in np.arange(repetitions):
            for frequency in frequencies:
                lockin.frequency(frequency)
                sleep(lockin.time_constant() * delay_scale)

                # Ensure we are not overloaded
                signal_strength = lockin.signal_strength()
                if signal_strength == 4:
                    print('Lockin range overloaded. Please increase lockin.input_range()')
                sensitivity_overloaded = bool(int(lockin.ask('CUROVLDSTAT?')) % 2)
                if sensitivity_overloaded:
                    print('Lockin sensitivity overloaded. Please increase lockin.sensitivity()')

                results[frequency].append(lockin.R())

        if source_lockin is not None:
            source_lockin.amplitude(0.1)
            source_lockin.frequency(16)

            # Ensure we are not overloaded
            signal_strength = lockin.signal_strength()
            if signal_strength == 4:
                print('Lockin range overloaded. Please increase lockin.input_range()')
            sensitivity_overloaded = bool(int(lockin.ask('CUROVLDSTAT?')) % 2)
            if sensitivity_overloaded:
                print('Lockin sensitivity overloaded. Please increase lockin.sensitivity()')

            sleep(lockin.time_constant() * delay_scale)
            results['100mV_excitation'] = []
            for k in np.arange(repetitions):
                results['100mV_excitation'].append(lockin.R())
                sleep(lockin.time_constant())

    finally:
        for param, val in initial_params.items():
            param(val)

    if return_mean:
        results = {key: np.round(np.mean(value), 6) for key, value in results.items()}
    return results