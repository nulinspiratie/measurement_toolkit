from contextlib import contextmanager
import numpy as np
import xarray
import numbers

import qcodes
from qcodes.dataset.sqlite.database import initialise_database, initialise_or_create_database_at
from qcodes.dataset.experiment_container import load_or_create_experiment
from qcodes.dataset.data_set import load_by_run_spec

__all__ = [
    'load_data',
    'convert_to_dataset',
    'get_latest_run_id',
    'smooth',
    'retrieve_station_component',
    'magnetic_fields',
    'modify_measurement_note',
    'test_database'
]

def load_data(run_id, dataset_type='xarray', print_info=False):
    """Loads a dataset by run_id, automatically converts it to target type
    """
    assert dataset_type in ['xarray', 'xarray_dict', 'pandas', 'pandas_dict', 'qcodes']
    qcodes_dataset = load_by_run_spec(captured_run_id=run_id)
    if print_info:
        dataset_information(qcodes_dataset, silent=False)

    dataset = convert_to_dataset(qcodes_dataset, dataset_type=dataset_type)

    return dataset


def convert_to_dataset(dataset_or_run_id, dataset_type='xarray'):
    if isinstance(dataset_or_run_id, numbers.Integral):
        dataset = load_data(dataset_or_run_id, dataset_type=dataset_type)
    elif isinstance(dataset_or_run_id, qcodes.dataset.data_set.DataSet):
        if dataset_type == 'xarray':
            dataset = dataset_or_run_id.to_xarray_dataset()
        elif dataset_type == 'xarray_dict':
            dataset = dataset_or_run_id.to_xarray_dataarray_dict()
        elif dataset_type == 'pandas':
            dataset = dataset_or_run_id.to_pandas_dataframe()
        elif dataset_type == 'pandas_dict':
            dataset = dataset_or_run_id.to_pandas_dataframe_dict()
        elif dataset_type == 'qcodes':
            dataset = dataset_or_run_id
    elif isinstance(dataset_or_run_id, xarray.Dataset) and dataset_type == 'xarray':
        dataset = dataset_or_run_id
    elif not isinstance(dataset_or_run_id, qcodes.dataset.data_set.DataSet) and dataset_type == 'qcodes':
        dataset = load_data(dataset_or_run_id, dataset_type='qcodes')
    else:
        raise NotImplementedError('This conversion has not yet been implemented')
    # dataset.__set qcodes = qcodes_dataset

    return dataset


def get_latest_run_id():
    from qcodes.dataset import experiments as get_experiments
    experiments = get_experiments()
    last_experiment = experiments[-1]
    last_run_id = last_experiment.last_counter
    return last_run_id


def smooth(y, window_size, order=3, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Taken from: http://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in
                range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


def magnetic_fields(dataset=None, silent=False):
    if isinstance(dataset, int):
        dataset = load_data(dataset, 'qcodes')

    if dataset is not None:
        magnetic_fields = {}
        for key in 'XYZ':
            magnetic_fields[key] = dataset.snapshot['station']['instruments'][f'magnet_{key}']['parameters']['field']['value']
            if not silent:
                print(f'magnet_{key} = {magnetic_fields[key]:.4g} T')
    return magnetic_fields


def retrieve_station_component(dataset, component_name, return_dict=False, **kwargs):
    if isinstance(dataset, int):
        dataset = load_data(dataset, 'qcodes')

    if dataset is not None:
        from measurement_toolkit.tools.parameter_container import print_parameters_from_container
        snapshot = dataset.snapshot['station']['components'][component_name]
        if return_dict:
            return {name: info.get('value') for name, info in snapshot.items()}
        else:
            print_parameters_from_container(snapshot, **kwargs)
    else:
        raise ValueError(f'Could not extract snapshot from dataset {dataset}')


def dataset_information(dataset, silent=True):
    if isinstance(dataset, int):
        dataset = load_data(dataset, 'qcodes')

    results = {}
    try:
        results['magnetic_fields'] = magnetic_fields(dataset, silent=silent)
    except Exception:
        print('Could not extract magnetic fields')

    try:
        from measurement_toolkit.tools.instruments.qdac_tools import gate_voltages
        results['voltages'] = gate_voltages(dataset, silent=True)
        if not silent:
            gate_voltages(dataset, silent=False)
    except Exception:
        print('Could not extract gate voltages')
    return results


def modify_measurement_note(run_id=None):
    if run_id is None:
        run_id = get_latest_run_id()

    dataset = load_data(run_id, dataset_type='qcodes', silent=True)

    # Get measurement notes
    try:
        measurement_notes = dataset.get_metadata('measurement_notes')
    except RuntimeError:
        measurement_notes = ''

    import easygui
    updated_notes = easygui.codebox(
        msg=f'Measurement notes for measurement #{run_id}',
        title=f'#{run_id} notes',
        text=measurement_notes
    )

    if updated_notes is not None:
        dataset.add_metadata('measurement_notes', updated_notes)
    else:
        updated_notes = measurement_notes

    return updated_notes


@contextmanager
def test_database():
    """
    Initializes or creates a database and restores the 'db_location' afterwards.

    Args:
        db_file_with_abs_path
            Database file name with absolute path, for example
            ``C:\\mydata\\majorana_experiments.db``
    """
    assert 'test_db_location' in qcodes.config.dataset
    test_location = qcodes.config.dataset.test_db_location

    try:
        db_location = qcodes.config["core"]["db_location"]
        initialise_or_create_database_at(test_location)
        load_or_create_experiment('Test_experiment')
        yield
    finally:
        qcodes.config["core"]["db_location"] = db_location
        initialise_database()

def get_Fourier_component(arr, frequency, xvals=None):
    """Extract the Fourier component at a specific frequency"""

    if isinstance(arr, xarray.DataArray):
        xvals = list(arr.coords.values())[-1].values
        yvals = arr.values
    else:
        assert xvals is not None, "Must provide xvals if arr is not a DataArray"
        yvals = arr


    xvals_shifted = xvals - xvals[0]

    sine = np.sin(2 * np.pi * frequency * xvals_shifted)
    cosine = np.cos(2 * np.pi * frequency * xvals_shifted)
    sine_component = sine * yvals
    cosine_component = cosine * yvals

    demodulated_signal = np.sum(cosine_component) + 1.j * np.sum(sine_component)

    return {
        'demodulated_signal': demodulated_signal,
        'sine_component': sine_component,
        'cosine_component': cosine_component,
    }