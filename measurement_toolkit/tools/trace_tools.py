from matplotlib import pyplot as plt
from typing import List, Union
from pathlib import Path
import os
import numpy as np
import h5py
from warnings import warn
from xarray import DataArray

import qcodes as qc
from qcodes.dataset.measurement_loop import running_measurement
from qcodes.dataset.sqlite.queries import get_last_run
from qcodes.dataset.sqlite.database import conn_from_dbpath_or_conn

from measurement_toolkit.tools.data_tools import load_data


__all__ = [
    'get_trace_filepath',
    'get_trace_file',
    'save_traces',
    'load_traces',
    'segment_traces',
]


def get_trace_filepath(
    filename_format='db{database_idx}_#{run_id}', 
    suffix=None, 
    ensure_new=False,
    database_idx=None,
    run_id=None,
):
    assert 'trace_folder' in qc.config.user

    trace_folder = Path(qc.config.user.trace_folder)
    if not trace_folder.exists():
        if not trace_folder.parent.exists():
            raise FileNotFoundError(f'Trace folder {trace_folder} does not exist')
        else:
            print(f'Creating trace folder {trace_folder}')
            os.mkdir(trace_folder)
        

    if database_idx is None:
        database_idx = qc.config.user.database_idx
    if run_id is None:
        conn = conn_from_dbpath_or_conn(conn=None, path_to_db=None)
        run_id = get_last_run(conn=conn)
        if run_id is None:
            run_id = 0

    # Fill in filename template using parameters
    base_filename = filename_format.format(database_idx=database_idx, run_id=run_id)

    if suffix:
        base_filename += f'_{suffix}'

    # Iterate through indices until a new file is found
    filename = base_filename + '.hdf5'
    trace_filepath = trace_folder / filename
    if ensure_new:
        for idx in range(100):
            # Check if filename already exists:
            if not trace_filepath.exists():
                break
            
            filename = (base_filename + f'_{idx}') + '.hdf5'
            trace_filepath = trace_folder / filename
        else:
            raise RuntimeError('All first 100 trace filenames are taken', trace_filepath)

    return trace_filepath


def get_trace_file(
    create_if_new: bool = True,
    ensure_new: bool = False,
    suffix=None,
    database_idx=None,
    run_id=None
):
    """Initialize an HDF5 file for saving traces
    Args:
        name: Name of trace file.
        folder: Folder path for trace file. If not set, the folder of the
            active loop dataset is used, with subfolder 'traces'.
        channels: List of channel labels to acquire. The channel labels must
            be defined as the second arg of an element in
            ``Layout.acquisition_channels``
        precision: Number of digits after the decimal points to retain.
            Set to 0 for lossless compression
        compression_level: gzip compression level, min=0, max=9
    Raises:
        AssertionError if folder is not provided and no active dataset
    """
    # Create new hdf5 file
    trace_filepath = get_trace_filepath(
        suffix=suffix, ensure_new=ensure_new, database_idx=database_idx, run_id=run_id
    )
    
    if not trace_filepath.exists() and not create_if_new:
        raise FileNotFoundError(f'Trace file {trace_filepath} does not exist')

    file = h5py.File(trace_filepath, 'a', libver='latest')

    return file
        


def save_traces(
    traces: np.ndarray,
    sweeps = None,
    ensure_new: bool = False,
    file_suffix=None,
    array_name: str = '',
    precision: Union[int, None] = 7,
    compression: int = 4,
    metadata: dict = None,
    silent=True,
):
    """Save traces to an HDF5 file.
    The HDF5 file contains a group 'traces', which contains a dataset for
    each channel. These datasets can be massive depending on the size of the
    loop, but shouldn't be an issue since HDF5 can save/load portions of the
    dataset.
    Args:
        name: Name of trace file.
            If not set, the name of the current loop parameter is used
        folder: Folder path for trace file. If not set, the folder of the
            active loop dataset is used, with subfolder 'traces'.
            ``Layout.acquisition_channels``"""
            
    msmt = running_measurement()

    if file_suffix is None:
        file_suffix = None if msmt else 'post'
    with get_trace_file(create_if_new=True, ensure_new=ensure_new, suffix=file_suffix) as file:
        if not silent:
            filepath = get_trace_filepath(ensure_new=False, suffix=file_suffix)
            print(f'Saving traces to {filepath}')
        # Create traces group
        if 'traces' not in file:
            file.create_group('traces')

        # Determine array_name, either based on action indices or an incrementer
        if array_name:
            array_name += '|'
        if msmt:
            action_indices = msmt.action_indices
            array_name += '_'.join(map(str, action_indices))
        else:
            for idx in range(9999):
                if f'{array_name}{idx}' not in file['traces']:
                    array_name = f'{array_name}{idx}'
                    break
            else:
                raise OverflowError('Could not determine traces name')

        # Determine data shape
        if msmt:
            loop_shape = msmt.loop_shape
        else:
            loop_shape = ()
        array_shape = np.shape(traces)
        data_shape = tuple([*loop_shape, *array_shape])

        if not silent:
            print(f'{loop_shape=}, {data_shape=}')

        # Create new dataset if it doesn't already exist
        if array_name not in file['traces']:
            file['traces'].create_dataset(
                name=array_name, 
                shape=data_shape,
                dtype=float, 
                scaleoffset=precision,
                chunks=True, 
                compression='gzip',
                compression_opts=compression
            )
            
        # Save metadata to array
        if metadata:
            for key, val in metadata.items():
                file['traces'][array_name].attrs[key] = val

        # Store sweeps
        if sweeps is not None:
            if 'sweeps' not in file:
                file.create_group('sweeps')
            if array_name not in file['sweeps']:
                file['sweeps'].create_group(array_name)

            array_sweeps = file['sweeps'][array_name]

            for k, sweep in enumerate(sweeps):
                param = sweep.parameter
                sweep_name = getattr(param, 'name', f'unknown_{k}')
                array_sweeps.create_dataset(
                    name=sweep_name, 
                    shape=len(sweep.sequence),
                    dtype=float, 
                )
                array_sweeps[sweep_name] = sweep.sequence
                if param.label is not None:
                    array_sweeps[sweep_name].attr['label'] = param.label
                if param.unit is not None:
                    array_sweeps[sweep_name].attr['unit'] = param.unit

        # Add traces
        loop_indices = msmt.loop_indices if msmt else ()
        file['traces'][array_name][loop_indices] = traces

        file.flush()
        file.swmr_mode = True  # Enable multiple readers to access this process


def load_traces(
    specifier=None, 
    run_id=None, 
    database_idx=None, 
    suffix=None, 
    array_name=None, 
    silent=True, 
    idxs=None,
    return_type='xarray',
    plot=False,
):
    assert return_type in ['numpy', 'xarray', 'hdf5']

    if specifier is not None:
        if '.' in specifier:
            run_id, array_name = specifier.split('.')
            run_id = int(run_id)
        elif isinstance(specifier, int) or (isinstance(specifier, str) and specifier.isnumeric()):
            # Run id is passed
            run_id = int(specifier)
        elif isinstance(specifier, str):
            array_name = specifier
        else:
            raise SyntaxError(f'Specifier {specifier} not understood')
        
    with get_trace_file(create_if_new=False, database_idx=database_idx, run_id=run_id, suffix=suffix) as file:
        if not silent:
            filepath = get_trace_filepath(ensure_new=False, database_idx=database_idx, run_id=run_id, suffix=suffix)
            print(f'Loading traces from {filepath}')

        trace_keys = list(file['traces'].keys())
        if len(trace_keys) == 1:
            array = next(iter(file['traces'].values()))
        else:
            assert array_name is not None, f"Multiple arrays found, please specify array_name from {trace_keys}"
            array_name = str(array_name)
            
            matching_arrays = []
            for trace_key in trace_keys:
                if (
                    trace_key == array_name 
                    or trace_key.startswith(array_name) 
                    or trace_key.endswith(array_name)
                ):
                    matching_arrays.append(trace_key)
            if not len(matching_arrays):
                raise ValueError(f'No array found matching {array_name}. Array names: {trace_keys}')
            elif len(matching_arrays) > 1:
                raise ValueError(f'Multiple arrays match {array_name}. Array names: {matching_arrays}')
            else:
                full_array_name = matching_arrays[0]
            array = file['traces'][full_array_name]

        if idxs:
            if not return_type == 'numpy':
                warn('return_type must be "numpy" when trace idxs are passed')
            array = array[idxs]
        elif return_type == 'numpy':
            idxs = (Ellipsis, )
            array = array[idxs]
        elif return_type == 'xarray':
            if run_id is not None and 'sample_rate' in array.attrs:
                data = load_data(run_id, print_summary=False)
                data_arr = next(iter(data.data_vars.values()))
                t_list = np.arange(np.shape(array)[-1]) / array.attrs['sample_rate']

                if (
                    len(data_arr.coords) == 1 
                    and list(data.coords.keys())[0] == 'index' 
                    and len(data.coords['index']) == 1
                ):
                    coords = {}
                else:
                    coords = {**data_arr.coords}
                
                if np.ndim(array) > len(coords) + 2:
                    # Too many unknown dimensions, simply return array without coords
                    array = DataArray(np.array(array), attrs=array.attrs)
                elif np.ndim(array) == len(coords) + 2:
                    coords['iteration'] = DataArray(np.arange(array.shape[-2]), dims='iteration')
                    coords['time'] = DataArray(t_list, attrs=dict(units='s'), dims='time')
                elif np.ndim(array) == len(coords) + 1:
                    coords['time'] = DataArray(t_list, attrs=dict(units='s'), dims='time')
                elif np.ndim(array) == len(coords):
                    pass
            array = DataArray(np.array(array), coords=coords, attrs=array.attrs)

        elif return_type == 'hdf5':
            pass

        if plot and return_type == 'xarray' and array.ndim in [1, 2]:
            array.plot()
            fig = plt.gcf()
            ax = plt.gca()
            ax.set_clim()
            if run_id is not None:
                fig.suptitle(f'#{run_id} RF traces')

    return array


def segment_traces(traces, pulses, time_constant, group_pulses=True):
    assert traces.ndim == 2

    if isinstance(traces, DataArray):
        traces = traces.values

    pulse_traces = {}
    t = 0
    for pulse in pulses:
        t_start = t
        t_stop = t + pulse.duration
        idx_start = int(round(t_start / time_constant))
        idx_stop = int(round(t_stop / time_constant))
        t += pulse.duration

        pulse_traces[pulse.name] = traces[:, idx_start:idx_stop]

    if group_pulses:
        pulse_names = [pulse.name for pulse in pulses]
        pulse_group_names = set([name.split('_', maxsplit=1)[-1] for name in pulse_names])
        pulse_groups = {pulse_group_name: [] for pulse_group_name in pulse_group_names}

        for pulse_name, pulse_trace in pulse_traces.items():
            pulse_group_name = pulse_name.split('_', maxsplit=1)[-1] 
            pulse_groups[pulse_group_name].append(pulse_trace)

        # Convert to array
        for pulse_group_name, pulse_group in pulse_groups.items():
            pulse_groups[pulse_group_name] = np.array(pulse_group)

        return pulse_groups
    else:
        return pulse_traces