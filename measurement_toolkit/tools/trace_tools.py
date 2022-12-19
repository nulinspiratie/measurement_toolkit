from typing import List, Union
from pathlib import Path
import os
import numpy as np
import h5py

import qcodes as qc
from qcodes.dataset.measurement_loop import running_measurement
from qcodes.dataset.sqlite.queries import get_last_run
from qcodes.dataset.sqlite.database import conn_from_dbpath_or_conn

def get_trace_filepath(
    filename_format='db{database_idx}_#{run_id}', 
    suffix=None, 
    ensure_new=False,
    database_idx=None,
    run_id=None,
):
    assert 'trace_folder' in qc.config.user

    trace_folder = Path(qc.config.user.trace_folder)
    assert trace_folder.exists()

    if database_idx is None:
        database_idx = qc.config.user.database_idx
    if run_id is None:
        conn = conn_from_dbpath_or_conn(conn=None, path_to_db=None)
        run_id = get_last_run(conn=conn)
        assert run_id is not None

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
    ensure_new: bool = False,
    suffix=None,
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

    if suffix is None:
        suffix = None if msmt else 'post'
    with get_trace_file(create_if_new=True, ensure_new=ensure_new, suffix=suffix) as file:
        if not silent:
            filepath = get_trace_filepath(ensure_new=False, suffix=suffix)
            print(f'Saving traces to {filepath}')
        # Create traces group
        if 'traces' not in file:
            file.create_group('traces')

        # Determine name, either based on action indices or an incrementer
        if msmt:
            action_indices = msmt.action_indices
            traces_name = '_'.join(map(str, action_indices))
        else:
            for idx in range(9999):
                if str(idx) not in file['traces']:
                    traces_name = str(idx)
                    break
            else:
                raise OverflowError('Could not determine traces name')
            
        # Save metadata to traces file
        if metadata:
            file.attrs.setdefault(traces_name, {})

            for key, val in metadata.items():
                file.attrs[traces_name][key] = val

        # Determine data shape
        if msmt:
            loop_shape = msmt.loop_shape
        else:
            loop_shape = ()
        array_shape = np.shape(traces)
        data_shape = tuple([*loop_shape, *array_shape])
        print(f'{loop_shape=}, {data_shape=}')

        # Create new dataset if it doesn't already exist
        if traces_name not in file['traces']:
            file['traces'].create_dataset(
                name=traces_name, 
                shape=data_shape,
                dtype=float, 
                scaleoffset=precision,
                chunks=True, 
                compression='gzip',
                compression_opts=compression
            )

        # Add traces
        loop_indices = msmt.loop_indices if msmt else ()
        file['traces'][traces_name][loop_indices] = traces

        file.flush()
        file.swmr_mode = True  # Enable multiple readers to access this process


def load_traces(run_id=None, database_idx=None, suffix=None, array_name=None, silent=True, idxs=None):

    with get_trace_file(create_if_new=False, database_idx=database_idx, run_id=run_id, suffix=suffix) as file:
        if not silent:
            filepath = get_trace_filepath(ensure_new=False, database_idx=database_idx, run_id=run_id, suffix=suffix)
            print(f'Loading traces from {filepath}')

        trace_keys = list(file['traces'].keys())
        if len(trace_keys) == 1:
            array = next(iter(file['traces'].values()))
        else:
            assert array_name is not None, f"Multiple arrays found, please specify name in {trace_keys}"
            array_name = str(array_name)
            assert array_name in trace_keys, f"Array name {array_name} not found, please choose from {trace_keys}"
            array = file['traces'][array_name]

        if not idxs:
            idxs = (Ellipsis, )
        array = array[idxs]

    return array