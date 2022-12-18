from typing import List, Union
from pathlib import Path
import os
import numpy as np
import h5py

import qcodes as qc
from qcodes.data.hdf5_format import HDF5Format
from qcodes.dataset import get_default_experiment_id


def get_trace_filepath(
    filename_format='db{database_idx}_#{experiment_id}', 
    suffix=None, 
    ensure_new=False,
):
    assert 'trace_folder' in qc.config.user

    trace_folder = Path(qc.config.user.trace_folder)
    assert trace_folder.exists()

    database_idx = qc.config.user.database_idx
    experiment_id = get_default_experiment_id()
    assert experiment_id is not None

    # Fill in filename template using parameters
    base_filename = filename_format.format(database_idx=database_idx, experiment_id=experiment_id)

    if suffix:
        base_filename += f'_{suffix}'

    # Iterate through indices until a new file is found
    filename = base_filename + '.hdf5'
    trace_filepath = trace_filepath / filename
    if ensure_new:
        for idx in range(100):
            # Check if filename already exists:
            if not trace_filepath.exists():
                break
            
            filename = (base_filename + f'_{idx}') + '.hdf5'
            trace_filepath = trace_filepath / filename
        else:
            raise RuntimeError('All first 100 trace filenames are taken', trace_filepath)

    return trace_filepath


def get_trace_file(
    create_if_new: bool = True,
    ensure_new: bool = False,
    suffix=None,
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
    trace_filepath = get_trace_filepath(suffix=suffix, ensure_new=ensure_new)
    
    if not trace_filepath.exists() and not create_if_new:
        raise FileNotFoundError(f'Trace file {trace_filepath} does not exist')

    file = h5py.File(trace_filepath, 'a', libver='latest')

    return file
        


def save_traces(
    traces: np.ndarray,
    name: str = None,
    folder: str = None,
    precision: Union[int, None] = 7,
    compression: int = 4,
    ensure_new: bool = False,
    metadata: dict = None,
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
            
    suffix = None if qc.active_measurement() else 'post'
    with get_trace_file(create_if_new=True, ensure_new=ensure_new, suffix=suffix) as file:
        # Create traces group
        if 'traces' not in file:
            file.create_group('traces')

        # Determine name, either based on action indices or an incrementer
        if qc.active_measurement():
            action_indices = qc.active_measurement().action_indices
            traces_name = '_'.join(action_indices)
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
        if qc.active_measurement():
            loop_shape = qc.active_measurement().loop_shape
        else:
            loop_shape = ()
        array_shape = np.shape(traces)
        data_shape = tuple([*loop_shape, *array_shape])

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
        file['traces'][traces_name][loop_shape] = traces

        file.flush()
        file.swmr_mode = True  # Enable multiple readers to access this process