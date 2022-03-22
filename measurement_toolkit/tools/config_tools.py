import os
from pathlib import Path

import qcodes as qc
from qcodes.configuration.config import Config
from qcodes.dataset.sqlite.database import initialise_database, initialise_or_create_database_at
from qcodes.dataset.experiment_container import load_or_create_experiment

__all__ = [
    'initialize_config',
    'initialize_from_config'
]


def initialize_config(
    chip_name, experiment_name, device_name, author, 
    create_db=False, verbose=False, use_mainfolder=True, silent=False
):
    global database, exp

    config = qc.config

    root_dir = Path(qc.config.user.mainfolder)
    assert root_dir.exists()

    # Load config
    if use_mainfolder:
        root_dir = Path(config.user.mainfolder)
        assert root_dir.exists()

        # Configure qcodes.config
        # Note that user.mainfolder must be set in ~/qcodesrc.json
        config.update_config(root_dir)


    # Update keyword config entries
    config.user.chip_name = chip_name
    config.user.experiment_name = experiment_name
    config.user.device_name = device_name
    config.user.author = author

    # Format all entries derived from experiment properties
    for key, val in list(config.user.items()):
        if key.endswith('_format'):
            label = key.split('_format')[0]
            config.user[label] = val.format(**config.user)

    if 'db_location_format' in config.core:
        db_location = Path(config.core.db_location_format.format(**config.user, incrementer='{incrementer}'))
        db_folder = db_location.parent

        if not db_folder.exists():
            if create_db:
                print(f'Creating database folder {db_folder}')
                os.makedirs(db_folder)
            else:
                raise FileNotFoundError(f'Cannot open database. Database folder {db_folder} does not exist')

        max_database_incrementer = 99
        database_filename = db_location.stem
        for k in range(max_database_incrementer, 0, -1):
            db_file = db_folder / db_location.name.format(incrementer=str(k))
            if db_file.exists():
                break
        else:
            if create_db:
                db_file = db_folder / db_location.name.format(incrementer=1)
                print(f'Creating new database at {db_file}')
                database = initialise_or_create_database_at(str(db_file))
            else:
                raise FileNotFoundError(f'No database found in {db_folder}. Can be created by passing kwarg "create_db=True"')

        config.core.db_location = str(db_file)
    else:
        database = initialise_database()
    print(f'Database: {qc.config.core.db_location}')

    exp = load_or_create_experiment(
        experiment_name=config.user.experiment_name,
        sample_name=config.user.sample_name
    )

    if not silent:
        print(
            f'Author: {config.user.author}\n'
            f'Experiment: {qc.config.user.experiment_name}\n'
            f'Device sample: {qc.config.user.sample_name}'
        )



def initialize_from_config(
        silent=False,
        use_mainfolder=True,
        experiment_name=None,
        sample_name=None,
        author=None
):
    global database, exp

    # Load config
    if use_mainfolder:
        root_dir = Path(qc.config.user.mainfolder)
        assert root_dir.exists()

        # Configure qcodes.config
        # Note that user.mainfolder must be set in ~/qcodesrc.json
        qc.config.update_config(root_dir)

    if experiment_name is None:
        experiment_name = qc.config.user.experiment_name
    if sample_name is None:
        sample_name = qc.config.user.sample_name

    if not silent:
        print(f'Database: {qc.config.core.db_location}')

    database = initialise_database()

    exp = load_or_create_experiment(
        experiment_name=experiment_name,
        sample_name=sample_name
    )

    if not silent:
        print(f'Experiment: {qc.config.user.experiment_name}\n'
              f'Device sample: {qc.config.user.sample_name}')