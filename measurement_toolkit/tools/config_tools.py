from pathlib import Path

import qcodes as qc
from qcodes.dataset.sqlite.database import initialise_database
from qcodes.dataset.experiment_container import load_or_create_experiment

__all__ = [
    'initialize_from_config'
]

def initialize_from_config(silent=False, use_mainfolder=True):
    global database, exp

    if use_mainfolder:
        root_dir = Path(qc.config.user.mainfolder)
        assert root_dir.exists()

        # Configure qcodes.config
        # Note that user.mainfolder must be set in ~/qcodesrc.json
        qc.config.update_config(root_dir)

    database = initialise_database()

    if not silent:
        print(f'Database: {qc.config.core.db_location}')

    exp = load_or_create_experiment(
        experiment_name=qc.config.user.experiment_name,
        sample_name=qc.config.user.sample_name
    )

    if not silent:
        print(f'Experiment: {qc.config.user.experiment_name}\n'
              f'Sample: {qc.config.user.sample_name}')