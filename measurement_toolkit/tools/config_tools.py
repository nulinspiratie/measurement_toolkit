from pathlib import Path

import qcodes as qc
from qcodes.dataset.sqlite.database import initialise_database
from qcodes.dataset.experiment_container import load_or_create_experiment

__all__ = [
    'initialize_from_config'
]

def initialize_from_config(
        silent=False,
        use_mainfolder=True,
        experiment_name=None,
        sample_name=None
):
    global database, exp

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
