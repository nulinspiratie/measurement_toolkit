import os
import json
from pathlib import Path
import warnings
from functools import partial

from measurement_toolkit.tools.plot_tools import show_image
from measurement_toolkit.tools.gate_tools import initialize_DC_lines
from measurement_toolkit.tools.data_tools import retrieve_station_component

import qcodes as qc
from qcodes.dataset import (
    Measurement, 
    initialise_database, 
    initialise_or_create_database_at, 
    load_or_create_experiment,
)

__all__ = [
    'update_plottr_database',
    'initialize_config',
]


def update_plottr_database(database_path):
    if not isinstance(database_path, Path):
        database_path = Path(database_path)
    assert database_path.suffix == '.db'

    root_folder = Path(rf'C:\Users\{os.getlogin()}\AppData\Local\Packages')#\Microsoft.WindowsTerminalPreview_8wekyb3d8bbwe\LocalState'
    terminal_folder = next(
        subfolder for subfolder in root_folder.iterdir()
        if subfolder.name.startswith(f'Microsoft.WindowsTerminal')
    )

    settings_file = terminal_folder / 'LocalState' / 'settings.json'
    assert settings_file.exists(), f'Could not find settings file {settings_file}'

    # Load settings file
    settings = json.loads(settings_file.read_text())

    # Create backup settings file
    settings_file.with_name('settings_backup.json').write_text(json.dumps(settings))

    # Update settings path
    profile = next(profile for profile in settings['profiles']['list'] if profile['name'] == 'Plottr')
    command = profile['commandline'].split('--dbpath')[0]
    profile['commandline'] = f'{command} --dbpath {database_path.name}'
    profile['startingDirectory'] = str(database_path.parent)

    # Overwrite settings file
    settings_file.write_text(json.dumps(settings, indent=4))

    return settings


def load_database_from_config(create_db=False, database_idx=None, silent=False):
    config = qc.config

    db_location = Path(config.core.db_location_format.format(
        **config.user, incrementer='{incrementer}'
    ))
    db_folder = db_location.parent

    if not db_folder.exists():
        if create_db:
            os.makedirs(db_folder)
            if not silent:
                print(f'Creating database folder {db_folder}')
        else:
            raise FileNotFoundError(f'Cannot open database. Database folder {db_folder} does not exist')

    if database_idx is not None:
        db_file = db_folder / db_location.name.format(incrementer=str(database_idx))
    else:
        max_database_incrementer = 99
        for k in range(max_database_incrementer, 0, -1):
            db_file = db_folder / db_location.name.format(incrementer=str(k))
            if db_file.exists():
                break
        
    if not db_file.exists() and create_db:
        db_file = db_folder / db_location.name.format(incrementer=1)
        if not silent:
            print(f'Creating new database at {db_file}')

    if db_file.exists():
        database = initialise_or_create_database_at(str(db_file))
    else:
        raise FileNotFoundError(f'No database found in {db_folder}. Can be created by passing kwarg "create_db=True"')

    config.core.db_location = str(db_file)
    if not silent:
        print(f'Database: {qc.config.core.db_location}')

    return database


def _initialize_parameter_containers(populate_namespace=True, add_to_station=True):
    from measurement_toolkit.tools import ParameterContainer
    gate_voltages = ParameterContainer('gate_voltages')
    def _call_gate_voltages(dataset, **kwargs):
        try:
            return retrieve_station_component(dataset, **kwargs)
        except Exception:
            from measurement_toolkit.tools.instruments.qdac_tools import qdac_gate_voltages
            return qdac_gate_voltages(dataset)
    gate_voltages.call_with_args = partial(_call_gate_voltages, component_name='gate_voltages')

    system_summary = ParameterContainer(name='system_summary', parameter_containers={'gates': gate_voltages})
    system_summary.call_with_args = partial(retrieve_station_component, component_name='system_summary', return_dict=False)
    Measurement.enteractions = [[system_summary, ()]]

    station = qc.Station.default
    if add_to_station and station is not None:
        if hasattr(station, 'gate_voltages'):
            station.remove_component('gate_voltages')
        if hasattr(station, 'system_summary'):
            station.remove_component('system_summary')
        station.add_component(gate_voltages)
        station.add_component(system_summary)
        
        if hasattr(station, 'instruments_summary'):
            system_summary.nested_containers['instruments_summary'] = station.instruments_summary
    
    if populate_namespace:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is not None:
            shell.user_ns['gate_voltages'] = gate_voltages
            shell.user_ns['system_summary'] = system_summary

    return gate_voltages, system_summary

def initialize_config(
    chip_name, 
    experiment_name, 
    device_name, 
    author, 
    create_db=False,
    database_idx=None, 
    use_mainfolder=True, 
    silent=False, 
    update_plottr=False,
    show_device=True,
    sample_holder='QDevil',
    configure_device_folder=False,
    populate_namespace=True,
):
    """Initializes the config from a template config

    Also performs some ancillary functions such as creating a database if necessary,
    updating plottr database, etc.
    
    Args:
        chip_name: Name of chip, e.g. "M35_B5"
        experiment_name: Name of experiment, e.g. "ParityQubit"
        device_name: Name of the device, e.g. "P1"
        author: Name of person measuring
        create_db: Create new database if none is found in ``db_location``
        use_mainfolder: whether to use ``config.user.mainfolder`` to update the config
            If using mainfolder, the user config (located in ~/qcodesrc.json) needs to 
            contain the entry "user.mainfolder"
        silent: Whether to suppress output printing
        update_plottr: Whether to update plottr database file
        show_device: Show initialized device image.
            The template should contain key "device_image_format"
            This functionality has not been verified.

    Notes
    - It is recommended to have a main folder. To set this up, make sure that the file
      "~/qcodesrc.json" exists (copy from qcodes if it doesn't) and contains the entry
      "user.mainfolder" that points to the main folder
    """
    config = qc.config
    station = qc.Station.default

    assert sample_holder in ['QDevil', 'Sydney']

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

    # Initialize database
    if 'db_location_format' in config.core:
        database = load_database_from_config(
            create_db=create_db,
            database_idx=database_idx,
            silent=silent
        )
    else:
        database = initialise_database()

    # Load experiment
    experiment = load_or_create_experiment(
        experiment_name=config.user.experiment_name,
        sample_name=config.user.sample_name
    )

    # Print information
    if not silent:
        print(
            f'Author: {config.user.author}\n'
            f'Experiment: {qc.config.user.experiment_name}\n'
            f'Device sample: {qc.config.user.sample_name}'
        )

    # Initialize system_summary and gate_voltages ParameterContainers
    gate_voltages, system_summary = _initialize_parameter_containers(
        populate_namespace=populate_namespace,
        add_to_station=True
    )

    # Load gates and create conductance parameters
    if 'gates_file' in config.user:
        if Path(config.user.gates_file).exists():
            initialize_DC_lines(
                gates_excel_file=config.user.gates_file,
                sample_holder=sample_holder,
                parameter_container=gate_voltages,
                populate_namespace=populate_namespace
            )
    
            # Create conductance parameters
            if getattr(station, 'instruments_loaded', False):
                from measurement_toolkit.parameters import create_conductance_parameters
                station.conductance_parameters = create_conductance_parameters(station.ohmics)
                station.measure_params = station.conductance_parameters
        elif not silent:
            print(f'Could not find gates file: {config.user.gates_file}')

    # Update plottr database
    if update_plottr:
        try:
            update_plottr_database(database_path=qc.config.core.db_location)
        except Exception:
            warnings.warn(
                'Could not update Plottr, to find out why, run:'
                'update_plottr_database(database_path=qc.config.core.db_location)'
            )

    if show_device and 'device_image' in config.user:
        device_image_filepath = Path(config.user.device_image)
        if device_image_filepath.with_suffix('.png').exists():
            show_image(device_image_filepath.with_suffix('.png'))
        elif device_image_filepath.with_suffix('.pdf').exists():
            show_image(device_image_filepath.with_suffix('.pdf'))

    if configure_device_folder and 'analysis_folder' in config.user:
        from measurement_toolkit.tools.notebook_tools import configure_device_folder
        configure_device_folder(
            root_folder=config.user.analysis_folder,
            silent=silent,
            create_daily_measurement_notebook=True
        )
    
    if populate_namespace:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is not None:
            shell.user_ns['database'] = database
            shell.user_ns['experiment'] = experiment
            if hasattr(station, 'conductance_parameters'):
                shell.user_ns['conductance_parameters'] = station.conductance_parameters
