from pathlib import Path
import pandas as pd


fridge_logs_path = Path(r'\\QT6CONTROLRACK\Users\QT6_Control_Rack\Documents\Fridge logs')
temperature_labels = {
    'main_temperatures': {'CH1': 'PT_50K', 'CH2': 'PT_4K', 'CH3': 'magnet', 'CH5': 'still', 'CH6': 'mixing_chamber'},
    'probe_temperatures': {'CH1': 'probe'}
}


def get_fridge_date_folders(path):
    """Sorted by latest"""
    folders = [folder for folder in path.iterdir() if folder.name.startswith('2')]
    sorted_folders = sorted(
        folders,
        key=lambda folder: int(folder.name[:2] + folder.name[3:5] + folder.name[6:8]),
        reverse=True
    )

    return sorted_folders


def extract_valve_control_data():
    pass

from datetime import datetime
dateparse = lambda x, y: datetime.strptime(f'{x} {y}', '%d-%m-%y %H:%M:%S')
def extract_temperature_data(temperature_date_folders):
    """Extract temperatures from all channels from a temperature folder"""
    data = {}

    if isinstance(temperature_date_folders, Path):
        temperature_date_folders = [temperature_date_folders]

    for temperature_date_folder in temperature_date_folders:
        for file in temperature_date_folder.iterdir():
            if 'RESISTANCE' in file.name:
                continue

            channel = file.name[:3]
            df = pd.read_csv(
                file,
                names=['Date', 'Time', 'Temperature'],
                index_col=[0],
                parse_dates={"Datetime": [0, 1]},
                # date_parser=pd.to_datetime()
            )

            # Determine label
            if '192.168.23.103' in str(temperature_date_folder):
                label = temperature_labels['main_temperatures'][channel]
            elif '192.168.23.104' in str(temperature_date_folder):
                label = temperature_labels['probe_temperatures'][channel]
            else:
                label = channel

            if label not in data:
                # Add new data frame
                data[label] = df
            else:
                # Merge with existing dataframe
                data[label] = pd.concat([data[label], df])

    return data


def get_latest_temperatures(silent=True):
    folders = {
        'main_temperatures': get_fridge_date_folders(fridge_logs_path / 'log-data' / '192.168.23.103')[0],
        'probe_temperatures': get_fridge_date_folders(fridge_logs_path / 'log-data' / '192.168.23.104')[0]
    }
    temperatures = {}
    for name, folder in folders.items():
        for file in folder.iterdir():
            if 'RESISTANCE' in file.name:
                continue

            channel = file.name[:3]
            temperature_label = temperature_labels[name][channel]

            file_text = file.read_text()
            final_line = file_text.rstrip('\n').split('\n')[-1]
            date_str, time_str, temperature_str = final_line.split(',')

            if not silent:
                print(date_str, time_str)
            temperature = float(temperature_str)

            temperatures[temperature_label] = float(temperature)

    return temperatures


def get_fridge_data(days=2):
    data = {}

    folders = {
        'valve_control': get_fridge_date_folders(fridge_logs_path)[0],
    }

    temperature_folders = {
        'main_temperatures': fridge_logs_path / 'log-data' / '192.168.23.103',
        'probe_temperatures': fridge_logs_path / 'log-data' / '192.168.23.104'
    }

    files = {
        'pressures': folders['valve_control'] / f'maxigauge {folders["valve_control"].name}.log',
        'flow': folders['valve_control'] / f'Flowmeter {folders["valve_control"].name}.log',
        'status': folders['valve_control'] / f'Status_{folders["valve_control"].name}.log',
    }

    # Extract temperatures
    data['temperatures'] = {}
    for label, temperature_folder in temperature_folders.items():
        temperature_date_folders = get_fridge_date_folders(temperature_folder)[:days]
        print('Folders: ', temperature_date_folders)
        temperatures = extract_temperature_data(temperature_date_folders)
        data['temperatures'].update(**temperatures)

    return data
