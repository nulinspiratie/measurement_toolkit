from pathlib import Path
import pandas as pd
from time import sleep, perf_counter


__all__ = [
    'get_fridge_date_folders', 
    'extract_temperature_data', 
    'get_latest_temperatures', 
    'get_fridge_data', 
    'HeaterTemperatureController',
    'wait_for_set_temperature'
]

fridge_logs_path = Path(r'\\QT6CONTROLRACK\Users\QT6_Control_Rack\QDev Dropbox\qdev\BF1\Fridge logs')
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


class HeaterTemperatureController():
    max_temperature = 0.7
    max_power = 2e-3

    def __init__(self, fridge_url='http://192.168.23.103/#/', chrome_filepath=r'C:\Program Files\chromedriver.exe'):
        from selenium import webdriver
        self.driver = webdriver.Chrome(r'C:\Program Files\chromedriver.exe')
        self.driver.get(fridge_url)
        print('Please navigate to the heater webpage by clicking on it in Chrome')
        
    def set_heater_temperature(self, temperature, execute=True):
        from selenium.webdriver.common.by import By
        from selenium.webdriver.common.keys import Keys
        from selenium.webdriver.common.action_chains import ActionChains

        assert temperature < self.max_temperature

        temperature = int(temperature * 1e3)

        # Clear any windows
        actions = ActionChains(self.driver)
        actions.send_keys(Keys.ESCAPE)
        actions.perform()
        sleep(0.2)

        # Get PID panel
        PID_panel = self.driver.find_element(By.CLASS_NAME, 'g_view_content_part_grey_addon')

        # Click on temperature
        set_temperature = PID_panel.find_elements(By.CLASS_NAME, 'g_setting_data_style')[0]
        set_temperature.click()
        sleep(1)

        # Ensure we have mK scale
        scale_window = self.driver.find_element(By.CLASS_NAME, 'xz2')
        elems = scale_window.find_elements(By.CLASS_NAME, 'ng-star-inserted')
        uK_scale = any(elem.get_attribute("innerHTML") == 'm' for elem in elems)
        assert uK_scale, "scale is not mK"

        # Clear old values
        for k in range(3):
            # Set temperature
            actions = ActionChains(self.driver)
            actions.send_keys(*[Keys.DELETE]*5)
            actions.send_keys(f'{temperature:.0f}')
            actions.perform()

            # Verify temperature
            temperature_group_elem = self.driver.find_element(By.CLASS_NAME, 'zinput_group')
            temperature_input_elem = temperature_group_elem.find_element(By.CLASS_NAME, 'xz1')
            temperature_str = temperature_input_elem.text
            if temperature_str == str(temperature):
                break
            else:
                print(f'Temperature "{temperature_str}" not equal to {temperature}')
        else:
            raise RuntimeError('Could not set temperature')
            
        if execute:
            actions = ActionChains(self.driver)
            actions.send_keys(Keys.ENTER)
            actions.perform()   

    def set_heater_power(self, power, execute=True):
        from selenium.webdriver.common.by import By
        from selenium.webdriver.common.keys import Keys
        from selenium.webdriver.common.action_chains import ActionChains

        assert power < self.max_power

        power_uW = int(power * 1e6)

        # Clear any windows
        actions = ActionChains(self.driver)
        actions.send_keys(Keys.ESCAPE)
        actions.perform()
        sleep(0.2)

        # Get PID panel
        PID_panel = self.driver.find_element(By.CLASS_NAME, 'g_view_content_part')

        # Click on power
        set_power = PID_panel.find_elements(By.CLASS_NAME, 'g_setting_data_style')[3]
        set_power.click()
        sleep(1)

        # Ensure we have uW scale
        scale_window = self.driver.find_element(By.CLASS_NAME, 'xz2')
        elems = scale_window.find_elements(By.CLASS_NAME, 'ng-star-inserted')
        mK_scale = any(elem.get_attribute("innerHTML") == 'µ' for elem in elems)
        assert mK_scale, "scale is not µW"
        # Clear old values
        for k in range(3):
            # Set power
            actions = ActionChains(self.driver)
            actions.send_keys(*[Keys.DELETE]*5)
            actions.send_keys(f'{power_uW:.0f}')
            actions.perform()

            # Verify power
            power_group_elem = self.driver.find_element(By.CLASS_NAME, 'zinput_group')
            power_input_elem = power_group_elem.find_element(By.CLASS_NAME, 'xz1')
            power_str = power_input_elem.text
            if power_str == str(power_uW):
                break
            else:
                print(f'Temperature "{power_str}" not equal to {power}')
        else:
            raise RuntimeError('Could not set power')
            
        if execute:
            actions = ActionChains(self.driver)
            actions.send_keys(Keys.ENTER)
            actions.perform() 

def wait_for_set_temperature(
    target_temperature, 
    max_wait, 
    label='mixing_chamber', 
    interval=5,
    temperature_uncertainty=2e-3,
    successive_points=3,
    silent=True
    ):
    t0 = perf_counter()

    successes = 0
    while perf_counter() - t0 < max_wait:
        temperatures = get_latest_temperatures()
        temperature = temperatures[label]
        if not silent:
            print(f'{temperature=}')

        if abs(temperature - target_temperature) < temperature_uncertainty:
            successes += 1
        else:
            successes = 0

        if successes == successive_points:
            return True

        sleep(interval)