import os
from pathlib import Path
from time import sleep, time

from core.ExperimentLogger import ExperimentLogger
import yaml

class Continuous:
    # This experiment can run without hardware trigger enabled
    requires_hardware_trigger = False
    # This experiment does not use the Teensy board
    skip_teensy = True
    
    def __init__(
        self,
        experiment_id,
        mouse_id,
        daq,
        monitor_config_filename,
        save_settings_config_filename,
        exp_config_filename,
        debug,
        preview=False,
        save_outputs=None,
    ):
        self.experiment_id = experiment_id
        self.mouse_id = mouse_id
        self.daq = daq
        self.debug = debug
        self.preview = bool(preview)
        self.save_outputs = (not self.preview) if save_outputs is None else bool(save_outputs)

        self.status_callback = None
        self.trial_callback = None
        self.acquisition_running = False
        self.experiment_running = False

        self.monitor_config_filename = self.resolve_config_path(monitor_config_filename)
        self.save_settings_config_filename = self.resolve_config_path(save_settings_config_filename)
        self.exp_parameters_filename = self.resolve_config_path(exp_config_filename)

        self.exp_parameters = None
        self.exp_protocol = 'continuous'
        self.wait_time_sec = 300.0

        self.save_dir = None
        self.data_log_dir = None
        self.experiment_log_filename = None
        self.ni_log_filename = None
        if self.save_outputs:
            self._create_save_directories()

        self.experiment_settings_filenames = [
            self.monitor_config_filename,
            self.save_settings_config_filename,
            self.exp_parameters_filename,
        ]
        self.exp_log = ExperimentLogger(
            self.experiment_log_filename,
            self.experiment_id,
            self.mouse_id,
            self.data_log_dir,
            self.experiment_settings_filenames,
            preview=self.preview,
            save_outputs=self.save_outputs,
        )
        self.daq.ni_log_filename = self.ni_log_filename
        self.exp_log.log['daq_sampling_rate'] = self.daq.sampling_rate

    @staticmethod
    def resolve_config_path(config_filename):
        config_path = Path(config_filename)
        if config_path.is_absolute():
            return str(config_path)

        repo_root = Path(__file__).resolve().parent.parent
        candidate_paths = [
            repo_root / 'config_files' / config_path,
            repo_root / config_path,
        ]
        for candidate in candidate_paths:
            if candidate.is_file():
                return str(candidate)

        return str(config_path)

    def _create_save_directories(self):
        with open(self.save_settings_config_filename, 'r') as file:
            save_settings = yaml.load(file, Loader=yaml.FullLoader)

        save_root = save_settings.get('SAVE_DIR')
        if not save_root:
            raise Exception(f"SAVE_DIR missing in {self.save_settings_config_filename}")

        self.save_dir = os.path.join(save_root, self.mouse_id, self.experiment_id)
        self.data_log_dir = self.save_dir
        self.experiment_log_filename = os.path.join(self.data_log_dir, f"{self.mouse_id}_{self.experiment_id}")
        self.ni_log_filename = os.path.join(self.data_log_dir, f"{self.experiment_id}_ni_log.npy")

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        elif not self.debug:
            raise Exception(f"Experiment ID: {self.experiment_id} already exists make new ID...")

    def set_status_callback(self, callback):
        self.status_callback = callback

    def set_trial_callback(self, callback):
        self.trial_callback = callback

    def update_status(self, message):
        if self.status_callback:
            self.status_callback(message)
        else:
            print(message)

    def load_experiment_config(self):
        with open(self.exp_parameters_filename, 'r') as file:
            self.exp_parameters = yaml.load(file, Loader=yaml.FullLoader)

        self.exp_protocol = self.exp_parameters.get('name', 'continuous')
        self.wait_time_sec = float(self.exp_parameters.get('wait_time_sec', 300))
        self.exp_log.log['exp_parameters'] = self.exp_protocol

    def start_data_acquisition(self):
        if self.daq is None:
            raise Exception('Please set the daq object, it has not been set.')

        if self.preview:
            return

        if os.sys.platform == 'win32' and not self.debug:
            self.acquisition_running = True
            self.daq.start_everything()

    def stop_data_acquisition(self):
        if os.sys.platform == 'win32' and not self.debug:
            self.daq.stop_everything()
            self.acquisition_running = False

    def run_experiment(self):
        self.experiment_running = True

        self.update_status(f"Starting {self.exp_protocol} experiment.")
        start_time = time()
        self.exp_log.log_exp_start(start_time)

        sleep(self.wait_time_sec)

        self.exp_log.log_exp_end(time(), 0)
        self.exp_log.save_log()
        self.update_status('Continuous experiment finished.')
        self.experiment_running = False

    def stop_experiment(self):
        self.experiment_running = False
