from abc import ABC, abstractmethod
from core.ExperimentLogger import ExperimentLogger
import os
import time
import yaml

class SpontaneousActivity(ABC):
	"""
	Simplified spontaneous activity experiment: no psychopy, photodiode, or monitor info.
	Loads wait time and experiment name from config_files/spontaneousActivity.yaml.
	Logs experiment start/end and saves log file.
	"""

	def __init__(self, experiment_id, mouse_id, daq=None, log_dir="./logs", config_path=None, *args, **kwargs):
		self.experiment_id = experiment_id
		self.mouse_id = mouse_id
		self.daq = daq
		self.experiment_running = False
		self.log_dir = log_dir
		os.makedirs(self.log_dir, exist_ok=True)
		self.config_path = config_path or os.path.join(os.path.dirname(__file__), '../config_files/spontaneousActivity.yaml')
		# Load config
		with open(self.config_path, 'r') as f:
			config = yaml.safe_load(f)
		self.experiment_name = config.get('name', 'spontaneousActivity')
		self.wait_time_sec = config.get('wait_time_sec', 300)

		self.experiment_log_filename = os.path.join(self.log_dir, f"{self.experiment_id}_exp_log")
		self.exp_log = ExperimentLogger(self.experiment_log_filename, self.experiment_id, self.mouse_id, self.log_dir, [])
		# Save experiment name as exp_protocol, as in VisualFieldMapping
		self.exp_protocol = self.experiment_name
		self.exp_log.log['exp_parameters'] = self.exp_protocol

	def run_experiment(self):
		self.experiment_running = True
		print(f"Experiment {self.experiment_name} ({self.experiment_id}) for mouse {self.mouse_id} started.")
		self.exp_log.log_exp_start(time.time())
		print(f"Waiting for {self.wait_time_sec} seconds...")
		time.sleep(self.wait_time_sec)
		print("Experiment finished.")
		self.exp_log.log_exp_end(time.time(), 0)
		self.exp_log.save_log()
		self.experiment_running = False
