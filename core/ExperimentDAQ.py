from time import sleep
import serial
import threading
import yaml
from pathlib import Path


# Minimal DAQ surface still used by BaseExperiment.
class ExperimentDAQ:
    def __init__(self, experiment_id, debug):
        self.experiment_id = experiment_id
        self.debug = debug
        self.sampling_rate = 4000
        self.ni_log_filename = None

    def start_everything(self):
        return

    def stop_everything(self):
        return