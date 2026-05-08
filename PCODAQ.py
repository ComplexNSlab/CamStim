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


class Teensy:
    def __init__(self, experiment_id, DEBUG, teensy_params):
        config_path = Path(teensy_params)
        if not config_path.is_absolute():
            config_path = Path(__file__).resolve().parent / config_path
        self.teensy_parameters_filename = str(config_path)
        with open(self.teensy_parameters_filename, 'r') as file:
            self.teensy_parameters = yaml.load(file, Loader=yaml.FullLoader)

        self.port = self.teensy_parameters['port']
        self.BAUD = self.teensy_parameters['BAUD']

        self.ser = serial.Serial(self.port, self.BAUD, timeout=1)
        sleep(2)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        print("Serial connected.")

        self.F_LED = self.teensy_parameters['F_LED']
        self.E_LED = self.teensy_parameters['E_LED']
        self.O_CAM = self.teensy_parameters['O_CAM']
        self.cam_start_mode = self.teensy_parameters['cam_start_mode']
        self.ttls_start_mode = self.teensy_parameters['ttls_start_mode']
        self.dual_mode = self.teensy_parameters['dual_mode']
        self.ttl_width = self.teensy_parameters['ttl_width']

        self.reader_thread = threading.Thread(target=self.read_teensy, daemon=True)
        self.reader_thread.start()
        self.stop_teensy()
        sleep(2)

    def read_teensy(self):
        while True:
            try:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode().strip()
                    if line:
                        print("Teensy:", line)
            except Exception as e:
                print("Teensy read error:", repr(e))
                break

    def start_teensy(self):
        command = f"S {self.F_LED} {self.E_LED} {self.O_CAM} {self.cam_start_mode} {self.ttls_start_mode} {self.dual_mode} {self.ttl_width}\n"
        self.ser.write(command.encode())
        print(f"Sent start command: {command.strip()}")

    def stop_teensy(self):
        self.ser.write(b"Q\n")
        print("Sent stop command: Q")
        sleep(2)