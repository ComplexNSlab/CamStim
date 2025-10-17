from sys import platform
import socket
from time import sleep, time
import numpy as np
import matplotlib.pyplot as plt
import os

# control NI dacs only on windows
if platform == "win32":
    import nidaqmx as ni
    from nidaqmx.constants import LineGrouping

class BlueDAQ:
    def __init__(self, experiment_id, DEBUG):
        self.experiment_id = experiment_id
        self.DEBUG = DEBUG

        # Network settings (kept from PCODAQ for camera triggering)
        self.ip_address_list = ["137.82.137.183"]
        if platform == "win32":
            self.port = 1001
        else:
            self.port = 10001

        self.sampling_rate = 4000
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Set up log filename
        self.ni_log_filename = os.path.join(os.getcwd(), f"{self.experiment_id}_ni.npy")
        self.data = []

        if platform == "win32" and not self.DEBUG:
            self.create_NI_tasks()

    def __del__(self):
        if platform == "win32" and not self.DEBUG:
            self.ai_log_task.close()
            self.out_ttl_task.close()
            self.di_task.close()

    def create_NI_tasks(self):
        # Analog input task (same as PCODAQ)
        self.ai_log_task = ni.Task()
        self.ai_log_task.ai_channels.add_ai_voltage_chan("Dev1/ai0:4")
        self.ai_log_task.timing.cfg_samp_clk_timing(
            rate=self.sampling_rate,
            sample_mode=ni.constants.AcquisitionType.CONTINUOUS,
            samps_per_chan=5000000
        )
        self.ai_log_task.register_every_n_samples_acquired_into_buffer_event(
            self.sampling_rate,
            self.data_read_callback
        )

        # Digital input task for LED monitoring
        self.di_task = ni.Task()
        self.di_task.di_channels.add_di_chan(
            "Dev1/port0/line0:1",
            line_grouping=LineGrouping.CHAN_PER_LINE
        )

        # Output TTL task (same as PCODAQ for camera triggering)
        self.out_ttl_task = ni.Task()
        self.out_ttl_task.do_channels.add_do_chan("Dev1/port1/line0")

    def data_read_callback(self, task_handle, every_n_samples_event_type,
                          number_of_samples, callback_data):
        """Callback for reading analog data"""
        self.data.append(self.ai_log_task.read(number_of_samples_per_channel=number_of_samples))
        return 0

    def read_digital_inputs(self):
        """Read both digital input channels"""
        if self.DEBUG or platform != "win32":
            return [False, False]  # Debug mode returns dummy values
        return self.di_task.read()

    def read_digital_input(self, channel):
        """Read specific digital input channel (0 for IR, 1 for Blue)"""
        if self.DEBUG or platform != "win32":
            return False  # Debug mode returns dummy value
        values = self.di_task.read()
        return values[channel]

    def start_acquisition(self):
        """Start data acquisition"""
        if platform == "win32" and not self.DEBUG:
            self.ai_log_task.start()
            self.di_task.start()

    def stop_acquisition(self):
        """Stop data acquisition"""
        if platform == "win32" and not self.DEBUG:
            self.ai_log_task.stop()
            self.di_task.stop()

    def save_data(self, filename):
        """Save acquired data"""
        if len(self.data) > 0:
            print(f"\nSaving NI data to {filename}")
            np.save(filename, np.hstack(self.data))
            print("Data saved successfully")

    # Camera triggering methods (kept from PCODAQ)
    def start_cameras(self):
        if platform == "win32" and not self.DEBUG:
            self.out_ttl_task.write(True)
        for ip_address in self.ip_address_list:
            message1 = "ExpStart {} 1 1".format(self.experiment_id)
            message2 = "StimStart {} 1 1 1".format(self.experiment_id)
            
            self.sock.sendto(str.encode(message1), (ip_address, self.port))
            self.sock.sendto(str.encode(message2), (ip_address, self.port))

    def stop_cameras(self):
        if platform == "win32" and not self.DEBUG:
            self.out_ttl_task.write(False)
        for ip_address in self.ip_address_list:
            message1 = "StimEnd {} 1 1 1".format(self.experiment_id)
            message2 = "ExpEnd {} 1 1".format(self.experiment_id)

            
            self.sock.sendto(str.encode(message1), (ip_address, self.port)) 
            self.sock.sendto(str.encode(message2), (ip_address, self.port)) 