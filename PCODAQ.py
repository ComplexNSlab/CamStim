from sys import platform
import socket
from time import sleep, time
import numpy as np
import matplotlib.pyplot as plt
import serial 
import threading 
import yaml

# control NI dacs only on windows
# if platform == "win32":
#     import nidaqmx as ni


# TODO consider making a BaseDAQ such that one can use the pco system too
class PCODAQ:
    def __init__(self, experiment_id, DEBUG):
        self.experiment_id = experiment_id
        self.DEBUG = DEBUG

        # # TODO make it a yaml settings
        # self.ip_address_list = ["137.82.137.183"]
        # if platform == "win32":
        #     self.port = 1001
        # else:
        #     self.port = 10001

        # # TODO make into a proper setting????
        # self.sampling_rate = 4000
        # self.sock = socket.socket(socket.AF_INET,
        #                           socket.SOCK_DGRAM)

        # self.ni_log_filename = None
        # self.data = []

        # if platform == "win32" and not self.DEBUG:
        #     self.create_NI_tasks()          


    def __del__(self):
        if platform == "win32" and not self.DEBUG:
            self.ai_log_task.close()
            self.out_ttl_task.close()



    # TODO make the channels into a settings file somehow...
    # def create_NI_tasks(self):
    #     self.ai_log_task = ni.Task()
    #     self.ai_log_task.ai_channels.add_ai_voltage_chan("Dev1/ai0:4")
    #     #self.ai_log_task.ci_channels.add_ci_xx() #TODO
    #     self.ai_log_task.timing.cfg_samp_clk_timing(rate=self.sampling_rate, sample_mode=ni.constants.AcquisitionType.CONTINUOUS, samps_per_chan=5000000)
    #     self.ai_log_task.register_every_n_samples_acquired_into_buffer_event(self.sampling_rate, self.data_read_callback)
		
	
    #     self.out_ttl_task = ni.Task()
    #     self.out_ttl_task.do_channels.add_do_chan("Dev1/port1/line0", line_grouping=ni.constants.LineGrouping.CHAN_FOR_ALL_LINES)
        



    def data_read_callback(self, task_handle, every_n_samples_event_type,
                    number_of_samples, callback_data):
        #print('Every N Samples callback invoked.')

        samples = self.ai_log_task.read(number_of_samples_per_channel=self.sampling_rate)
        self.data.append(samples)

        return 0


    def send_message_to_list(self, message):
        message = message.encode()
        for address in self.ip_address_list:
            print(message, address, self.port)
            self.sock.sendto(message, (address, self.port)) 


    def start_cameras(self):
        message1 = "ExpStart {} 1 1".format(self.experiment_id)
        message2 = "StimStart {} 1 1 1".format(self.experiment_id)

        self.send_message_to_list(message1)
        self.send_message_to_list(message2)


    def stop_cameras(self):
        message1 = "StimEnd {} 1 1 1".format(self.experiment_id)
        message2 = "ExpEnd {} 1 1".format(self.experiment_id)

        self.send_message_to_list(message1)
        self.send_message_to_list(message2)

    def start_everything(self):
        self.start_logging()
        sleep(1)  
        self.start_cameras()
        sleep(1)
        self.start_pco()
        sleep(1)
    
    def stop_everything(self):
        self.stop_pco()
        self.stop_cameras()
        print("Waiting additional 3 seconds before stopping logging.")
        sleep(3)
        self.stop_logging()

        # self.save_log(self.ni_log_filename)
        self.acquisition_running = False

    def start_logging(self):
        self.ai_log_task.start()


    def stop_logging(self):
        self.ai_log_task.stop()
		
	# send a high ttl to trigger 2p acquisition
    def start_pco(self):
        print("TURNED TRUE")
        self.out_ttl_task.write([True])
		
    def stop_pco(self):
        self.out_ttl_task.write([False])


    def save_log(self, filename):
        self.data = np.hstack(self.data).astype(np.float16)
        np.save(filename, self.data)
        print("Saved NI log (size): ", filename, self.data.shape)
        
        

if __name__ == "__main__":
    experiment_id = "test_experiment"
    DEBUG = False

    # Create an instance of the PCODAQ class
    daq = PCODAQ(experiment_id, DEBUG)


    input("Press enter to start the fake experiment.")
    try:
        # Start data acquisition
        daq.start_logging()
        print("Data acquisition started. Running for 10 seconds...")
        sleep(1)

        daq.start_cameras()
        # Wait for 10 seconds
        sleep(10)

        daq.stop_cameras()
        sleep(1)
        # Stop data acquisition
        daq.stop_logging()
        print("Data acquisition stopped.")

        # Save the logged data
        filename = "labjack_data.npy"
        daq.save_log(filename)
        print("Data saved to", filename)

        # Load the saved data
        saved_data = np.load(filename)

        # Plotting the data
        plt.figure(figsize=(10, 8))

        n_channels = saved_data.shape[0]
        time_axis = np.arange(saved_data.shape[1]) / float(daq.sampling_rate)

        for i in range(n_channels):
            plt.subplot(n_channels, 1, i+1)
            plt.plot(time_axis, saved_data[i, :])
            plt.title(f'Channel {i}')
            plt.xlabel('Time (s)')
            plt.ylabel('Voltage (V)')

        plt.tight_layout()
        plt.show()

    finally:
        # Clean up
        del daq



####
class Teensy:
    def __init__(self, experiment_id, DEBUG, teensy_params):
        self.teensy_parameters_filename = teensy_params
        with open (self.teensy_parameters_filename, 'r') as file:
            self.teensy_parameters = yaml.load(file, Loader=yaml.FullLoader)

        # Load n trials and timing lengths
        self.port = self.teensy_parameters['port']
        self.BAUD = self.teensy_parameters['BAUD']

        self.ser = serial.Serial(self.port, self.BAUD, timeout = 1)
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

        self.reader_thread = threading.Thread(target = self.read_teensy, daemon = True)
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
####