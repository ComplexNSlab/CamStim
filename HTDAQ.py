import socket
import u3
import numpy as np
from time import sleep
import threading
import platform
import matplotlib.pyplot as plt

class HTDAQ:
    def __init__(self, experiment_id, DEBUG):
        self.experiment_id = experiment_id
        self.DEBUG = DEBUG
        self.ip_address_list = ["137.82.137.183"]
        self.port = 1001 if platform == "win32" else 1001
        self.sampling_rate = 5000
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.data = []
        self.acquisition_thread = None
        self.running = False

        if not self.DEBUG:
            self.create_LabJack_tasks()

    def __del__(self):
        if not self.DEBUG:
            self.u3_device.close()

    def create_LabJack_tasks(self):
        self.u3_device = u3.U3()
        self.u3_device.configU3()
        self.u3_device.getCalibrationData()
        self.u3_device.configIO(FIOAnalog=0x0F)
        self.u3_device.streamConfig(NumChannels=4, PChannels=[0, 1, 2, 3], NChannels=[31, 31, 31, 31], Resolution=3, ScanFrequency=self.sampling_rate)

    def data_read_callback(self):
        for r in self.u3_device.streamData():
            if r is not None:
                samples = [r['AIN%d' % i] for i in range(4)]
                self.data.append(samples)
                if not self.running:
                    break
            else:
                break

    def start_logging(self):
        self.running = True
        self.u3_device.streamStart()
        self.acquisition_thread = threading.Thread(target=self.data_read_callback)
        self.acquisition_thread.start()

    def stop_logging(self):
        try:
            self.running = False
            if self.acquisition_thread and self.acquisition_thread.is_alive():
                self.acquisition_thread.join()
            self.u3_device.streamStop()
        except IndexError as e:
            print(f"Error stopping stream: {e}")

        

    def save_log(self, filename):
        # Convert list of samples to a 2D numpy array (n_samples, n_channels)
        self.data = np.hstack(self.data).astype(np.float16)
        np.save(filename, self.data)
        print("Saved LabJack log (size): ", filename, self.data.shape)


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
    
    def stop_everything(self):
        self.stop_cameras()
        print("Waiting additional 3 seconds before stopping logging.")
        sleep(3)
        self.stop_logging()

        self.save_log(self.ni_log_filename)
        self.acquisition_running = False

    # Other methods (send_message_to_list, start_cameras, stop_cameras, etc.) remain the same

# Rest of your class implementation remains unchanged
if __name__ == "__main__":
    experiment_id = "test_experiment"
    DEBUG = False

    # Create an instance of the PCODAQ class
    daq = HTDAQ(experiment_id, DEBUG)
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
