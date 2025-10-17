from sys import platform
import socket
from time import sleep, time
import numpy as np

# control National Instruments data aq board only on windows
if platform == "win32":
    import nidaqmx as ni



# TODO consider making a BaseDAQ such that one can use the pco system too
class NISDAQ:
    def __init__(self, experiment_id):
        self.experiment_id = experiment_id

        # TODO make it a yaml settings
        self.ip_address_list = ["172.17.150.226", "172.17.150.220"]
        if platform == "win32":
            self.port = 1001
        else:
            self.port = 10001

        # TODO make into a proper setting????
        self.sampling_rate = 5000
        self.sock = socket.socket(socket.AF_INET,
                                  socket.SOCK_DGRAM)

        self.ni_log_filename = None
        self.data = []

        if platform == "win32":
            self.create_NI_tasks()                        


    def __del__(self):
        if platform == "win32":
            self.ai_log_task.close()
            self.out_ttl_task.close()
            self.in_ttl_task.close()


    # TODO make the channels into a settings file somehow...
    def create_NI_tasks(self):
        self.ai_log_task = ni.Task()
        self.ai_log_task.ai_channels.add_ai_voltage_chan("Dev1/ai0:4")
        #self.ai_log_task.ai_channels.add_ai_voltage_chan("Dev1/ai18")
        #self.ai_log_task.ai_channels.add_ai_voltage_chan("Dev1/ai19")
        #self.ai_log_task.ci_channels.add_ci_xx() #TODO
        self.ai_log_task.timing.cfg_samp_clk_timing(rate=self.sampling_rate, sample_mode=ni.constants.AcquisitionType.CONTINUOUS, samps_per_chan=5000000)
        self.ai_log_task.register_every_n_samples_acquired_into_buffer_event(self.sampling_rate, self.data_read_callback)

        self.out_ttl_task = ni.Task()
        self.out_ttl_task.do_channels.add_do_chan("Dev1/port0/line1", line_grouping=ni.constants.LineGrouping.CHAN_FOR_ALL_LINES)

        self.in_ttl_task = ni.Task()
        self.in_ttl_task.di_channels.add_di_chan("Dev1/port0/line3", line_grouping=ni.constants.LineGrouping.CHAN_FOR_ALL_LINES)
        self.in_ttl_task.di_channels.add_di_chan("Dev1/port1/line0", line_grouping=ni.constants.LineGrouping.CHAN_FOR_ALL_LINES)    


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
        message2 = "BlockStart {} 1 1 1".format(self.experiment_id)

        self.send_message_to_list(message1)
        self.send_message_to_list(message2)


    def stop_cameras(self):
        message1 = "BlockEnd {} 1 1 1".format(self.experiment_id)
        message2 = "ExpEnd {} 1 1".format(self.experiment_id)

        self.send_message_to_list(message1)
        self.send_message_to_list(message2)

    def start_everything(self):
        self.start_logging()
        sleep(1)  
        self.start_2p()
        self.start_cameras()

        if not self.wait_for_2p_aq():
            raise Exception("2-photon microscope did not start acqsuisition...")


    def stop_everything(self):
        self.stop_cameras()
        self.stop_2p()
        #TODO how does 2p tell us it's done capturing.
        print("Waiting additional 3 seconds before stopping logging.")
        sleep(3)
        self.stop_logging()

        self.save_log(self.ni_log_filename)


    # send a high ttl to trigger 2p acquisition
    def start_2p(self):
        self.out_ttl_task.write([True])

        
    def wait_for_2p_aq(self):
        t_start = time()
        print("Waiting for 2p")
        while (time()-t_start < 3):
            ret = self.in_ttl_task.read()[0]
            if not ret:
                print("Response received after: {}".format(time()-t_start))
                return True

        return False

    
    def stop_2p(self):
        self.out_ttl_task.write([False])


    def start_logging(self):
        self.ai_log_task.start()


    def stop_logging(self):
        self.ai_log_task.stop()


    def save_log(self, filename):
        self.data = np.asarray(self.data).astype(np.float16)
        
        np.save(filename, self.data)
        print("Saved NI log (size): ", filename, self.data.shape)
 

if __name__ == "__main__":
    data = []
    data2 = []
    DEBUG = True

    def callback(task_handle, every_n_samples_event_type,
                    number_of_samples, callback_data):
        print('Every N Samples callback invoked.')

        samples = ai_task.read(number_of_samples_per_channel=1000)
        data.append(samples)

        return 0

    def callback2(task_handle, every_n_samples_event_type,
                    number_of_samples, callback_data):
        print('Encoder read')

        samples = c0_task.read(number_of_samples_per_channel=1000)
        data2.append(samples)

        return 0

    ai_task = ni.Task()
    ai_task.ai_channels.add_ai_voltage_chan("Dev1/ai0:7")
    ai_task.timing.cfg_samp_clk_timing(rate=1000, sample_mode=ni.constants.AcquisitionType.CONTINUOUS)
    ai_task.register_every_n_samples_acquired_into_buffer_event(1000, callback)

    c0_task = ni.Task()
    c0_task.ci_channels.add_ci_ang_encoder_chan("Dev1/ctr0",)# units=ni.constants.AngleUnits.TICKS)
    #c0_task.timing.cfg_samp_clk_timing(rate=1000, source="Dev1/di1", sample_mode=ni.constants.AcquisitionType.CONTINUOUS)
    #c0_task.register_every_n_samples_acquired_into_buffer_event(1000, callback2)
   
    # in_ttl_task = ni.Task()
    # in_ttl_task.di_channels.add_di_chan("Dev1/port0/line3", line_grouping=ni.constants.LineGrouping.CHAN_FOR_ALL_LINES)
    # in_ttl_task.di_channels.add_di_chan("Dev1/port1/line0", line_grouping=ni.constants.LineGrouping.CHAN_FOR_ALL_LINES)

    # start analog input acquisition
    
    c0_task.start()
    ai_task.start()

    for i in range(10):
        print(c0_task.read())
        sleep(1)

    ai_task.stop()
    c0_task.stop()
##    data = np.asarray(data)
##    print(data)
##    print(data.shape)

    # stop the 2p aq.
    # out_ttl_task.write([False])
    # sleep(1)
    # data = np.asarray(data)
    # print(data.shape)

    #np.save("data-1-1005.npy", data)

    #out_ttl_task.close()
    #in_ttl_task.close()
    ai_task.close()
    c0_task.close()
    # task = ni.Task()
    # task.ai_channels.add_ai_voltage_chan("Dev0/ai1:7")
    # task.CreateDOChan("/TestDevice/port0/line2","",PyDAQmx.DAQmx_Val_ChanForAllLines)
    # task.StartTask()
    # task.WriteDigitalLines(1,1,10.0,PyDAQmx.DAQmx_Val_GroupByChannel,data,None,None)
    # task.StopTask()
