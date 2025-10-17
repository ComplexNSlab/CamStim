#coding=utf-8
import cv2
import numpy as np
import mvsdk
import time
import platform
import queue
import threading
import serial
import socket
import select
import yaml
from pathlib import Path
import os
from roi_module import ROIDrawer, ROIPlotter
###
import matplotlib.pyplot as plt
###

CONFIG_FILE = 'config.yml'

def load_camera_config(yaml_file_path):
    if Path(yaml_file_path).is_file():
        with open(yaml_file_path, 'r') as file:
            try:
                config = yaml.safe_load(file)
                return config
            except yaml.YAMLError:
                raise Exception("There is an error in the config yaml file, please check it. {}".format(yaml_file_path))
    else:
        raise Exception("Configuration file does not exist, please create it.")



class App(object):
    def __init__(self, config):
        super(App, self).__init__()

        self.config = config
        self.vmin = 0
        self.vmax = 30    
        self.pFrameBuffer = 0
        self.minI = 0
        self.maxI = 255
        self.autoI = 0.05
        self.quit = False
        self.acquiring = False
        self.saving = False
        self.normalizeImage = False
        self.experiment_mode = False
        self.experiment_ongoing = False
        self.removeBackground = False
        self.frame_queue = queue.Queue()  # Buffer for frames
        self.display_queue = queue.Queue()  # Queue for displaying frames
        self.save_thread = threading.Thread(target=self.save_frames)  # Thread for saving frames
        self.display_thread = threading.Thread(target=self.display_frames)  # Create display thread
        self.frame_count = 0  # To keep track of saved 
        self.frames_written = 0
        self.live_speck = config['USE_LIVE_SPECKLE']
        self.exposure = config['EXPOSURE_TIME'] # in ms
        self.analog_gain = config['ANALOG_GAIN'] 
        self.save_dir = config['SAVE_DIR']
        self.filename = config['EXPERIMENT']
        self.bin_exp = config['BIN_EXP_LIVE']
        self.bin_size = config['BIN_SIZE']
        self.force_framerate = config['FORCE_FRAMERATE']
        self.special_framerate = config['SPECIAL_FRAMERATE']
        self.special_frame_period = 1.0/self.special_framerate
        self.last_timestamp = None
        self.zeros = np.zeros((255, 255), dtype=np.uint8) # debug image in case I have problems with camera
        self.frame_timestamps = []  # List to store timestamps
        self.sys_clock_timestamps = []
        self.USE_MONO16 = False # If false defaults to 8 bit although current camera doesn't have true 16 bit
        self.t_start = None
        self.t_end = None
        self.hCamera = None
        self.n_saturated_pixels = 0

        self.roi_drawer = ROIDrawer()
        self.roi_plotter = ROIPlotter()
        self.plot_roi = False

        ###
        self.histogram_open = False
        self.dFoF_open = False
        self.F0 = None
        ###

        self.check_and_fix_existing_experiment()

        # To control the RPi pico that triggers the camera
        if config['PICO_SERIAL_PORT'] is not None:
            self.ser = serial.Serial(config['PICO_SERIAL_PORT'], 115200, timeout=1)
            self.pwm_duty = config['PICO_PWM_DUTY']
            self.pwm_freq = config['PICO_PWM_FREQUENCY']
        else:
            self.pwm_duty = None
            self.pwm_freq = None

        # UDP socket to listen for the commands
        self.udp_port = config['UDP_TRIGGER_PORT']
        self.ip = '0.0.0.0'
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.udp_port))

        self.udp_thread = threading.Thread(target=self.wait_udp_trigger)
        self.udp_thread.start()

        self.dtype = 'uint16' if self.USE_MONO16 else 'uint8'

    def setup_live_speckle_variables(self):
        self.buffer_size = self.config['BUFFER_SIZE']
        self.circular_buffer = np.zeros((self.buffer_size, self.height//self.bin_size, self.width//self.bin_size), dtype=np.float32)
        self.mean_image = np.zeros((self.height//self.bin_size, self.width//self.bin_size), dtype=np.float32)
        self.backgroundImg = np.zeros((self.height//self.bin_size, self.width//self.bin_size), dtype=np.float32)

        self.current_buffer_item = 0
        self.enable_live_speckle = False
        self.buffer_loop_reached = False

    def check_and_fix_existing_experiment(self):
        if os.path.exists(os.path.join(self.save_dir, self.filename+'.bin')):
            print("WARNING!!! Experiment file already exists, modifying name to avoid overwriting.")
            self.filename += "1"
            

    def bin_frame(self, frame):
        # Binning
        binned_frame = frame.reshape((self.height//self.bin_size, self.bin_size, self.width//self.bin_size, self.bin_size)).sum(axis=(1, 3), dtype=np.uint16)

        return binned_frame

    def std_filter_frame(self, frame):
        # Binning
        binned_frame = frame.reshape((self.height//self.bin_size, self.bin_size, self.width//self.bin_size, self.bin_size)).std(axis=(1, 3), dtype=np.float32)

        return binned_frame

    def save_frames(self):
        with open(os.path.join(self.save_dir, self.filename+'.bin'), 'ab') as f:  # Open a binary file for appending
            while True:
                if self.saving or not self.frame_queue.empty():
                    # Process frames if available
                    if not self.frame_queue.empty():
                        frame_data, count, timestamp, sys_stamp = self.frame_queue.get()
                        self.frame_timestamps.append(timestamp)
                        self.sys_clock_timestamps.append(sys_stamp)

                        ###
                        frame = np.frombuffer(frame_data, dtype=self.dtype).reshape((self.height, self.width))
                        frame = cv2.flip(frame, 1)
                        ###
                        
                        if self.bin_exp:
                            frame = self.bin_frame(frame)

                        frame.tofile(f)
                        f.flush()

                        self.frames_written += 1
                    # Check if it's time to exit: quit is True and no frames left in the queue
                    elif self.quit and self.frame_queue.empty():
                        break
                    else:
                        # Optionally, sleep for a very short time to prevent high CPU usage
                        time.sleep(0.005)
                else:
                    if self.quit:
                        break
                    time.sleep(0.005)
        
        # After processing all frames, save metadata
        metadata = {
            'num_frames': self.frames_written,
            'frame_width': self.width if not self.bin_exp else self.width//self.bin_size,
            'frame_height': self.height if not self.bin_exp else self.height//self.bin_size,
            'data_type': self.dtype  if not self.bin_exp else 'uint16',
            'frame_timestamps': self.frame_timestamps,
            'sys_clock_timestamps': self.sys_clock_timestamps,
            'frame_exposure': self.exposure,
            'frame_gain': self.analog_gain,
            'pwm_frequency': self.pwm_freq,
            'pwm_duty': self.pwm_duty,
            'binned_live': self.bin_exp,
            'bin_size': self.bin_size,
            'force_framerate': self.force_framerate,
            'special_framerate': self.special_framerate
        }
        np.save(os.path.join(self.save_dir, '{}_metadata.npy'.format(self.filename)), metadata)

    def display_frames(self):
        cv2.namedWindow("Live View")
        
        cv2.setMouseCallback("Live View", self.roi_drawer.handle_mouse_events)

        while not self.quit or not self.display_queue.empty():
            if not self.display_queue.empty():
                frame_data = self.display_queue.get()

                 # Display frame
                frame = np.frombuffer(frame_data, dtype=self.dtype)
                frame = frame.reshape((self.height, self.width))

                ###
                frame = cv2.flip(frame, 1)
                ###

                self.n_saturated_pixels = (frame.flatten() == 255).sum()
                
                if self.live_speck and self.enable_live_speckle:
                    frame = self.std_filter_frame(frame)
                    self.circular_buffer[self.current_buffer_item, :, :] = frame
                    self.current_buffer_item += 1
                    self.current_buffer_item %= self.buffer_size

                    frame = self.circular_buffer.mean(axis=0)

                    # Clip the values in the image to the desired range
                    clipped = np.clip(frame, self.vmin, self.vmax)
                    # Scale the values to the full 0-255 range
                    scaled = ((clipped - self.vmin) / (self.vmax - self.vmin)) * 255
                    frame = scaled.astype(np.uint8)

                    #frame = 255-frame
            
                if self.removeBackground:
                    frame = frame - self.backgroundImg
                    frame = np.clip(frame, 0, 255)
                if self.normalizeImage:
                    clipped = np.clip(frame, self.minI, self.maxI)
                    scaled = ((clipped - self.minI) / (self.maxI - self.minI)) * 255
                    frame = scaled.astype(np.uint8)
###
                if self.dFoF_open and self.F0 is not None:
                    dfof = (frame.astype(np.float32) - self.F0) / self.F0
                    dfof = np.nan_to_num(dfof, nan=0.0)
                    fmin, fmax = dfof.min(), dfof.max()
                    if fmax > fmin:
                        frame = ((dfof - fmin) / (fmax - fmin) * 255).astype(np.uint8)
                    else:
                        frame = np.zeros_like(dfof, dtype=np.uint8)
###

                frame  = cv2.resize(frame, (self.width//2,self.height//2), interpolation = None)
              #  frame = frame.T
                
                frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
                frame = self.roi_drawer.draw_rectangle(frame)    
                if self.plot_roi and self.roi_drawer.top_left_pt != (-1, -1) and self.roi_drawer.bottom_right_pt != (-1, -1):
                    average_intensity = frame[self.roi_drawer.top_left_pt[1]:self.roi_drawer.bottom_right_pt[1], self.roi_drawer.top_left_pt[0]:self.roi_drawer.bottom_right_pt[0]].mean()
                    self.roi_plotter.update_plot(average_intensity)
                    
                if self.plot_roi or self.live_speck:
                    self.display_queue.queue.clear()

                cv2.imshow("Live View", frame)


            pressed_key = cv2.waitKey(1) & 0xFF
            if pressed_key == 255:
                continue
            elif pressed_key == ord('q'):
                self.roi_plotter.deinitialize_plot()
                self.plot_roi = False
                self.quit = True
                self.t_end = time.time()


            ###
            elif pressed_key == ord('a'):
                if self.histogram_open:
                    plt.close('Histogram')
                    # for num in plt.get_fignums():
                    #     fig = plt.figure(num)
                    #     if fig.get_label() == 'Histogram':
                    #         plt.close(fig)
                    #        break
                    self.histogram_open = False
                else:
                    if not self.display_queue.empty():
                        frame_data = self.display_queue.get()
                        frame = np.frombuffer(frame_data, dtype=self.dtype)
                        frame = frame.reshape((self.height, self.width))
                        frame_8bit = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                        fig = plt.figure('Histogram')
                        plt.clf()
                        plt.hist(frame_8bit.ravel(), bins=256, range=(0,256), color='black')
                        plt.title('Histogram for Frame {}'.format(self.frame_count))
                        plt.xlabel('Pixel Intensity (0-255)')
                        plt.ylabel('Number of Pixels')
                        plt.grid(True)
                        plt.tight_layout()
                        fig.canvas.draw()
                        plt.pause(0.001)
                        #fig.show()
                        self.histogram_open = True

            elif pressed_key == ord('y'):
                if self.dFoF_open:
                    self.dFoF_open = False
                    print("\n Stopping live dFoF view.")
                else:
                    print("\n Starting live dFoF view.")
                    baseline_frames = []
                    F0_n_frames = 50
                    try:
                        for i in range(F0_n_frames):
                            frame_data = self.display_queue.get(timeout=1)
                            frame = np.frombuffer(frame_data, dtype=self.dtype)
                            frame = frame.reshape((self.height, self.width))
                            baseline_frames.append(frame.astype(np.float32))
                    except queue.Empty:
                        print("Warning: Not enough frames in queue to compute baseline F0.")
                        self.dFoF_open = False
                    else:
                        if len(baseline_frames) < F0_n_frames:
                            print("Insufficient baseline frames, cannot start dFoF.")
                            self.dFoF_open = False
                        else:
                            sampled_stack = np.stack(baseline_frames, axis=0) 
                            self.F0 = np.percentile(sampled_stack, 10, axis=0)
                            self.F0[self.F0 == 0] = np.nan 
                            del sampled_stack
                            self.dFoF_open = True
            ###


            elif pressed_key == ord('t'):
                # Switch camera mode to hardware trigger capture
                print("\n Switching to hardware trigger mode.")
                mvsdk.CameraSetTriggerMode(self.hCamera, 2)
            elif pressed_key == ord('c'):
                # Switch camera mode to continuous capture
                print("\n Switching to continuous capture.")
                mvsdk.CameraSetTriggerMode(self.hCamera, 0)
            elif pressed_key == ord('s'):
                if not self.saving:
                    print("\n Switching to continuous capture.")
                    mvsdk.CameraSetTriggerMode(self.hCamera, 0)
                    time.sleep(1) 
                    print("\n Switching to hardware trigger mode.")
                    mvsdk.CameraSetTriggerMode(self.hCamera, 2)
                    self.saving = True
                else:
                    self.saving = False
                    time.sleep(1)
                    print("\n Switching to continuous mode.")
                    mvsdk.CameraSetTriggerMode(self.hCamera, 0)
            elif pressed_key == ord('e'):
                try:
                    self.exposure = float(input("\nEnter new exposure (current: {}ms): \n".format(self.exposure)))
                    mvsdk.CameraSetExposureTime(self.hCamera, self.exposure*1000) 
                except ValueError:
                    print("\n Failed to set exposure, invalid value.")
                
                #print("TrigCap: ", mvsdk.CameraGetExtTrigCapability(self.hCamera))
                #print("Trigger delay time: ", mvsdk.CameraGetExtTrigDelayTime(self.hCamera))
            elif pressed_key == ord('g'):
                try:
                    self.analog_gain = int(input("\nEnter new gain (current: {}): \n".format(self.analog_gain)))
                    mvsdk.CameraSetAnalogGain(self.hCamera, self.analog_gain) 
                except ValueError:
                    print("\n Failed to set gain, invalid value.")
            elif pressed_key == ord('r') and self.roi_drawer.active_roi:
                self.plot_roi = True if not self.plot_roi else False
                if not self.plot_roi:
                    self.roi_drawer.top_left_pt = (-1, -1)
                    self.roi_drawer.bottom_right_pt = (-1, -1)
                    self.roi_drawer.active_roi = False
                    self.roi_plotter.deinitialize_plot()
            elif pressed_key == ord('p'):
                self.print_camera_stats()
            elif pressed_key == ord('d'):
                if self.removeBackground:
                    self.removeBackground = False
                    print("\n Stopping background substraction.")
                else:
                    self.removeBackground = True
                    frame_data = self.display_queue.get()
                    frame = np.frombuffer(frame_data, dtype=self.dtype)
                    frame = frame.reshape((self.height, self.width))                     
                    kernel = np.ones((10,10),np.float32)/100
                    frame = cv2.filter2D(frame,-1,kernel)
                    self.backgroundImg = frame
                    print("\n Substracting background image.")
            elif pressed_key == ord('z'):
                #if self.normalizeImage:
                #    self.normalizeImage = False
                #    print("\n Stopping image normalization.")
                #else:
                self.normalizeImage = True
                self.autoI = self.autoI*2
                if self.autoI > 49:
                    self.autoI = 0.05
                frame_data = self.display_queue.get()
                frame = np.frombuffer(frame_data, dtype=self.dtype)
                self.minI = np.percentile(frame[:], self.autoI)
                self.maxI = np.percentile(frame[:], 100-self.autoI)
                print("\n Changing dynamical range to: {}, {} pixel values. Percentiles: {}, {}".format(self.minI, self.maxI, self.autoI, 100-self.autoI))
            elif pressed_key == ord('h'):
                self.print_keyboard_commands()
            elif pressed_key == ord('m'):
                # initiate PWM signal of raspberry pi pico (not used for speckle)
                self.send_command_and_wait_for_response('init_pwm()')
                self.send_command_and_wait_for_response('start_pwm({}, {})'.format(self.pwm_freq, self.pwm_duty))
            elif pressed_key == ord('n'):
                # stop the PWM signal from the pico (not used for speckle)
                self.send_command_and_wait_for_response('stop_pwm()')
            elif pressed_key == ord('x'):
                if not self.experiment_mode:
                    # set camera in experiment mode
                    self.send_command_and_wait_for_response('init_pwm()')
                    # Switch camera mode to hardware trigger capture
                    mvsdk.CameraSetTriggerMode(self.hCamera, 2)
                    self.experiment_mode = True
                else:
                    mvsdk.CameraSetTriggerMode(self.hCamera, 0)
                    self.send_command_and_wait_for_response('stop_pwm()')
                    self.experiment_mode = False
            elif self.live_speck and pressed_key == ord('b'):
                self.enable_live_speckle = True if not self.enable_live_speckle else False
                if self.enable_live_speckle:
                    print("Enabling live speckle imaging, please wait a few seconds for the buffer to fill up.")
                else:
                    print("Disabled live speckle imaging.")

        print("Quit order received for display thread.")
        self.display_queue.queue.clear()
        

    def wait_udp_trigger(self):
        self.sock.setblocking(0)
        while not self.quit:
            # Receive message
            ready = select.select([self.sock], [], [], 1)
            if ready[0]:
                data, addr = self.sock.recvfrom(1024)  # buffer size is 1024 bytes

                msg = data.decode()

                if self.experiment_mode:
                    if 'ExpStart' in msg:
                        #self.send_command_and_wait_for_response('init_pwm()')
                        self.send_command_and_wait_for_response('start_pwm({}, {})'.format(self.pwm_freq, self.pwm_duty))
                        self.experiment_ongoing = True
                        self.frame_count = 0
                        self.t_start = time.time()
                    elif 'ExpEnd' in msg:
                        if self.experiment_ongoing:
                            self.send_command_and_wait_for_response('stop_pwm()')
                            self.experiment_ongoing = False
                        else:
                            print("Warning, ExpEnd received when no experiment was ongoing. Check things pls")
                    else:
                        print("Received: ", msg)
                else:
                    print("Not in experiment mode, received: {}".format(msg))


    def print_camera_stats(self):
        print("\nExposure(ms): {} Gain: {} ".format(self.exposure, self.analog_gain))

    # function to display keyboard commands help 
    def print_keyboard_commands(self):
        print("\n\nUse the keyboard commands listed below to navigate and change settings within the program. To use a command, first click on the display window before entering the key. If additional follow up inputs are needed (eg exposure numbers, gain numbers), click back to the terminal window before entering.\n\n" +\
            "q -- quit\n" +\
            "t -- switch camera mode to hardware trigger capture\n" +\
            "c -- switch camera mode to continuous capture\n" +\
            "s -- if in continuous capture, switches to hardware trigger mode (ready to save frames); if in hardware trigger mode, switches to continuous capture (save frames off)\n" +\
            "e -- edit exposure\n" +\
            "g -- edit gain\n" +\
            "r -- draw ROI\n" +\
            "a -- display pixel intensity histogram for current frame\n" +\
            "p -- print camera stats\n" +\
            "d -- subtract backgroud image\n" +\
            "z -- normaliZe image dynamic range\n" +\
            "m -- start pwm\n" +\
            "b -- speckle mode on/off\n" +\
            "r -- ROI selection\n" +\
            "m -- start pwm\n" +\
            "n -- stop pwm\n" +\
            "x -- start experiment mode. Press again to stop experiment mode.\n\n" +\

            "To display these commands again, press h\n")


    # Function to send a command and wait for response
    def send_command_and_wait_for_response(self, command):
        self.ser.write((command + '\r\n').encode())  # Send command with newline
        time.sleep(0.05)  # Wait for the response
        response = self.ser.read(self.ser.in_waiting).decode()
        print(response)
    

    def main(self):
        # Enumerate cameras
        DevList = mvsdk.CameraEnumerateDevice()
        nDev = len(DevList)
        if nDev < 1:
            print("No camera was found!")
            return

        for i, DevInfo in enumerate(DevList):
            print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
        i = 0 if nDev == 1 else int(input("Select camera: "))
        DevInfo = DevList[i]
        #print(DevInfo)
        self.print_keyboard_commands()

        # Open camera
        self.hCamera = 0
        try:
            self.hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
        except mvsdk.CameraException as e:
            print("CameraInit Failed({}): {}".format(e.error_code, e.message) )
            return

        # Get camera capability description
        cap = mvsdk.CameraGetCapability(self.hCamera)

        # Determine if it is a monochrome camera or a color camera
        monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

        # For monochrome cameras, let ISP output MONO data directly, instead of expanding it into 24-bit grayscale with R=G=B
        if monoCamera:
            if self.USE_MONO16:
                print("Using 16-bit.")
                mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO16)
            else:
                mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
        else:
            mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

        # Switch camera mode to continuous capture
        mvsdk.CameraSetTriggerMode(self.hCamera, 0)

        # Switch the camera to full speed transmission
        mvsdk.CameraSetFrameSpeed(self.hCamera, 1)

        # Manual exposure, exposure time 
        mvsdk.CameraSetAeState(self.hCamera, 0)
        mvsdk.CameraSetExposureTime(self.hCamera, self.exposure * 1000)
        mvsdk.CameraSetAnalogGain(self.hCamera, self.analog_gain)

        self.height = cap.sResolutionRange.iHeightMax
        self.width = cap.sResolutionRange.iWidthMax
        print(self.height, self.width, "##############")
        if self.live_speck:
            self.setup_live_speckle_variables()

        # Let the SDK's internal image capture thread start working
        mvsdk.CameraPlay(self.hCamera)

        # Calculate the size of the RGB buffer required, here directly allocated according to the camera's maximum resolution
        FrameBufferSize = 1*cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)

        # Allocate RGB buffer for storing images output by ISP
        # Note: The data transferred from the camera to the PC is RAW data, which is converted to RGB data by software ISP on the PC 
        # #(If it is a monochrome camera, no format conversion is needed, but ISP has other processing, so this buffer also needs to be allocated)
        self.pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

        # Set the capture callback function
        self.quit = False
        mvsdk.CameraSetCallbackFunction(self.hCamera, self.GrabCallback, 0)
        self.print_camera_stats()
        time.sleep(1)

        # main loop to print info from the camera
        while not self.quit:
            current_time = time.time()
            elapsed_time = current_time - self.t_start
            average_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0

            # Print stats on the same line
            print("\rSave Queue: {}, Frames Saved: {}, Display Queue: {}, Frames Displayed: {}, Average FPS: {:.2f} Saturated Pixels: {:06d}".format(
                  self.frame_queue.qsize(), self.frames_written, self.display_queue.qsize(), self.frame_count, average_fps, self.n_saturated_pixels), end='')

            time.sleep(0.1)

        print("\n")  # Ensure to move to a new line after quitting
        print("Main thread received quit order.")
        while not self.frame_queue.empty():
            print("\rWaiting for queue to empty... Queue Size: {}".format(self.frame_queue.qsize()), end='')
            time.sleep(0.1)

        print("\n")  # Ensure to move to a new line after quitting
        # Uninitialize camera
        mvsdk.CameraUnInit(self.hCamera)
        # Free the memory buffer
        mvsdk.CameraAlignFree(self.pFrameBuffer)

    @mvsdk.method(mvsdk.CAMERA_SNAP_PROC)
    def GrabCallback(self, hCamera, pRawData, pFrameHead, pContext):
        if self.quit:
            #print("Returning without adding frames to the list")
            return

        current_time = time.time()
        #print(pFrameHead)
        FrameHead = pFrameHead[0]
        pFrameBuffer = self.pFrameBuffer

        # TODO check ImageProcess
        #mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
        #mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

        # At this time, the image is already stored in pFrameBuffer. 
        # For color cameras, pFrameBuffer=RGB data, for monochrome cameras, pFrameBuffer=8-bit grayscale data
        # Convert pFrameBuffer into OpenCV image format for subsequent algorithm processing
        #print(FrameHead.uBytes)
        #print(FrameHead.uiMediaType)
        frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pRawData)
        mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)
        #frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
        
        if not self.acquiring:
            self.acquiring = True
            self.t_start = time.time()


        if not self.force_framerate:
            if self.saving:
                frame_timestamp = time.time()
                # Copy frame data to a new buffer
                #frame_copy = np.copy(np.frombuffer(frame_data, dtype=np.uint8))
                #self.frame_queue.put((frame_copy, self.frame_count, frame_timestamp))
                self.frame_queue.put((frame_data, self.frame_count, FrameHead.uiTimeStamp, frame_timestamp))  # Add frame and timestamp to the buffer
            
            
            self.display_queue.put(frame_data)  # Add frame to display buffer    
            self.frame_count += 1
        else:
            if self.last_timestamp is None or current_time-self.last_timestamp >= self.special_frame_period:
                self.last_timestamp = current_time

                if self.saving:
                    frame_timestamp = time.time()
                    self.frame_queue.put((frame_data, self.frame_count, FrameHead.uiTimeStamp, frame_timestamp))  # Add frame and timestamp to the buffer
                
                self.display_queue.put(frame_data)  # Add frame to display buffer    
                self.frame_count += 1
            else:
                return

    ###
    def close_all_windows(self):
        try:
            plt.close('all')
        except Exception as e:
            print(f'Error closing matplotlib figures: {e}')
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            print(f'Error closing OpenCV windows: {e}')
    ###



def main():
    try:
        config = load_camera_config(CONFIG_FILE)
        app = App(config)
        app.save_thread.start()  # Start the save thread
        app.display_thread.start()  # Start the display thread
        app.main()
    finally:
        cv2.destroyAllWindows()
        app.quit = True
        app.save_thread.join()  # Ensure the save thread has finished
        app.display_thread.join()  # Ensure the display thread has finished
        app.udp_thread.join()
        plt.close('all')
        #app.close_all_windows()
        t_len = app.t_end-app.t_start 


if __name__ == '__main__':
    main()