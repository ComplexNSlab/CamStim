	# coding=utf-8
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
import matplotlib.pyplot as plt
import subprocess
import tkinter as tk
from tkinter import simpledialog
import json


CONFIG_FILE = 'cam_config.yml'

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
    def __init__(self, config, gui_mode=False):
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
        self.removeBackground = False
        self.save_queue_max_frames = int(config.get('SAVE_QUEUE_MAX_FRAMES', 2000))
        self.frame_queue = queue.Queue(maxsize=self.save_queue_max_frames)  # Buffer for save path
        self.display_queue = queue.Queue(maxsize=3)  # Keep only recent frames for display
        self.save_thread = threading.Thread(target=self.save_frames)  # Thread for saving frames
        self.display_thread = threading.Thread(target=self.display_frames)  # Create display thread
        self.frame_count = 0  # To keep track of saved 
        self.frames_written = 0
        self.dropped_display_frames = 0
        self.dropped_save_frames = 0
        self.save_overflow = False
        self.strict_no_drop_save = bool(config.get('STRICT_NO_DROP_SAVE', True))
        self.save_batch_frames = int(config.get('SAVE_BATCH_FRAMES', 64))
        self.save_processed_frames = bool(config.get('SAVE_PROCESSED_FRAMES', False))
        self.live_speck = config['USE_LIVE_SPECKLE']
        self.exposure = config['EXPOSURE_TIME'] # in ms
        self.analog_gain = config['ANALOG_GAIN'] 
        self.filename = config['EXPERIMENT']
        self.pwm_freq = config['PICO_PWM_FREQUENCY']
        self.pwm_duty = config['PICO_PWM_DUTY']
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

        self.histogram_open = False
        self.histogram_thread_running = False
        self.histogram_thread = None
        self.dFoF_open = False
        self.F0 = None

        self.exp_thread = None
        self.stim_thread = None
        self.logic_thread = None
        self.stim_progress = None
        self.logic_progress = None
        self.exp_list = {1: "Locally Sparse Noise", 
            2:"Dynamic Battery", 
            3: "Simple Orientation", 
            4: "Elevation Mapper", 
            5: "Retinotopy", 
            6: "Texture FB", 
            7: "Texture FB-VGG", 
            8: "Texture FB-VGGMultiTime", 
            9: "Square", 
            10: "Visual Field Mapping"}

        self.save_dir = None
        self.save_dir_ready = False
        self.save_file_handle = None
        self.gui_mode = gui_mode
        self.exp_status_queue = queue.Queue()
        self.latest_frame_data = None
        self.latest_frame_lock = threading.Lock()

        # self.check_and_fix_existing_experiment()


        # UDP socket to listen for the commands
        self.udp_port = config['UDP_TRIGGER_PORT']
        self.ip = '0.0.0.0'
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.udp_port))

        self.udp_thread = threading.Thread(target=self.wait_udp_trigger)
        self.udp_thread.start()

        self.dtype = 'uint16' if self.USE_MONO16 else 'uint8'

    def cleanup_udp(self):
        self.quit = True
        if hasattr(self, 'sock') and self.sock:
            try:
                self.sock.close()
            except:
                pass
        if hasattr(self, 'udp_thread') and self.udp_thread.is_alive():
            self.udp_thread.join(timeout=2.0)

    def get_frame_for_display(self):
        return self.get_latest_frame()

    def _clear_queue(self, q):
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break

    def _put_display_frame(self, frame_data):
        with self.latest_frame_lock:
            self.latest_frame_data = frame_data

        try:
            self.display_queue.put_nowait(frame_data)
        except queue.Full:
            try:
                self.display_queue.get_nowait()
            except queue.Empty:
                pass

            try:
                self.display_queue.put_nowait(frame_data)
            except queue.Full:
                # If a concurrent producer refilled it, keep the latest snapshot only.
                pass

            self.dropped_display_frames += 1

    def get_latest_frame(self):
        with self.latest_frame_lock:
            if self.latest_frame_data is None:
                return None
            return bytes(self.latest_frame_data)

    def experiment_status_callback(self, message):
        if hasattr(self, 'exp_status_queue'):
            self.exp_status_queue.put(("status", message))
        if not self.gui_mode:
            print(f"[Experiment] {message}")

    def experiment_trial_callback(self, current, total, message):
        if hasattr(self, 'exp_status_queue'):
            self.exp_status_queue.put(("trial", current, total, message))
        if not self.gui_mode:
            print(message)

    def get_exp_status(self):
        messages = []
        try:
            while True:
                msg = self.exp_status_queue.get_nowait()
                messages.append(msg)
        except queue.Empty:
            pass
        return messages

    def get_exp_params(self):
        if hasattr(self, 'exp_thread') and self.exp_thread and self.exp_thread.is_alive():
            print('\nExperiment selection already in progress.')
            return

        print("\n The available experiments are listed:")
        for num, name in self.exp_list.items():
            print(f"{num}: {name}")

        exp_name, experiment_id, mouse_id, method = self.show_exp_dialogs()
        if not exp_name: 
            return

        self.exp_thread = threading.Thread(target=self.run_exp, args=(exp_name, experiment_id, mouse_id, method), daemon=True)
        self.exp_thread.start()

    def show_exp_dialogs(self):
        try:
            if not hasattr(self, '_tk_root'):
                self._tk_root = tk.Tk()
                self._tk_root.withdraw()

            root = self._tk_root

            while True:
                exp_num = simpledialog.askinteger("Experiment", "Enter experiment number:", parent=root)
                if exp_num is None:
                    return (None, None, None, None)
                exp_name = self.exp_list.get(exp_num)
                if exp_name:
                    break

            experiment_id = simpledialog.askstring("Experiment ID", "Enter experiment ID:", parent=root)
            if experiment_id is None:
                return (None, None, None, None)

            self.save_dir = None
            self.save_dir_ready = False

            mouse_id = simpledialog.askstring("Mouse ID", "Enter mouse ID:", parent=root)
            if mouse_id is None:
                return (None, None, None, None)

            method = simpledialog.askstring("Method", "Send inputs via 'subprocess (s)' or '(u)'?", parent=root)
            if method is None:
                return (None, None, None, None)

            return (exp_name, experiment_id, mouse_id, method)

        except Exception as e:
            print(f"Dialog Error: {e}.")
            return (None, None, None, None)

    def run_exp(self, exp_name, experiment_id, mouse_id, method):
        try:
            self.exp_name = exp_name
            self.experiment_id = experiment_id
            self.mouse_id = mouse_id

            self.start_logic_analyzer(experiment_id, mouse_id)
            self.start_stim(exp_name, experiment_id, mouse_id, method,
                                self.experiment_status_callback,
                                self.experiment_trial_callback)

            threading.Thread(target=self.wait_for_directories, args=(experiment_id,), daemon=True).start()

        except Exception as e:
            print(f'\nError in experiment selection: {e}.')
        finally:
            self.exp_thread = None

    def wait_for_directories(self, experiment_id):
        base_dir = f'C:\\\\Data\\{experiment_id}'
        wf_recordings_dir = os.path.join(base_dir, "WF_Recordings")

        while not os.path.exists(base_dir) and not self.quit:
            time.sleep(0.1)

        if os.path.exists(base_dir):
            while not os.path.exists(wf_recordings_dir) and not self.quit:
                time.sleep(0.1)

            if os.path.exists(wf_recordings_dir):
                self.save_dir = wf_recordings_dir
                self.save_dir_ready = True
            else:
                print(f'\nWarning: video save directory nout found.')
        else:
            print(f'Warning: base directory never created.')

    def start_logic_analyzer(self, experiment_id, mouse_id):
        self.stop_logic_analyzer()

        print("\nStarting logic analyzer...")

        self.logic_progress = subprocess.Popen(["python", "-u", "C:/Users/admin/source/camstim/continuous_sigrok.py", experiment_id, mouse_id],
            stdout = subprocess.PIPE, stderr = subprocess.STDOUT, stdin = subprocess.PIPE, text=True, cwd="C:/Data/logicAnalyzer_Recordings")

        self.logic_thread = threading.Thread(target=self.track_logic, daemon=True)
        self.logic_thread.start()

    def stop_logic_analyzer(self):
        if getattr(self, 'logic_progress', None) is None:
            print("\nNo current logic analyzer session running.")
            return

        print("\nStopping logic analyer session...")

        if isinstance(self.logic_progress, subprocess.Popen):
            try:
                if self.logic_progress.poll() is None:  
                    print("\nSending STOP command to logic analyzer...")
                    self.logic_progress.stdin.write("STOP\n")
                    self.logic_progress.stdin.flush()

                    try:
                        self.logic_progress.wait(timeout=5.0)
                        print("\nLogic analyzer stopped gracefully.")
                    except subprocess.TimeoutExpired:
                        print("\nLogic analyzer didn't respond to STOP, forcing terminate...")
                        self.logic_progress.terminate()
                        self.logic_progress.wait(timeout=2.0)
                    print("\nLogic analyzer terminated.")
                else:
                    print("Logic analyzer process has already finished.")
            except (OSError, subprocess.TimeoutExpired) as e:
                print(f"\nError stopping logic analyzer: {e}.")
                if self.logic_progress.poll() is None:
                    self.logic_progress.kill()

        self.logic_progress = None

        if hasattr(self, 'logic_thread') and self.logic_thread and self.logic_thread.is_alive():
            self.logic_thread.join(timeout=1.0)
            self.logic_thread = None
       
    def track_logic(self):
        if isinstance(self.logic_progress, subprocess.Popen):
            for line in self.logic_progress.stdout:
                print(f"{line.strip()}")

    def start_stim(self, exp_name, experiment_id, mouse_id, method, status_callback=None, trial_callback=None):
        if getattr(self, 'stim_progress', None):
            print("Experiment currently running. Stopping...")
            if hasattr(self.stim_progress, 'terminate'):
                self.stim_progress.terminate()
            self.stop_stim()

        print(f"\nStarting {exp_name} with exp. ID {experiment_id} and mouse {mouse_id}...")

        self.status_callback = status_callback
        self.trial_callback = trial_callback

        if method.lower() == 's':
            self.stim_progress = subprocess.Popen(["python", "-u", "C:/Users/admin/source/camstim/wf_main.py", exp_name, experiment_id, mouse_id],
                stdout = subprocess.PIPE, stderr = subprocess.STDOUT, stdin = subprocess.PIPE, text=True)
            threading.Thread(target=self.track_stim, daemon=True).start()
            print("\nStarted stim via subprocess.")

        elif method.lower() == 'u':
            msg = {"cmd": "START", 
                'exp_name': exp_name, 
                "experiment_id": experiment_id, 
                "mouse_id": mouse_id}
            
            UDP_IP = "127.0.0.1"
            UDP_PORT = 5005

            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as send_sock:
                send_sock.sendto(json.dumps(msg).encode(), (UDP_IP, UDP_PORT))
                print(f"Sent UDP trigger to visual stim at {UDP_IP}:{UDP_PORT}")

            self.stim_progress = True
        else:
            raise ValueError(f"Unknown method {method}. Choose 's' or 'u''.")

    def stop_stim(self):
        if getattr(self, 'stim_progress', None) is None:
            print("No current experiment running.")
        else:
            if isinstance(self.stim_progress, subprocess.Popen):
                print("Stopping stim via subprocess...")
                try:
                    if self.stim_progress.poll() is None:
                        self.stim_progress.stdin.write("STOP\n")
                        self.stim_progress.stdin.flush()
                        self.stim_progress.terminate()
                        self.stim_progress.wait(timeout=2.0)
                        print("\nStim stopped.")
                    else:
                        print("\nStim process already finished.")
                except (OSError, subprocess.TimeoutExpired) as e:
                    print(f'\nError stopping stim: {e}.')
                    if self.stim_progress.poll() is None:
                        self.stim_progress.kill()

            elif self.stim_progress is True:
                msg = {"cmd": "STOP"}
                UDP_IP = "127.0.0.1"
                UDP_PORT = 5005
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as send_sock:
                    send_sock.sendto(json.dumps(msg).encode(), (UDP_IP, UDP_PORT))
                    print(f"\nSent stop to visual stim at {UDP_IP}:{UDP_PORT}")
                
            self.stim_progress = None

        self.stop_logic_analyzer()

        if hasattr(self, 'stim_thread') and self.stim_thread and self.stim_thread.is_alive():
            self.stim_thread.join(timeout=1.0)
            self.stim_thread = None

    def track_stim(self):
        if isinstance(self.stim_progress, subprocess.Popen):
            exp_completed = False

            for line in self.stim_progress.stdout:
                line = line.strip()
                print(line)

                if "Trial" in line and "out of" in line:
                    try:
                        parts = line.split()
                        if len(parts) >= 4 and parts[0] == "Trial" and parts[2] == "out" and parts[3] == "of":
                            current = int(parts[1])
                            total = int(parts[4]) if len(parts) > 4 else 0
                            if hasattr(self, 'trial_callback') and self.trial_callback:
                                self.trial_callback(current, total, line)

                    except (ValueError, IndexError):
                        if hasattr(self, 'status_callback') and self.status_callback:
                            self.status_callback(line)
                else:
                    if hasattr(self, 'status_callback') and self.status_callback:
                        self.status_callback(line)

                if "Experiment routine completed." in line:
                    print("\nStimulus completed message detected, stopping logic analyzer...")
                    exp_completed = True
                    self.stop_logic_analyzer()

            print("\nStim process finished.")
            
            if not exp_completed:
                if (getattr(self, 'logic_progress', None) and 
                    isinstance(self.logic_progress, subprocess.Popen) and
                    self.logic_progress.poll() is None):
                    print("\nEnsuring logic analyzer is stopped...")
                    self.stop_logic_analyzer()  

            self.status_callback = None
            self.trial_callback = None     

    def setup_live_speckle_variables(self):
        self.buffer_size = self.config['BUFFER_SIZE']
        self.circular_buffer = np.zeros((self.buffer_size, self.height//self.bin_size, self.width//self.bin_size), dtype=np.float32)
        self.mean_image = np.zeros((self.height//self.bin_size, self.width//self.bin_size), dtype=np.float32)
        self.backgroundImg = np.zeros((self.height//self.bin_size, self.width//self.bin_size), dtype=np.float32)

        self.current_buffer_item = 0
        self.enable_live_speckle = False
        self.buffer_loop_reached = False

    def check_and_fix_existing_experiment(self):
        if self.save_dir and os.path.exists(os.path.join(self.save_dir, self.filename+'.bin')):
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
        last_flush_time = time.time()
        flush_interval_s = 0.5
        flush_every_n_frames = max(1, self.save_batch_frames)
        write_batch = []
        ts_batch = []
        sys_ts_batch = []

        def flush_batch(force_flush=False):
            nonlocal last_flush_time
            if not write_batch or self.save_file_handle is None:
                return

            if self.save_processed_frames:
                # Process and write each frame when explicitly requested.
                for frame_data in write_batch:
                    frame = np.frombuffer(frame_data, dtype=self.dtype).reshape((self.height, self.width))
                    frame = cv2.flip(frame, 1)
                    if self.bin_exp:
                        frame = self.bin_frame(frame)
                    frame.tofile(self.save_file_handle)
            else:
                # Fast path: write raw camera bytes exactly as acquired.
                self.save_file_handle.write(b''.join(write_batch))

            self.frame_timestamps.extend(ts_batch)
            self.sys_clock_timestamps.extend(sys_ts_batch)
            self.frames_written += len(write_batch)

            should_flush = force_flush or ((time.time() - last_flush_time) >= flush_interval_s)
            if should_flush:
                self.save_file_handle.flush()
                last_flush_time = time.time()

            write_batch.clear()
            ts_batch.clear()
            sys_ts_batch.clear()

        while True:
            if self.saving:
                if not self.save_dir_ready or self.save_dir is None:
                    time.sleep(0.01)
                    continue

                if self.save_file_handle is None:
                    file_path = os.path.join(self.save_dir, self.filename + '.bin')
                    # Buffered appends improve sustained throughput when frame rate is high.
                    self.save_file_handle = open(file_path, 'ab', buffering=4 * 1024 * 1024)

                # Process frames if available
                try:
                    frame_data, count, timestamp, sys_stamp = self.frame_queue.get(timeout=0.02)
                except queue.Empty:
                    flush_batch(force_flush=True)
                    if self.quit and self.frame_queue.empty():
                        break
                    continue

                write_batch.append(frame_data)
                ts_batch.append(timestamp)
                sys_ts_batch.append(sys_stamp)
                if len(write_batch) >= flush_every_n_frames:
                    flush_batch(force_flush=False)

                # Check if it's time to exit: quit is True and no frames left in the queue
                if self.quit and self.frame_queue.empty():
                    flush_batch(force_flush=True)
                    break
            else:
                flush_batch(force_flush=True)
                if self.quit:
                    break
                time.sleep(0.005)

        if self.save_file_handle is not None:
            flush_batch(force_flush=True)
            self.save_file_handle.flush()
            self.save_file_handle.close()
            self.save_file_handle = None

        if self.frames_written > 0 and self.save_dir_ready:
            # After processing all frames, save metadata
            metadata = {
                'num_frames': self.frames_written,
                'frame_width': self.width if (not self.save_processed_frames or not self.bin_exp) else self.width//self.bin_size,
                'frame_height': self.height if (not self.save_processed_frames or not self.bin_exp) else self.height//self.bin_size,
                'data_type': self.dtype if (not self.save_processed_frames or not self.bin_exp) else 'uint16',
                'frame_timestamps': self.frame_timestamps,
                'sys_clock_timestamps': self.sys_clock_timestamps,
                'frame_exposure': self.exposure,
                'frame_gain': self.analog_gain,
                'pwm_frequency': self.pwm_freq,
                'pwm_duty': self.pwm_duty,
                'binned_live': self.bin_exp,
                'save_processed_frames': self.save_processed_frames,
                'bin_size': self.bin_size,
                'force_framerate': self.force_framerate,
                'special_framerate': self.special_framerate
            }
            np.save(os.path.join(self.save_dir, '{}_metadata.npy'.format(self.filename)), metadata)

    def display_frames(self):
        print(f"DEBUG: display_frames started, gui_mode = {self.gui_mode}")
        if self.gui_mode:
            while not self.quit:
                if not self.display_queue.empty():
                    frame_data = self.display_queue.get()
                    time.sleep(0.001)
                else:
                    time.sleep(0.005)
            return

        print("DEBUG: Running in OpenCV mode.")
        cv2.namedWindow("Live View")
        cv2.setMouseCallback("Live View", self.roi_drawer.handle_mouse_events)

        while not self.quit or not self.display_queue.empty():
            if not self.display_queue.empty():
                frame_data = self.display_queue.get()

                # Display frame
                frame = np.frombuffer(frame_data, dtype=self.dtype)
                frame = frame.reshape((self.height, self.width))
                frame = cv2.flip(frame, 1)

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
            
                if self.removeBackground:
                    frame = frame - self.backgroundImg
                    frame = np.clip(frame, 0, 255)

                if self.normalizeImage:
                    clipped = np.clip(frame, self.minI, self.maxI)
                    scaled = ((clipped - self.minI) / (self.maxI - self.minI)) * 255
                    frame = scaled.astype(np.uint8)

                if self.dFoF_open and self.F0 is not None:
                    dfof = (frame.astype(np.float32) - self.F0) / self.F0
                    dfof = np.nan_to_num(dfof, nan=0.0)
                    
                    if self.normalizeImage:
                        clipped = np.clip(dfof, self.minI, self.maxI)
                        frame = ((clipped - self.minI) / (self.maxI - self.minI)) * 255
                    else:
                        fmin, fmax = dfof.min(), dfof.max()
                        if fmax > fmin:
                            frame = ((dfof - fmin) / (fmax - fmin)) * 255
                        else:
                            frame = np.zeros_like(dfof)

                    frame = frame.astype(np.uint8)

                frame  = cv2.resize(frame, (self.width//2,self.height//2), interpolation = None)
                
                frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
                frame = self.roi_drawer.draw_rectangle(frame)    

                if self.plot_roi and self.roi_drawer.top_left_pt != (-1, -1) and self.roi_drawer.bottom_right_pt != (-1, -1):
                    average_intensity = frame[self.roi_drawer.top_left_pt[1]:self.roi_drawer.bottom_right_pt[1], self.roi_drawer.top_left_pt[0]:self.roi_drawer.bottom_right_pt[0]].mean()
                    self.roi_plotter.update_plot(average_intensity)
                    
                if self.plot_roi or self.live_speck:
                    self._clear_queue(self.display_queue)

                cv2.imshow("Live View", frame)

            pressed_key = cv2.waitKey(1) & 0xFF
            if pressed_key == 255:
                continue

            elif pressed_key == ord('q'):
                self.roi_plotter.deinitialize_plot()
                self.plot_roi = False
                self.quit = True
                self.t_end = time.time()

            elif pressed_key == ord('i'):
                if getattr(self, 'stim_progress', None) or getattr(self, 'logic_progress', None):
                    print("\nStopping experiment and logic analyzer...")
                    self.stop_stim()
                    self.stim_progress = None
                    self.logic_progress = None
                else:
                    print("No current experiment running.")

            elif pressed_key == ord('a'):
                self.toggle_histogram()

            elif pressed_key == ord('y'):
                self.toggle_dFoF()

            elif pressed_key == ord('x'):
                self.get_exp_params()

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
                self.change_exposure()

            elif pressed_key == ord('g'):
                self.change_gain()

            elif pressed_key == ord('r') and self.roi_drawer.active_roi:
                self.toggle_roi_plot()

            elif pressed_key == ord('p'):
                self.print_camera_stats()

            elif pressed_key == ord('d'):
                self.toggle_background_removal()

            elif pressed_key == ord('z'):
                self.adjust_dynamic_range()

            elif pressed_key == ord('h'):
                self.print_keyboard_commands()

            elif self.live_speck and pressed_key == ord('b'):
                self.toggle_speckle()

        print("Quit order received for display thread.")
        self._clear_queue(self.display_queue)

    def toggle_speckle(self):
        self.enable_live_speckle = True if not self.enable_live_speckle else False
        if self.enable_live_speckle:
            print("Enabling live speckle imaging, please wait a few seconds for the buffer to fill up.")
        else:
            print("Disabled live speckle imaging.")

    def toggle_histogram(self):
        if self.histogram_open:
            plt.close('Histogram')
            self.histogram_open = False
            self.histogram_thread_running = False
            if hasattr(self, 'histogram_thread') and self.histogram_thread.is_alive():
                self.histogram_thread.join(timeout=1.0)
        else:
            self.histogram_open = True
            self.histogram_thread_running = True
            self.histogram_thread = threading.Thread(target=self.update_histogram, daemon=True)
            self.histogram_thread.start()


    def update_histogram(self):
        hist_width = 512
        hist_height = 400
        bin_width = 2

        cv2.namedWindow('Live Histogram', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Live Histogram', hist_width, hist_height)

        max_history = 50
        intensity_history = []

        while self.histogram_open and self.histogram_thread_running and not self.quit:
            frame_data = self.get_latest_frame()
            if frame_data is not None:
                try:
                    frame = np.frombuffer(frame_data, dtype=self.dtype)
                    frame = frame.reshape((self.height, self.width))

                    if self.dFoF_open and self.F0 is not None:
                        dfof = (frame.astype(np.float32) - self.F0) / self.F0
                        dfof = np.nan_to_num(dfof, nan=0.0)

                        if self.normalizeImage:
                            clipped = np.clip(dfof, self.minI, self.maxI)
                            normalized_frame = ((clipped - self.minI) / (self.maxI - self.minI)) * 255
                        else:
                            fmin, fmax = dfof.min(), dfof.max()
                            if fmax > fmin:
                                normalized_frame = ((dfof - fmin) / (fmax - fmin)) * 255
                            else:
                                normalized_frame = np.zeros_like(dfof) * 255

                        display_frame = normalized_frame.astype(np.uint8)
                        title_suffix = " (dFoF)"

                    else:
                        if self.normalizeImage:
                            clipped = np.clip(frame, self.minI, self.maxI)
                            display_frame = ((clipped - self.minI) / (self.maxI - self.minI)) * 255
                            display_frame = display_frame.astype(np.uint8)
                        else:
                            display_frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                        title_suffix = " (Raw)"

                    intensity_history.extend(display_frame.ravel().tolist())
                    if len(intensity_history) > max_history * self.height * self.width:
                        intensity_history = intensity_history[-max_history * self.height * self.width:]

                    hist = cv2.calcHist([np.array(intensity_history, dtype=np.uint8)], [0], None, [256], [0, 256])

                    cv2.normalize(hist, hist, 0, hist_height, cv2.NORM_MINMAX)

                    hist_image = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)

                    for i in range(256):
                        intensity = int(hist[i])
                        cv2.rectangle(hist_image, (i * bin_width, hist_height - intensity),((i+1) * bin_width - 1, hist_height), (255, 255, 255), -1)

                    if self.normalizeImage:
                        min_x = int(self.minI * bin_width)
                        max_x = int(self.maxI * bin_width)
                        cv2.line(hist_image, (min_x, 0), (min_x, hist_height), (0, 0, 255), 2)
                        cv2.line(hist_image, (max_x, 0), (max_x, hist_height), (255, 0, 0), 2)

                        cv2.putText(hist_image, f'min: {self.minI:.2f}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(hist_image, f'max: {self.maxI:.2f}', (hist_width - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                    cv2.putText(hist_image, f'Frame {self.frame_count}{title_suffix}', (hist_width // 2 - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    cv2.imshow('Live Histogram', hist_image)
                    cv2.waitKey(1)

                except queue.Empty:
                    pass
                except Exception as e:
                    print(f"\nHistogram error: {e}.")

            time.sleep(0.01)

        cv2.destroyWindow('Live Histogram')


    def toggle_dFoF(self):
        if self.dFoF_open:
            self.dFoF_open = False
            print("\n Stopping live dFoF view.")
        else:
            print("\n Starting live dFoF view.")
            baseline_frames = []
            F0_n_frames = 50

            try:
                for i in range(F0_n_frames):
                    frame_data = self.get_latest_frame()
                    if frame_data is None:
                        time.sleep(0.02)
                        continue
                    frame = np.frombuffer(frame_data, dtype=self.dtype)
                    frame = frame.reshape((self.height, self.width))
                    baseline_frames.append(frame.astype(np.float32))
                    time.sleep(0.01)
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

                    epsilon = 1e-6
                    self.F0[self.F0 == 0] = epsilon

                    del sampled_stack
                    
                    if not self.normalizeImage:
                        test_dfof = []
                        for i in range(min(10, len(baseline_frames))):
                            dfof_test = (baseline_frames[i] - self.F0) / self.F0
                            test_dfof.append(dfof_test)

                        test_dfof_stack = np.stack(test_dfof)

                        valid_values = test_dfof_stack[~np.isnan(test_dfof_stack)]
                        if len(valid_values) > 0:
                            self.minI = np.percentile(valid_values, 1)
                            self.maxI = np.percentile(valid_values, 99)
                            self.normalizeImage = True
                            print(f"\nAuto-set dFoF range: [{self.minI:.3f}, {self.maxI:.3f}]")
                        else:
                            print("\nWarning: All dFoF values are NaN..")
                            return

                    self.dFoF_open = True
                    print("\ndFoF started with dynamic range normalization.")

    def change_exposure(self):
        try:
            self.exposure = float(input("\nEnter new exposure (current: {}ms): \n".format(self.exposure)))
            mvsdk.CameraSetExposureTime(self.hCamera, self.exposure*1000) 
        except ValueError:
            print("\n Failed to set exposure, invalid value.")
        
        print("TrigCap: ", mvsdk.CameraGetExtTrigCapability(self.hCamera))
        print("Trigger delay time: ", mvsdk.CameraGetExtTrigDelayTime(self.hCamera))

    def change_gain(self):
        try:
            self.analog_gain = int(input("\nEnter new gain (current: {}): \n".format(self.analog_gain)))
            mvsdk.CameraSetAnalogGain(self.hCamera, self.analog_gain) 
        except ValueError:
            print("\n Failed to set gain, invalid value.")

    def toggle_roi_plot(self):
        self.plot_roi = True if not self.plot_roi else False
        if not self.plot_roi:
            self.roi_drawer.top_left_pt = (-1, -1)
            self.roi_drawer.bottom_right_pt = (-1, -1)
            self.roi_drawer.active_roi = False
            self.roi_plotter.deinitialize_plot()

    def toggle_background_removal(self):
        if self.removeBackground:
            self.removeBackground = False
            print("\n Stopping background substraction.")
        else:
            self.removeBackground = True
            frame_data = self.get_latest_frame()
            if frame_data is None:
                print("\n No frame available, could not capture background image.")
                self.removeBackground = False
                return
            frame = np.frombuffer(frame_data, dtype=self.dtype)
            frame = frame.reshape((self.height, self.width))                     
            kernel = np.ones((10,10),np.float32)/100
            frame = cv2.filter2D(frame,-1,kernel)
            self.backgroundImg = frame
            print("\n Substracting background image.")

    def adjust_dynamic_range(self):
        if self.normalizeImage:
            self.normalizeImage = False
            print("\nDynamic range normalization turned off.")
            return

        self.normalizeImage = True
        self.autoI = self.autoI * 2
        if self.autoI > 49:
            self.autoI = 0.05

        frame_data = self.get_latest_frame()
        if frame_data is not None:
            frame = np.frombuffer(frame_data, dtype=self.dtype)
            frame = frame.reshape((self.height, self.width))

            if self.dFoF_open and self.F0 is not None:
                dfof = (frame.astype(np.float32) - self.F0) / self.F0
                dfof = np.nan_to_num(dfof, nan=0.0)
                data_for_percentile = dfof
                data_type = "dfof"
            else:
                data_for_percentile = frame
                data_type = "raw"
            
            self.minI = np.percentile(data_for_percentile[:], self.autoI)
            self.maxI = np.percentile(data_for_percentile[:], 100 - self.autoI)

            print("\nDynamic range normalization turned on.")
            print(f"\nChanging {data_type} dynamic range to: [{self.minI:.3f}, {self.maxI:.3f}]. Percentiles: [{self.autoI}, {100-self.autoI}]")  # FIXED: added closing bracket
        else:
            print("\nDynamic range normalization turned on.")
            print("\nNo frame available for dynamic range adjustment.")

        # self.minI = np.percentile(frame[:], self.autoI)
        # self.maxI = np.percentile(frame[:], 100-self.autoI)
        # print("\n Changing dynamical range to: {}, {} pixel values. Percentiles: {}, {}".format(self.minI, self.maxI, self.autoI, 100-self.autoI))

    def wait_udp_trigger(self):
        self.sock.setblocking(0)
        while not self.quit:
            # Receive message
            ready = select.select([self.sock], [], [], 1)
            if ready[0]:
                try:
                    data, addr = self.sock.recvfrom(1024)  # buffer size is 1024 bytes
                    msg = data.decode()

                    if 'ExpStart' in msg:
                        print("Experiment started - waiting for hardware triggers.")
                        self.frame_count = 0
                        self.t_start = time.time()
                    elif 'ExpEnd' in msg:
                        print("Experiment ended.")
                    else:
                        print("Received: ", msg)
                except:
                    if self.quit:
                        break
        print("UDP thread stopped.")


    def print_camera_stats(self):
        print("\nExposure(ms): {} Gain: {} ".format(self.exposure, self.analog_gain))

    # function to display keyboard commands help 
    def print_keyboard_commands(self):
        print("\n\nUse the keyboard commands listed below to navigate and change settings within the program. To use a command, first click on the display window before entering the key. If additional follow up inputs are needed (eg exposure numbers, gain numbers), click back to the terminal window before entering.\n\n" +\
            "q -- quit\n" +\
            "t -- switch camera mode to hardware trigger capture\n" +\
            "c -- switch camera mode to continuous capture\n" +\
            "s -- if in continuous capture, switches to hardware trigger mode (ready to save frames); \n if in hardware trigger mode, switches to continuous capture (save frames off)\n" +\
            "x -- enter experiment/stimulus parameters (make sure to hit 's' before this!)\n" +\
            "i -- interrupt experiment (stop data acquistion, teensy, logic analyzer)\n" +\
            "e -- change exposure\n" +\
            "g -- edit gain\n" +\
            "r -- draw ROI\n" +\
            "a -- display pixel intensity histogram for current frame\n" +\
            "p -- print camera stats\n" +\
            "d -- subtract backgroud image\n" +\
            "z -- normaliZe image dynamic range\n" +\
            "b -- speckle mode on/off\n" +\
            "r -- ROI selection\n" +\
            "h -- display these commands again\n")

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
        print(f"Camera resolution: {self.width}x{self.height}")

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
            print("\rSave Queue: {}, Frames Saved: {}, Save Drops: {}, Display Queue: {}, Display Drops: {}, Frames Displayed: {}, Average FPS: {:.2f} Saturated Pixels: {:06d}".format(
                self.frame_queue.qsize(), self.frames_written, self.dropped_save_frames, self.display_queue.qsize(), self.dropped_display_frames, self.frame_count, average_fps, self.n_saturated_pixels), end='')
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
        FrameHead = pFrameHead[0]
        pFrameBuffer = self.pFrameBuffer

        # TODO check ImageProcess
        # mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
        # mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

        # At this time, the image is already stored in pFrameBuffer. 
        # For color cameras, pFrameBuffer=RGB data, for monochrome cameras, pFrameBuffer=8-bit grayscale data
        # Convert pFrameBuffer into OpenCV image format for subsequent algorithm processing
        
        # 0506 JO update
        # frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pRawData)
        # mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)
        frame_data = bytes((mvsdk.c_ubyte * FrameHead.uBytes).from_address(pRawData))
        mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)


        if not self.acquiring:
            self.acquiring = True
            self.t_start = time.time()


        if not self.force_framerate:
            if self.saving:
                frame_timestamp = time.time()
                try:
                    self.frame_queue.put_nowait((frame_data, self.frame_count, FrameHead.uiTimeStamp, frame_timestamp))
                except queue.Full:
                    self.save_overflow = True
                    if self.strict_no_drop_save:
                        print("CRITICAL: save queue overflow, stopping acquisition to prevent silent frame loss.")
                        self.quit = True
                        return
                    self.dropped_save_frames += 1
                    if self.dropped_save_frames % 100 == 1:
                        print("Warning: save queue full, dropping frames to keep acquisition real-time.")
            # Stop addding to the display queue if the frame queue is getting too full
            if self.frame_queue.qsize() < 0.95 * self.frame_queue.maxsize:
                self._put_display_frame(frame_data)
            self.frame_count += 1
        else:
            if self.last_timestamp is None or current_time-self.last_timestamp >= self.special_frame_period:
                self.last_timestamp = current_time

                if self.saving:
                    frame_timestamp = time.time()
                    try:
                        self.frame_queue.put_nowait((frame_data, self.frame_count, FrameHead.uiTimeStamp, frame_timestamp))
                    except queue.Full:
                        self.save_overflow = True
                        if self.strict_no_drop_save:
                            print("CRITICAL: save queue overflow, stopping acquisition to prevent silent frame loss.")
                            self.quit = True
                            return
                        self.dropped_save_frames += 1
                        if self.dropped_save_frames % 100 == 1:
                            print("Warning: save queue full, dropping frames to keep acquisition real-time.")
                # Stop addding to the display queue if the frame queue is getting too full
                if self.frame_queue.qsize() < 0.95 * self.frame_queue.maxsize:
                    self._put_display_frame(frame_data)
                self.frame_count += 1
            else:
                return

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

if __name__ == '__main__':
    main()
