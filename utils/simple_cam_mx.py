	# coding=utf-8
import sys
import cv2
import numpy as np
from core import mvsdk
import time
import platform
import queue
import threading
import serial
import yaml
from pathlib import Path
import os
from core.roi_module import ROIDrawer, ROIPlotter
import matplotlib.pyplot as plt
import subprocess
import tkinter as tk
from tkinter import simpledialog
import shutil
from core.experiment_discovery import get_experiment_list

REPO_ROOT = Path(__file__).resolve().parent.parent

CONFIG_DIR = REPO_ROOT / 'config_files'
CONFIG_FILE = str(CONFIG_DIR / 'cam_config.yaml')
SAVE_SETTINGS_CONFIG_FILE = str(CONFIG_DIR / 'config.yaml')


def _resolve_config_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path

    repo_candidate = REPO_ROOT / path
    if repo_candidate.is_file():
        return repo_candidate

    return Path(__file__).resolve().parent / path

def load_camera_config(yaml_file_path):
    config_path = _resolve_config_path(yaml_file_path)

    if config_path.is_file():
        with open(config_path, 'r') as file:
            try:
                config = yaml.safe_load(file)
                return config
            except yaml.YAMLError:
                raise Exception("There is an error in the config yaml file, please check it. {}".format(config_path))
    else:
        raise Exception("Configuration file does not exist, please create it.")


def load_save_root(yaml_file_path=SAVE_SETTINGS_CONFIG_FILE):
    config_path = _resolve_config_path(yaml_file_path)

    if not config_path.is_file():
        raise Exception("Save settings file does not exist, please create it. {}".format(config_path))

    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file) or {}
        except yaml.YAMLError:
            raise Exception("There is an error in the save settings yaml file, please check it. {}".format(config_path))

    save_root = config.get('SAVE_DIR')
    if not save_root:
        raise Exception("SAVE_DIR missing in {}".format(config_path))

    return save_root



class App(object):
    def __init__(self, config, gui_mode=False):
        super(App, self).__init__()

        self.config = config
        self.gui_mode = gui_mode
        debug_skip_teensy_cfg = config.get('DEBUG_SKIP_TEENSY', False)
        if isinstance(debug_skip_teensy_cfg, str):
            self.debug_skip_teensy = debug_skip_teensy_cfg.strip().lower() in ('1', 'true', 'yes', 'on')
        else:
            self.debug_skip_teensy = bool(debug_skip_teensy_cfg)
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
        self.save_batch_frames = int(config.get('SAVE_BATCH_FRAMES', 64))
        self.live_speck = config['USE_LIVE_SPECKLE']
        self.exposure = config['EXPOSURE_TIME'] # in ms
        self.analog_gain = config['ANALOG_GAIN'] 
        self.filename = str(config.get('EXPERIMENT', 'recording'))
        self.bin_exp = bool(config['BIN_EXP_LIVE'])
        self.bin_size = config['BIN_SIZE']
        self.bin_mode = str(config.get('BIN_MODE', 'software')).strip().lower()
        if self.bin_mode not in ('software', 'camera'):
            print(f"Warning: invalid BIN_MODE '{self.bin_mode}', defaulting to software.")
            self.bin_mode = 'software'
        self.hardware_bin_enabled = False
        self.request_camera_bin = self.bin_exp and self.bin_mode == 'camera'
        self.bin_exp = self.bin_exp and self.bin_mode == 'software'
        self.zeros = np.zeros((255, 255), dtype=np.uint8) # debug image in case I have problems with camera
        self.frame_timestamps = []  # List to store timestamps
        self.sys_clock_timestamps = []
        self.session_frame_timestamps = []
        self.session_sys_clock_timestamps = []
        self.USE_MONO16 = False # If false defaults to 8 bit although current camera doesn't have true 16 bit
        self.t_start = None
        self.t_end = None
        self.hCamera = None
        self.n_saturated_pixels = 0
        self.session_frames_written = 0
        self.on_logic_analyzer_terminated = None

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
        self.exp_list = get_experiment_list()  # Dynamically discover experiments

        self.save_dir = None
        self.save_dir_ready = False
        self.save_file_handle = None
        self.gui_mode = gui_mode
        self.exp_status_queue = queue.Queue()
        self.latest_frame_data = None
        self.latest_frame_lock = threading.Lock()

        # self.check_and_fix_existing_experiment()

        self.dtype = 'uint16' if self.USE_MONO16 else 'uint8'

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

        if self.gui_mode:
            return

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

    def _enqueue_save_frame(self, frame_data, frame_index, camera_timestamp, system_timestamp):
        queued_frame = (frame_data, frame_index, camera_timestamp, system_timestamp)

        while not self.quit:
            try:
                self.frame_queue.put(queued_frame, timeout=0.1)
                return True
            except queue.Full:
                self.save_overflow = True

        return False

    def _decode_mode_mask(self, mask):
        return [bit + 2 for bit in range(32) if mask & (1 << bit)]

    def _probe_camera_binning_support(self, cap, mono_camera):
        res_range = cap.sResolutionRange
        print(f"Hardware skip modes: {self._decode_mode_mask(res_range.uSkipModeMask)}")
        print(f"Hardware bin-sum modes: {self._decode_mode_mask(res_range.uBinSumModeMask)}")
        print(f"Hardware bin-average modes: {self._decode_mode_mask(res_range.uBinAverageModeMask)}")

        if not mono_camera:
            print("MONO16 probe skipped: camera is not monochrome.")
            return

        original_format = None
        try:
            original_format = mvsdk.CameraGetIspOutFormat(self.hCamera)
            probe_error = mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO16)
            if probe_error == mvsdk.CAMERA_STATUS_SUCCESS:
                applied_format = mvsdk.CameraGetIspOutFormat(self.hCamera)
                mono16_enabled = (applied_format == mvsdk.CAMERA_MEDIA_TYPE_MONO16)
                print(f"MONO16 probe: supported={mono16_enabled}, applied_format={applied_format}")
            else:
                print(f"MONO16 probe: rejected with SDK status {probe_error}")
        except Exception as e:
            print(f"MONO16 probe failed: {e}")
        finally:
            if original_format is not None:
                try:
                    mvsdk.CameraSetIspOutFormat(self.hCamera, original_format)
                except Exception as e:
                    print(f"Warning: failed to restore original ISP format after probe: {e}")

    def _fallback_to_software_binning(self, reason):
        print(f"Camera binning unavailable: {reason}. Falling back to software binning.")
        self.request_camera_bin = False
        self.hardware_bin_enabled = False
        self.bin_mode = 'software'
        self.bin_exp = True

    def _apply_camera_binning(self, cap, mono_camera):
        if not self.request_camera_bin:
            return

        if not mono_camera:
            self._fallback_to_software_binning('camera is not monochrome')
            return

        if self.bin_size < 2:
            self._fallback_to_software_binning(f'invalid BIN_SIZE {self.bin_size}')
            return

        bin_bit = 1 << (self.bin_size - 2)
        if not (cap.sResolutionRange.uBinSumModeMask & bin_bit):
            self._fallback_to_software_binning(f'hardware sum binning {self.bin_size}x{self.bin_size} is not supported')
            return

        try:
            mono16_error = mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO16)
            if mono16_error != mvsdk.CAMERA_STATUS_SUCCESS:
                self._fallback_to_software_binning(f'MONO16 rejected with SDK status {mono16_error}')
                return

            applied_format = mvsdk.CameraGetIspOutFormat(self.hCamera)
            if applied_format != mvsdk.CAMERA_MEDIA_TYPE_MONO16:
                self._fallback_to_software_binning(f'MONO16 did not stick (applied format {applied_format})')
                return

            image_res = mvsdk.CameraGetImageResolution(self.hCamera)
            image_res.uSkipMode = 0
            image_res.uBinAverageMode = 0
            image_res.uBinSumMode = bin_bit

            set_error = mvsdk.CameraSetImageResolution(self.hCamera, image_res)
            if set_error != mvsdk.CAMERA_STATUS_SUCCESS:
                self._fallback_to_software_binning(f'CameraSetImageResolution failed with SDK status {set_error}')
                return

            applied_res = mvsdk.CameraGetImageResolution(self.hCamera)
            self.width = applied_res.iWidth
            self.height = applied_res.iHeight
            self.USE_MONO16 = True
            self.dtype = 'uint16'
            self.hardware_bin_enabled = True
            self.bin_exp = False
            self.bin_mode = 'camera'
            print(f"Enabled camera binning: sum {self.bin_size}x{self.bin_size}, output {self.width}x{self.height}, format MONO16")
        except Exception as e:
            self._fallback_to_software_binning(str(e))

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

    def get_exp_params(self, exp_name=None, experiment_id=None, mouse_id=None):
        if hasattr(self, 'exp_thread') and self.exp_thread and self.exp_thread.is_alive():
            print('\nExperiment selection already in progress.')
            return False

        if exp_name is None or experiment_id is None or mouse_id is None:
            print("\n The available experiments are listed:")
            for i, name in enumerate(self.exp_list, 1):
                print(f"{i}: {name}")

            exp_name, experiment_id, mouse_id = self.show_exp_dialogs()

        exp_name = (exp_name or '').strip()
        experiment_id = (experiment_id or '').strip()
        mouse_id = (mouse_id or '').strip()
        if not exp_name or not experiment_id or not mouse_id:
            return False

        self.exp_name = exp_name
        self.experiment_id = experiment_id
        self.mouse_id = mouse_id
        self.filename = f"{mouse_id}_{experiment_id}"

        self.exp_thread = threading.Thread(target=self.run_exp, args=(exp_name, experiment_id, mouse_id), daemon=True)
        self.exp_thread.start()
        return True

    def show_exp_dialogs(self):
        try:
            if not hasattr(self, '_tk_root'):
                self._tk_root = tk.Tk()
                self._tk_root.withdraw()

            root = self._tk_root
            selected_exp = [None]

            def select_experiment():
                from tkinter import Toplevel, Listbox, Button, Label
                sel_window = Toplevel(root)
                sel_window.title("Select Experiment")
                sel_window.geometry("300x400")

                label = Label(sel_window, text="Select an experiment:")
                label.pack(padx=5, pady=5)

                listbox = Listbox(sel_window, height=15)
                listbox.pack(padx=5, pady=5, fill="both", expand=True)

                for exp_name in self.exp_list:
                    listbox.insert("end", exp_name)

                def confirm_selection():
                    selection = listbox.curselection()
                    if selection:
                        selected_exp[0] = listbox.get(selection[0])
                    sel_window.destroy()

                confirm_btn = Button(sel_window, text="Select", command=confirm_selection)
                confirm_btn.pack(pady=5)

                sel_window.transient(root)
                sel_window.wait_window()

            select_experiment()
            exp_name = selected_exp[0]

            if exp_name is None:
                return (None, None, None)

            self.save_dir = None
            self.save_dir_ready = False

            experiment_id = simpledialog.askstring("Experiment ID", "Enter experiment ID:", parent=root)
            if experiment_id is None:
                return (None, None, None)

            mouse_id = simpledialog.askstring("Mouse ID", "Enter mouse ID:", parent=root)
            if mouse_id is None:
                return (None, None, None)

            return (exp_name, experiment_id, mouse_id)

        except Exception as e:
            print(f"Dialog Error: {e}.")
            return (None, None, None)

    def run_exp(self, exp_name, experiment_id, mouse_id):
        try:
            self.exp_name = exp_name
            self.experiment_id = experiment_id
            self.mouse_id = mouse_id
            self.filename = f"{mouse_id}_{experiment_id}"
            self._prepare_save_directory(mouse_id, experiment_id)

            if self.debug_skip_teensy:
                self.experiment_status_callback("DEBUG_SKIP_TEENSY enabled: skipping logic analyzer.")
            else:
                self.start_logic_analyzer(experiment_id, mouse_id, self.filename)
            self.start_stim(exp_name, experiment_id, mouse_id,
                                self.experiment_status_callback,
                                self.experiment_trial_callback)

        except Exception as e:
            print(f'\nError in experiment selection: {e}.')
            self.experiment_status_callback(f"Error in experiment selection: {e}.")
        finally:
            self.exp_thread = None

    def _prepare_save_directory(self, mouse_id, experiment_id):
        save_root = load_save_root()
        save_dir = os.path.join(save_root, mouse_id, experiment_id)
        os.makedirs(save_dir, exist_ok=True)

        cam_config_source = Path(CONFIG_FILE)
        cam_config_target = Path(save_dir) / 'cam_config.yaml'
        if cam_config_source.is_file():
            shutil.copyfile(cam_config_source, cam_config_target)
        else:
            print("Warning: {} not found; skipping config copy.".format(cam_config_source))

        self.save_dir = save_dir
        self.save_dir_ready = True

    def start_logic_analyzer(self, experiment_id, mouse_id, base_filename=None):
        self.stop_logic_analyzer()

        print("\nStarting logic analyzer...")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        sigrok_script = os.path.join(script_dir, "continuous_sigrok.py")

        cmd = [sys.executable, "-u", sigrok_script, experiment_id, mouse_id]
        if base_filename:
            cmd.append(base_filename)

        self.logic_progress = subprocess.Popen(cmd,
            stdout = subprocess.PIPE, stderr = subprocess.STDOUT, stdin = subprocess.PIPE, text=True,
            cwd=str(REPO_ROOT))

        self.logic_thread = threading.Thread(target=self.track_logic, daemon=True)
        self.logic_thread.start()

    def stop_logic_analyzer(self, from_stop_stim=False):
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
                    if not from_stop_stim and callable(self.on_logic_analyzer_terminated):
                        self.on_logic_analyzer_terminated()
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

    def start_stim(self, exp_name, experiment_id, mouse_id, status_callback=None, trial_callback=None):
        if getattr(self, 'stim_progress', None):
            print("Experiment currently running. Stopping...")
            if hasattr(self.stim_progress, 'terminate'):
                self.stim_progress.terminate()
            self.stop_stim()

        print(f"\nStarting {exp_name} with exp. ID {experiment_id} and mouse {mouse_id}...")

        self.status_callback = status_callback
        self.trial_callback = trial_callback

        script_dir = os.path.dirname(os.path.abspath(__file__))
        wf_script = os.path.join(script_dir, "wf_main.py")
        cmd = [sys.executable, "-u", wf_script, exp_name, experiment_id, mouse_id]
        if self.debug_skip_teensy:
            cmd.append("--skip-teensy")
            print("DEBUG_SKIP_TEENSY enabled: running experiment without Teensy.")

        self.stim_progress = subprocess.Popen(cmd,
            stdout = subprocess.PIPE, stderr = subprocess.STDOUT, stdin = subprocess.PIPE, text=True,
            cwd=str(REPO_ROOT))
        threading.Thread(target=self.track_stim, daemon=True).start()
        print("\nStarted stim via subprocess.")

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
                
            self.stim_progress = None

        if self.saving:
            print("Stopping frame saving and finalizing metadata...")
            self.saving = False
            if self.hCamera:
                try:
                    mvsdk.CameraSetTriggerMode(self.hCamera, 0)
                except mvsdk.CameraException as e:
                    print("Failed to switch camera back to continuous mode after stopping experiment: {}".format(e))

        self.stop_logic_analyzer(from_stop_stim=True)

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

            if self.stim_progress.poll() not in (0, None):
                exit_code = self.stim_progress.returncode
                msg = f"Experiment subprocess exited with code {exit_code}."
                print(msg)
                if hasattr(self, 'status_callback') and self.status_callback:
                    self.status_callback(msg)
            
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

    def _write_metadata(self):
        if self.session_frames_written <= 0 or not self.save_dir_ready or self.save_dir is None:
            return

        metadata = {
            'num_frames': self.session_frames_written,
            'frame_width': self.width if not self.bin_exp else self.width//self.bin_size,
            'frame_height': self.height if not self.bin_exp else self.height//self.bin_size,
            'data_type': self.dtype if not self.bin_exp else 'uint16',
            'frame_timestamps': self.session_frame_timestamps,
            'sys_clock_timestamps': self.session_sys_clock_timestamps,
            'frame_exposure': self.exposure,
            'frame_gain': self.analog_gain,
            'binned_live': self.bin_exp or self.hardware_bin_enabled,
            'bin_mode': self.bin_mode if (self.bin_exp or self.hardware_bin_enabled) else 'none',
            'bin_size': self.bin_size,
        }
        np.save(os.path.join(self.save_dir, '{}.npy'.format(self.filename)), metadata)

    def _reset_save_session_metadata(self):
        self.session_frames_written = 0
        self.session_frame_timestamps = []
        self.session_sys_clock_timestamps = []

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

            # Process entire batch at once using vectorized 3D operations.
            n = len(write_batch)
            batch = np.frombuffer(b''.join(write_batch), dtype=self.dtype).reshape((n, self.height, self.width))
            batch = np.ascontiguousarray(batch[:, :, ::-1])  # horizontal flip
            if self.bin_exp:
                bs = self.bin_size
                if bs in (2, 4, 8, 16, 32, 64):
                    # Faster path for power-of-two bin sizes via iterative 2x2 reductions.
                    batch = batch.astype(np.uint16, copy=False)
                    for _ in range(bs.bit_length() - 1):
                        batch = (
                            batch[:, 0::2, 0::2]
                            + batch[:, 1::2, 0::2]
                            + batch[:, 0::2, 1::2]
                            + batch[:, 1::2, 1::2]
                        )
                else:
                    batch = batch.reshape(n, self.height // bs, bs, self.width // bs, bs).sum(axis=(2, 4), dtype=np.uint16)
            batch.tofile(self.save_file_handle)

            self.frame_timestamps.extend(ts_batch)
            self.sys_clock_timestamps.extend(sys_ts_batch)
            self.session_frame_timestamps.extend(ts_batch)
            self.session_sys_clock_timestamps.extend(sys_ts_batch)
            self.frames_written += len(write_batch)
            self.session_frames_written += len(write_batch)

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
                while not self.frame_queue.empty():
                    try:
                        frame_data, count, timestamp, sys_stamp = self.frame_queue.get_nowait()
                    except queue.Empty:
                        break

                    write_batch.append(frame_data)
                    ts_batch.append(timestamp)
                    sys_ts_batch.append(sys_stamp)
                    if len(write_batch) >= flush_every_n_frames:
                        flush_batch(force_flush=False)

                flush_batch(force_flush=True)
                if self.save_file_handle is not None and self.frame_queue.empty():
                    self.save_file_handle.flush()
                    self.save_file_handle.close()
                    self.save_file_handle = None
                    self._write_metadata()
                    self._reset_save_session_metadata()
                if self.quit:
                    break
                time.sleep(0.005)

        if self.save_file_handle is not None:
            flush_batch(force_flush=True)
            self.save_file_handle.flush()
            self.save_file_handle.close()
            self.save_file_handle = None
            self._write_metadata()
            self._reset_save_session_metadata()

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
        if not self.gui_mode:
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

        self._probe_camera_binning_support(cap, monoCamera)

        self.height = cap.sResolutionRange.iHeightMax
        self.width = cap.sResolutionRange.iWidthMax

        self._apply_camera_binning(cap, monoCamera)

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

        _stats_written = False
        # main loop to print info from the camera
        while not self.quit:
            current_time = time.time()
            elapsed_time = current_time - self.t_start
            average_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0

            # Print stats, reusing the same terminal line
            msg = "Save Q: {}, Frames Saved: {}, Save Drop: {}, Disp Q: {}, Disp Drop: {}, Frames Disp: {}, Average FPS: {:.2f} Sat Pixels: {:06d}".format(
                self.frame_queue.qsize(), self.frames_written, self.dropped_save_frames, self.display_queue.qsize(), self.dropped_display_frames, self.frame_count, average_fps, self.n_saturated_pixels)
            if _stats_written:
                sys.stdout.write('\033[F\033[2K' + msg + '\n')
            else:
                sys.stdout.write(msg + '\n')
                _stats_written = True
            sys.stdout.flush()
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


        if self.saving:
            frame_timestamp = time.time()
            if not self._enqueue_save_frame(frame_data, self.frame_count, FrameHead.uiTimeStamp, frame_timestamp):
                return
        # Stop addding to the display queue if the frame queue is getting too full
        if self.frame_queue.qsize() < 0.95 * self.frame_queue.maxsize:
            self._put_display_frame(frame_data)
        self.frame_count += 1

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
        plt.close('all')

if __name__ == '__main__':
    main()
