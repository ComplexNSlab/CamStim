# coding=utf-8
import ctypes
import sys
import numpy as np
from core import mvsdk
import time
import platform
import queue
import threading
import yaml
from pathlib import Path
import os
import subprocess
import shutil

REPO_ROOT = Path(__file__).resolve().parent.parent

CONFIG_DIR = REPO_ROOT / 'config_files'

_VERSION_FILE = REPO_ROOT / 'VERSION'


def _load_app_version(default='0.0.0'):
    try:
        version_text = _VERSION_FILE.read_text(encoding='utf-8').strip()
        return version_text if version_text else default
    except OSError:
        return default


APP_VERSION = _load_app_version()


def _inject_version_into_yaml(yaml_path, version=APP_VERSION):
    """Prepend 'VERSION: <version>' to a YAML file if no VERSION key already exists."""
    path = Path(yaml_path)
    if not path.is_file():
        return
    try:
        text = path.read_text(encoding='utf-8')
        # Check whether a VERSION key already exists at the start of any line.
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith('VERSION') and ':' in stripped:
                return  # Already present – leave the file untouched.
        path.write_text(f'VERSION: {version}\n' + text, encoding='utf-8')
    except OSError as exc:
        print(f'Warning: could not inject VERSION into {yaml_path}: {exc}')
CONFIG_FILE = str(CONFIG_DIR / 'cam_config.yaml')
TEENSY_PARAMS_FILE = str(CONFIG_DIR / 'teensyParams.yaml')
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
    def __init__(self, config, frame_output_queue=None, command_queue=None, status_queue=None):
        super(App, self).__init__()

        self.config = config
        debug_skip_teensy_cfg = config.get('DEBUG_SKIP_TEENSY', False)
        if isinstance(debug_skip_teensy_cfg, str):
            self.debug_skip_teensy = debug_skip_teensy_cfg.strip().lower() in ('1', 'true', 'yes', 'on')
        else:
            self.debug_skip_teensy = bool(debug_skip_teensy_cfg)
        self.quit = False
        self.acquiring = False
        self.saving = False
        self.software_mirror_horizontal = False
        self.mirror_enabled = False
        self.mirror_flip_flags = 1
        self._mirror_runtime_warning_emitted = False
        self.save_queue_max_frames = int(config.get('SAVE_QUEUE_MAX_FRAMES', 2000))
        self.frame_queue = queue.Queue(maxsize=self.save_queue_max_frames)  # Buffer for save path
        self.save_thread = threading.Thread(target=self.save_frames)  # Thread for saving frames
        self.frame_count = 0  # To keep track of saved 
        self.frames_written = 0
        self.dropped_save_frames = 0
        self.save_overflow = False
        self.save_batch_frames = int(config.get('SAVE_BATCH_FRAMES', 64))
        # Cap batch memory to avoid periodic large allocations that can stall writes.
        self.save_target_batch_bytes = int(config.get('SAVE_TARGET_BATCH_BYTES', 32 * 1024 * 1024))
        self.live_speck = config['USE_LIVE_SPECKLE']
        self.exposure = config['EXPOSURE_TIME'] # in ms
        self.analog_gain = float(config['ANALOG_GAIN'])
        self.analog_gain_step = 1.0
        self.analog_gain_min_units = 1
        self.analog_gain_max_units = 100

        self.filename = str(config.get('EXPERIMENT', 'recording'))
        self.bin_exp = bool(config['BIN_EXP_LIVE'])
        self.bin_size = config['BIN_SIZE']
        self.frame_timestamps = []  # List to store timestamps
        self.sys_clock_timestamps = []
        self.session_frame_timestamps = []
        self.session_sys_clock_timestamps = []
        self.USE_MONO16 = False
        self.bytes_per_pixel = 1
        self.t_start = None
        self.hCamera = None
        self.session_frames_written = 0
        self.on_logic_analyzer_terminated = None

        self.exp_thread = None
        self.stim_thread = None
        self.logic_thread = None
        self.stim_progress = None
        self.logic_progress = None

        self.save_dir = None
        self.save_dir_ready = False
        self.save_file_handle = None
        self.exp_status_queue = queue.Queue()
        self.frame_output_queue = frame_output_queue
        self.command_queue = command_queue
        self.status_queue = status_queue
        self.ready_state_published = False
        self.latest_frame_data = None
        self.latest_frame_lock = threading.Lock()
        self._windows_camera_thread_priority_boosted = False
        self._windows_callback_thread_priority_boosted = False

        # self.check_and_fix_existing_experiment()

        self.dtype = 'uint8'

    def get_frame_for_display(self):
        return self.get_latest_frame()

    def _queue_put_latest(self, q, item):
        if q is None:
            return
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass

    def _publish_status(self, payload):
        if self.status_queue is None:
            return

        try:
            self.status_queue.put_nowait(payload)
        except queue.Full:
            # Status queue is typically unbounded; if bounded, drop the oldest and retry.
            try:
                self.status_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.status_queue.put_nowait(payload)
            except queue.Full:
                pass

    def _publish_ready_state(self):
        if self.ready_state_published:
            return
        if not hasattr(self, 'width') or not hasattr(self, 'height'):
            return

        self.ready_state_published = True
        gain_min = self.analog_gain_min_units * self.analog_gain_step
        gain_max = self.analog_gain_max_units * self.analog_gain_step
        self._publish_status({
            'type': 'ready',
            'width': int(self.width),
            'height': int(self.height),
            'dtype': self.dtype,
            'bin_exp': bool(self.bin_exp),
            'bin_size': int(self.bin_size),
            'analog_gain_step': float(self.analog_gain_step),
            'analog_gain_min': float(gain_min),
            'analog_gain_max': float(gain_max),
            'analog_gain': float(self.analog_gain),
        })

    def _handle_command(self, command):
        if not command:
            return

        if isinstance(command, tuple):
            name = command[0]
            payload = command[1] if len(command) > 1 else None
        elif isinstance(command, dict):
            name = command.get('name')
            payload = command.get('payload')
        else:
            return

        try:
            if name == 'stop_camera':
                self.stop_stim()
                self.quit = True
            elif name == 'set_trigger_mode':
                mode = int(payload)
                if self.hCamera:
                    mvsdk.CameraSetTriggerMode(self.hCamera, mode)
                self.saving = (mode == 2)
                self._publish_status({'type': 'trigger_mode', 'mode': mode})
            elif name == 'set_exposure':
                exposure_ms = float(payload)
                self.exposure = exposure_ms
                if self.hCamera:
                    mvsdk.CameraSetExposureTime(self.hCamera, exposure_ms * 1000)
                self._publish_status({'type': 'exposure', 'value': exposure_ms})
            elif name == 'set_gain':
                gain = float(payload)
                applied_gain = self._apply_analog_gain_multiplier(gain)
                self._publish_status({'type': 'gain', 'value': applied_gain})
            elif name == 'set_saving':
                self.saving = bool(payload)
                self._publish_status({'type': 'saving', 'value': self.saving})
            elif name == 'preview_experiment':
                exp_name, experiment_id, mouse_id = payload
                self.get_exp_params(exp_name=exp_name, experiment_id=experiment_id, mouse_id=mouse_id)
            elif name == 'start_experiment':
                exp_name, experiment_id, mouse_id = payload
                self.saving = True
                self.get_exp_params(exp_name=exp_name, experiment_id=experiment_id, mouse_id=mouse_id)
            elif name == 'stop_preview':
                self.stop_stim()
            elif name == 'stop_experiment':
                self.saving = False
                self.stop_stim()
        except Exception as e:
            self._publish_status({'type': 'error', 'message': str(e)})

    def _process_command_queue(self):
        if self.command_queue is None:
            return

        while True:
            try:
                command = self.command_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_command(command)

    def _boost_windows_camera_thread_priority(self):
        if platform.system() != 'Windows' or self._windows_camera_thread_priority_boosted:
            return

        try:
            kernel32 = ctypes.windll.kernel32
            thread_handle = kernel32.GetCurrentThread()
            high_thread_priority = 2
            if not kernel32.SetThreadPriority(thread_handle, high_thread_priority):
                raise ctypes.WinError(ctypes.get_last_error())
            self._windows_camera_thread_priority_boosted = True
            print('Windows camera thread priority raised to HIGH.')
        except Exception as e:
            print(f'Warning: could not raise Windows camera thread priority: {e}')

    def _boost_windows_callback_thread_priority(self):
        if platform.system() != 'Windows' or self._windows_callback_thread_priority_boosted:
            return

        try:
            kernel32 = ctypes.windll.kernel32
            thread_handle = kernel32.GetCurrentThread()
            highest_thread_priority = 2
            if not kernel32.SetThreadPriority(thread_handle, highest_thread_priority):
                raise ctypes.WinError(ctypes.get_last_error())
            self._windows_callback_thread_priority_boosted = True
        except Exception as e:
            print(f'Warning: could not raise Windows callback thread priority: {e}')

    def _put_display_frame(self, frame_data, frame_index):
        with self.latest_frame_lock:
            self.latest_frame_data = frame_data

        if self.frame_output_queue is not None:
            self._queue_put_latest(self.frame_output_queue, frame_data)


    def get_latest_frame(self):
        with self.latest_frame_lock:
            if self.latest_frame_data is None:
                return None
            return self.latest_frame_data

    def _enqueue_save_frame(self, frame_data, frame_index, camera_timestamp, system_timestamp):
        queued_frame = (frame_data, frame_index, camera_timestamp, system_timestamp)
        try:
            # Never block the camera callback thread; callback stalls can cause SDK-level drops.
            self.frame_queue.put_nowait(queued_frame)
            return True
        except queue.Full:
            self.save_overflow = True
            return False

    def _ensure_unique_filename(self):
        if not self.save_dir_ready or self.save_dir is None:
            return

        base_name = self.filename
        candidate = base_name
        suffix = 1
        while True:
            bin_path = Path(self.save_dir) / f"{candidate}.bin"
            meta_path = Path(self.save_dir) / f"{candidate}.npy"
            if not bin_path.exists() and not meta_path.exists():
                break
            candidate = f"{base_name}_{suffix}"
            suffix += 1

        if candidate != base_name:
            print(f"Warning: save file exists for '{base_name}'. Using '{candidate}' instead.")
            self.filename = candidate

    def _configure_analog_gain_scale(self, cap):
        expose = cap.sExposeDesc
        self.analog_gain_step = float(expose.fAnalogGainStep) if float(expose.fAnalogGainStep) > 0 else 1.0
        self.analog_gain_min_units = int(expose.uiAnalogGainMin)
        self.analog_gain_max_units = int(expose.uiAnalogGainMax)

        gain_min = self.analog_gain_min_units * self.analog_gain_step
        gain_max = self.analog_gain_max_units * self.analog_gain_step
        print(
            f"Analog gain scale: step={self.analog_gain_step} "
            f"units=[{self.analog_gain_min_units}, {self.analog_gain_max_units}] , "
            f"multiplier=[{gain_min}, {gain_max}]"
        )

    def _gain_multiplier_to_units(self, gain_multiplier):
        units = int(round(float(gain_multiplier) / self.analog_gain_step))
        return max(self.analog_gain_min_units, min(self.analog_gain_max_units, units))

    def _gain_units_to_multiplier(self, gain_units):
        return float(gain_units) * self.analog_gain_step

    def _apply_analog_gain_multiplier(self, gain_multiplier):
        units = self._gain_multiplier_to_units(gain_multiplier)
        if self.hCamera:
            mvsdk.CameraSetAnalogGain(self.hCamera, units)
        self.analog_gain = self._gain_units_to_multiplier(units)
        return self.analog_gain

    def _apply_output_bit_depth(self, mono_camera):
        if not mono_camera:
            raise RuntimeError("Only monochrome cameras are supported in this application.")

        # Mono8 is enforced for stability and compatibility.
        mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
        self.USE_MONO16 = False
        self.bytes_per_pixel = 1
        self.dtype = 'uint8'
        print("Using 8-bit output format (MONO8).")

        # Mirror handling for frame orientation consistency across display/save.
        # Uses CameraFlipFrameBuffer in the callback path.
        mirror_cfg = self.config.get('CAMERA_MIRROR_HORIZONTAL', False)
        if isinstance(mirror_cfg, str):
            mirror_h = mirror_cfg.strip().lower() in ('1', 'true', 'yes', 'on')
        else:
            mirror_h = bool(mirror_cfg)

        mirror_flip_flags_cfg = self.config.get('CAMERA_MIRROR_FLIP_FLAGS', 1)
        try:
            mirror_flip_flags = int(mirror_flip_flags_cfg)
        except (TypeError, ValueError):
            mirror_flip_flags = 1

        self.mirror_enabled = mirror_h
        self.mirror_flip_flags = mirror_flip_flags
        self.software_mirror_horizontal = False

        if not mirror_h:
            print("Horizontal mirror disabled (CAMERA_MIRROR_HORIZONTAL=false).")
            return

        print(f"Horizontal mirror enabled via CameraFlipFrameBuffer (flags={self.mirror_flip_flags}).")

    def experiment_status_callback(self, message):
        if hasattr(self, 'exp_status_queue'):
            self.exp_status_queue.put(("status", message))
        self._publish_status({'type': 'status', 'message': message})
        print(f"[Experiment] {message}")

    def experiment_trial_callback(self, current, total, message):
        if hasattr(self, 'exp_status_queue'):
            self.exp_status_queue.put(("trial", current, total, message))
        self._publish_status({'type': 'trial', 'current': current, 'total': total, 'message': message})
        print(message)

    def get_exp_params(self, exp_name=None, experiment_id=None, mouse_id=None):
        if hasattr(self, 'exp_thread') and self.exp_thread and self.exp_thread.is_alive():
            print('\nExperiment selection already in progress.')
            return False

        if exp_name is None or experiment_id is None or mouse_id is None:
            print("Missing experiment parameters (exp_name, experiment_id, mouse_id).")
            return False

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
            _inject_version_into_yaml(cam_config_target)
        else:
            print("Warning: {} not found; skipping config copy.".format(cam_config_source))

        self.save_dir = save_dir
        self.save_dir_ready = True
        self._ensure_unique_filename()

    def _copy_teensy_params_to_save_dir(self):
        if not self.save_dir_ready or self.save_dir is None:
            return

        teensy_params_source = Path(TEENSY_PARAMS_FILE)
        teensy_params_target = Path(self.save_dir) / 'teensyParams.yaml'
        if teensy_params_source.is_file():
            shutil.copyfile(teensy_params_source, teensy_params_target)
            _inject_version_into_yaml(teensy_params_target)
        else:
            print("Warning: {} not found; skipping config copy.".format(teensy_params_source))

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
                    if not from_stop_stim:
                        self._publish_status({'type': 'logic_analyzer_terminated'})
                else:
                    print("Logic analyzer process has already finished.")
                    if not from_stop_stim:
                        self._publish_status({'type': 'logic_analyzer_terminated'})
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
                    self._publish_status({'type': 'logic_analyzer_terminated'})

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

            if exp_completed:
                self._publish_status({'type': 'experiment_finished'})

            self._copy_teensy_params_to_save_dir()

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

    def _write_metadata(self):
        if self.session_frames_written <= 0 or not self.save_dir_ready or self.save_dir is None:
            return

        camera_gap_stats = self._analyze_timestamp_gaps(self.session_frame_timestamps)
        system_gap_stats = self._analyze_timestamp_gaps(self.session_sys_clock_timestamps)

        estimated_total_triggered_frames = self.session_frames_written + camera_gap_stats['estimated_missing_frames']

        metadata = {
            'num_frames': self.session_frames_written,
            'frame_width': self.width if not self.bin_exp else self.width//self.bin_size,
            'frame_height': self.height if not self.bin_exp else self.height//self.bin_size,
            'data_type': self.dtype if not self.bin_exp else 'uint16',
            'frame_timestamps': self.session_frame_timestamps,
            'sys_clock_timestamps': self.session_sys_clock_timestamps,
            'frame_exposure': self.exposure,
            'frame_gain': self.analog_gain,
            'binned_live': self.bin_exp,
            'bin_mode': 'software' if self.bin_exp else 'none',
            'bin_size': self.bin_size,
            'timestamp_gap_analysis': {
                'camera_timestamp': camera_gap_stats,
                'system_timestamp': system_gap_stats,
                'estimated_total_triggered_frames': estimated_total_triggered_frames,
            },
        }
        np.save(os.path.join(self.save_dir, '{}.npy'.format(self.filename)), metadata)

        cam_gap = metadata['timestamp_gap_analysis']['camera_timestamp']
        summary = (
            f"Timestamp gap summary: expected_dt={cam_gap['expected_delta']}, "
            f"max_dt={cam_gap['max_delta']}, gap_events={cam_gap['gap_events']}, "
            f"estimated_missing_frames={cam_gap['estimated_missing_frames']}, "
            f"saved_frames={self.session_frames_written}, "
            f"estimated_total_triggered={metadata['timestamp_gap_analysis']['estimated_total_triggered_frames']}"
        )
        print(summary)
        try:
            self.experiment_status_callback(summary)
        except Exception as exc:
            # Status publishing can fail during shutdown if IPC handles are already closed.
            print(f"Warning: failed to publish metadata summary status: {exc}")

    def _analyze_timestamp_gaps(self, timestamps):
        """Estimate missing frames from timestamp gaps using a robust expected interval."""
        if timestamps is None or len(timestamps) < 3:
            return {
                'num_timestamps': 0 if timestamps is None else int(len(timestamps)),
                'expected_delta': None,
                'max_delta': None,
                'gap_events': 0,
                'estimated_missing_frames': 0,
            }

        ts = np.asarray(timestamps, dtype=np.float64)
        diffs = np.diff(ts)
        diffs = diffs[diffs > 0]

        if diffs.size == 0:
            return {
                'num_timestamps': int(len(timestamps)),
                'expected_delta': None,
                'max_delta': None,
                'gap_events': 0,
                'estimated_missing_frames': 0,
            }

        expected_delta = float(np.median(diffs))
        if expected_delta <= 0:
            return {
                'num_timestamps': int(len(timestamps)),
                'expected_delta': None,
                'max_delta': float(np.max(diffs)),
                'gap_events': 0,
                'estimated_missing_frames': 0,
            }

        # Treat deltas larger than 1.5x expected as likely containing at least one missing frame.
        gap_mask = diffs > (1.5 * expected_delta)
        gap_diffs = diffs[gap_mask]

        if gap_diffs.size == 0:
            estimated_missing_frames = 0
        else:
            estimated_missing_frames = int(np.maximum(0, np.rint(gap_diffs / expected_delta).astype(np.int64) - 1).sum())

        return {
            'num_timestamps': int(len(timestamps)),
            'expected_delta': expected_delta,
            'max_delta': float(np.max(diffs)),
            'gap_events': int(gap_diffs.size),
            'estimated_missing_frames': estimated_missing_frames,
        }

    def _reset_save_session_metadata(self):
        self.session_frames_written = 0
        self.session_frame_timestamps = []
        self.session_sys_clock_timestamps = []

    def _close_save_file_handle(self):
        if self.save_file_handle is None:
            return

        try:
            if not getattr(self.save_file_handle, 'closed', True):
                self.save_file_handle.flush()
                self.save_file_handle.close()
        except (OSError, ValueError) as exc:
            print(f"Warning: save file handle was already closed during teardown: {exc}")
        finally:
            self.save_file_handle = None

    def _finalize_save_session(self):
        try:
            self._write_metadata()
        except (OSError, ValueError) as exc:
            print(f"Warning: failed to write session metadata: {exc}")
        finally:
            self._close_save_file_handle()
            self._reset_save_session_metadata()

    def save_frames(self):
        last_flush_time = time.time()
        flush_interval_s = 0.5
        flush_every_n_frames = max(1, self.save_batch_frames)
        last_reported_batch = None
        write_batch = []
        ts_batch = []
        sys_ts_batch = []

        def flush_batch(force_flush=False):
            nonlocal last_flush_time
            if not write_batch or self.save_file_handle is None or getattr(self.save_file_handle, 'closed', False):
                return

            n = len(write_batch)
            h, w = int(self.height), int(self.width)

            # b''.join: single C-level allocation + memcpy.
            # frombuffer: zero-copy view.
            raw = np.frombuffer(b''.join(write_batch), dtype=self.dtype).reshape(n, h, w)

            if self.software_mirror_horizontal:
                # Fallback path when CameraFlipFrameBuffer fails at runtime.
                raw = np.ascontiguousarray(raw[:, :, ::-1])

            if self.bin_exp:
                bs = self.bin_size
                data = raw.astype(np.uint16, copy=False)
                if bs in (2, 4, 8, 16, 32, 64):
                    for _ in range(bs.bit_length() - 1):
                        data = (
                            data[:, 0::2, 0::2]
                            + data[:, 1::2, 0::2]
                            + data[:, 0::2, 1::2]
                            + data[:, 1::2, 1::2]
                        )
                else:
                    data = data.reshape(n, h // bs, bs, w // bs, bs).sum(axis=(2, 4), dtype=np.uint16)
                write_slice = data
            else:
                write_slice = raw  # contiguous, no extra copy

            try:
                write_slice.tofile(self.save_file_handle)
            except (OSError, ValueError) as exc:
                print(f"Warning: failed to write frame batch to save file: {exc}")
                self._close_save_file_handle()
                write_batch.clear(); ts_batch.clear(); sys_ts_batch.clear()
                return

            self.frame_timestamps.extend(ts_batch)
            self.sys_clock_timestamps.extend(sys_ts_batch)
            self.session_frame_timestamps.extend(ts_batch)
            self.session_sys_clock_timestamps.extend(sys_ts_batch)
            self.frames_written += n
            self.session_frames_written += n

            if force_flush or ((time.time() - last_flush_time) >= flush_interval_s):
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

                if hasattr(self, 'width') and hasattr(self, 'height'):
                    frame_bytes = max(1, int(self.width) * int(self.height) * int(self.bytes_per_pixel))
                    max_frames_by_bytes = max(1, self.save_target_batch_bytes // frame_bytes)
                    effective_batch = max(1, min(self.save_batch_frames, max_frames_by_bytes))
                    if effective_batch != flush_every_n_frames:
                        flush_every_n_frames = effective_batch
                    if last_reported_batch != flush_every_n_frames:
                        print(f"Using effective save batch size: {flush_every_n_frames} frame(s) (~{flush_every_n_frames * frame_bytes / (1024*1024):.1f} MiB raw)")
                        last_reported_batch = flush_every_n_frames

                if self.save_file_handle is None:
                    file_path = os.path.join(self.save_dir, self.filename + '.bin')
                    if os.path.exists(file_path):
                        self._ensure_unique_filename()
                        file_path = os.path.join(self.save_dir, self.filename + '.bin')
                    # Large Python-level buffer reduces syscall overhead for fast cameras.
                    self.save_file_handle = open(file_path, 'wb', buffering=64 * 1024 * 1024)

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
                    self._finalize_save_session()
                if self.quit:
                    break
                time.sleep(0.005)

        if self.save_file_handle is not None:
            flush_batch(force_flush=True)
            self._finalize_save_session()

    def print_camera_stats(self):
        print("\nExposure(ms): {} Gain: {} ".format(self.exposure, self.analog_gain))

    def main(self):
        # Enumerate cameras
        self._boost_windows_camera_thread_priority()

        DevList = mvsdk.CameraEnumerateDevice()
        nDev = len(DevList)
        if nDev < 1:
            print("No camera was found!")
            return

        for i, DevInfo in enumerate(DevList):
            print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
        i = 0 if nDev == 1 else int(input("Select camera: "))
        DevInfo = DevList[i]

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
        if not monoCamera:
            print("This application supports monochrome cameras only.")
            mvsdk.CameraUnInit(self.hCamera)
            return

        self.height = cap.sResolutionRange.iHeightMax
        self.width = cap.sResolutionRange.iWidthMax
        self._configure_analog_gain_scale(cap)

        # For monochrome cameras, output configured bit depth directly.
        self._apply_output_bit_depth(monoCamera)

        # Read back actual active resolution after ISP/output setup.
        applied_res = mvsdk.CameraGetImageResolution(self.hCamera)
        self.width = applied_res.iWidth
        self.height = applied_res.iHeight

        # Switch camera mode to continuous capture
        mvsdk.CameraSetTriggerMode(self.hCamera, 0)
        # Switch the camera to full speed transmission
        mvsdk.CameraSetFrameSpeed(self.hCamera, 1)
        # Manual exposure, exposure time 
        mvsdk.CameraSetAeState(self.hCamera, 0)
        mvsdk.CameraSetExposureTime(self.hCamera, self.exposure * 1000)
        self._apply_analog_gain_multiplier(self.analog_gain)
        self._publish_ready_state()

        print(f"Camera resolution: {self.width}x{self.height}")

        if self.live_speck:
            self.setup_live_speckle_variables()

        # Let the SDK's internal image capture thread start working
        mvsdk.CameraPlay(self.hCamera)

        # Allocate ISP output buffer for worst-case mono frame size at current bit depth.
        FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * self.bytes_per_pixel

        # Set the capture callback function
        self.quit = False
        mvsdk.CameraSetCallbackFunction(self.hCamera, self.GrabCallback, 0)
        self.print_camera_stats()
        time.sleep(1)

        _stats_written = False
        # main loop to print info from the camera
        while not self.quit:
            self._process_command_queue()
            current_time = time.time()
            elapsed_time = current_time - self.t_start
            average_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0

            self._publish_status({
                'type': 'stats',
                'frame_count': self.frame_count,
                'frames_written': self.frames_written,
                'save_queue_size': self.frame_queue.qsize(),
                'display_queue_size': 0,
            })

            # Print stats, reusing the same terminal line
            msg = "Save Q: {}, Frames Saved: {}, Save Drop: {}, Frames Disp: {}, Average FPS: {:.2f}".format(
                self.frame_queue.qsize(), self.frames_written, self.dropped_save_frames, self.frame_count, average_fps)
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
    @mvsdk.method(mvsdk.CAMERA_SNAP_PROC)
    def GrabCallback(self, hCamera, pRawData, pFrameHead, pContext):
        if self.quit:
            #print("Returning without adding frames to the list")
            return

        self._boost_windows_callback_thread_priority()

        current_time = time.time()
        FrameHead = pFrameHead[0]

        if self.mirror_enabled:
            flip_err = mvsdk.CameraFlipFrameBuffer(pRawData, FrameHead, self.mirror_flip_flags)
            if flip_err != 0 and not self._mirror_runtime_warning_emitted:
                self._mirror_runtime_warning_emitted = True
                print(
                    f"Warning: CameraFlipFrameBuffer failed (err={flip_err}, flags={self.mirror_flip_flags}); "
                    "falling back to software mirror."
                )
                self.software_mirror_horizontal = True

        frame_data = bytes((mvsdk.c_ubyte * FrameHead.uBytes).from_address(pRawData))
        mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

        if not self.acquiring:
            self.acquiring = True
            self.t_start = time.time()


        if self.saving:
            frame_timestamp = time.time()
            if not self._enqueue_save_frame(frame_data, self.frame_count, FrameHead.uiTimeStamp, frame_timestamp):
                self.dropped_save_frames += 1
                return
        # Stop addding to the display queue if the frame queue is getting too full
        if self.frame_queue.qsize() < 0.95 * self.frame_queue.maxsize:
            self._put_display_frame(frame_data, self.frame_count)
        self.frame_count += 1

def main():
    raise SystemExit("Standalone simple_cam_mx execution is disabled. Launch camstim.py instead.")


def run_camera_worker(config, frame_output_queue=None, command_queue=None, status_queue=None):
    app = App(config, frame_output_queue=frame_output_queue, command_queue=command_queue, status_queue=status_queue)
    app.save_thread.start()
    app.main()

if __name__ == '__main__':
    main()
