import sys
import numpy as np
import multiprocessing as mp
import queue
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
							QWidget, QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
							QGroupBox, QTextEdit, QCheckBox, QComboBox, QLineEdit)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
import time
from collections import deque
from utils.simple_cam_mx import App, load_camera_config
import threading
from core import mvsdk
from core.TeensyController import TeensyController
from core.experiment_discovery import get_experiment_list
import cv2
import yaml
from pathlib import Path

from utils.simple_cam_mx import load_camera_config, run_camera_worker
from core.experiment_discovery import get_experiment_list

CONFIG_DIR = Path(__file__).resolve().parent / 'config_files'


class CameraProcessClient:
	def __init__(self, config):
		self.config = config
		self.frame_queue = None
		self.command_queue = None
		self.status_queue = None
		self.process = None
		self.latest_frame_data = None
		self.latest_frame_lock = threading.Lock()
		self.ready = False
		self.dtype = 'uint8'
		self.width = None
		self.height = None
		self.bin_exp = bool(config.get('BIN_EXP_LIVE'))
		self.bin_size = int(config.get('BIN_SIZE', 1))
		self.vmin = 0
		self.vmax = 30
		self.live_speck = bool(config.get('USE_LIVE_SPECKLE', False))
		self.enable_live_speckle = False
		self.removeBackground = False
		self.saving = False
		self.hardware_trigger_enabled = False
		self.normalizeImage = False
		self.dFoF_open = False
		self.F0 = None
		self.minI = 0
		self.maxI = 255
		self.backgroundImg = None
		self.buffer_size = int(config.get('BUFFER_SIZE', 50))
		self.circular_buffer = None
		self.current_buffer_item = 0
		self.buffer_loop_reached = False
		self.histogram_open = False
		self.frame_count = 0
		self.frames_written = 0
		self.save_queue_size = 0
		self.display_queue_size = 0
		self.exp_status_queue = queue.Queue()
		self.exp_list = get_experiment_list()
		self.exp_name = None
		self.experiment_id = None
		self.mouse_id = None
		self.on_logic_analyzer_terminated = None
		self.on_experiment_finished = None

	def start(self):
		ctx = mp.get_context('spawn')
		self.frame_queue = ctx.Queue(maxsize=1)
		self.command_queue = ctx.Queue()
		self.status_queue = ctx.Queue()
		self.process = ctx.Process(
			target=run_camera_worker,
			args=(self.config, self.frame_queue, self.command_queue, self.status_queue),
			daemon=True,
		)
		self.process.start()
		self._wait_for_ready()

	def _wait_for_ready(self, timeout=10.0):
		deadline = time.time() + timeout
		while time.time() < deadline:
			self.poll_messages()
			if self.ready:
				return True
			time.sleep(0.05)
		return self.ready

	def _send_command(self, name, payload=None):
		if self.command_queue is None:
			return False
		self.command_queue.put({'name': name, 'payload': payload})
		return True

	def poll_messages(self):
		if self.status_queue is None:
			return
		while True:
			try:
				msg = self.status_queue.get_nowait()
			except queue.Empty:
				break
			if not isinstance(msg, dict):
				continue
			msg_type = msg.get('type')
			if msg_type == 'ready':
				self.ready = True
				self.width = msg.get('width', self.width)
				self.height = msg.get('height', self.height)
				self.dtype = msg.get('dtype', self.dtype)
				self.bin_exp = bool(msg.get('bin_exp', self.bin_exp))
				self.bin_size = int(msg.get('bin_size', self.bin_size))
				if self.circular_buffer is None and self.width and self.height:
					self.circular_buffer = np.zeros((self.buffer_size, self.height // self.bin_size, self.width // self.bin_size), dtype=np.float32)
			elif msg_type == 'stats':
				self.frame_count = int(msg.get('frame_count', self.frame_count))
				self.frames_written = int(msg.get('frames_written', self.frames_written))
				self.save_queue_size = int(msg.get('save_queue_size', self.save_queue_size))
				self.display_queue_size = int(msg.get('display_queue_size', self.display_queue_size))
			elif msg_type == 'trigger_mode':
				self.hardware_trigger_enabled = msg.get('mode') == 2
			elif msg_type == 'exposure':
				self.exposure = msg.get('value', getattr(self, 'exposure', None))
			elif msg_type == 'gain':
				self.analog_gain = msg.get('value', getattr(self, 'analog_gain', None))
			elif msg_type == 'status':
				self.exp_status_queue.put(("status", msg.get('message', '')))
			elif msg_type == 'trial':
				self.exp_status_queue.put(("trial", msg.get('current', 0), msg.get('total', 0), msg.get('message', '')))
			elif msg_type == 'logic_analyzer_terminated':
				if callable(self.on_logic_analyzer_terminated):
					self.on_logic_analyzer_terminated()
			elif msg_type == 'experiment_finished':
				if callable(self.on_experiment_finished):
					self.on_experiment_finished()
			elif msg_type == 'error':
				self.exp_status_queue.put(("status", f"Camera worker error: {msg.get('message')}"))
			else:
				self.exp_status_queue.put(("status", str(msg)))

	def get_frame_for_display(self):
		self.poll_messages()
		latest = None
		if self.frame_queue is not None:
			while True:
				try:
					latest = self.frame_queue.get_nowait()
				except queue.Empty:
					break
		if latest is not None:
			with self.latest_frame_lock:
				self.latest_frame_data = latest
		with self.latest_frame_lock:
			return self.latest_frame_data

	def get_exp_status(self):
		self.poll_messages()
		messages = []
		try:
			while True:
				messages.append(self.exp_status_queue.get_nowait())
		except queue.Empty:
			pass
		return messages

	def set_trigger_mode(self, enabled):
		self.hardware_trigger_enabled = bool(enabled)
		return self._send_command('set_trigger_mode', 2 if enabled else 0)

	def set_exposure(self, value):
		return self._send_command('set_exposure', value)

	def set_gain(self, value):
		return self._send_command('set_gain', value)

	def set_saving(self, enabled):
		self.saving = bool(enabled)
		return self._send_command('set_saving', bool(enabled))

	def toggle_background_removal(self):
		self.removeBackground = not self.removeBackground
		return self.removeBackground

	def toggle_speckle(self):
		self.enable_live_speckle = not self.enable_live_speckle
		return self.enable_live_speckle

	def toggle_dFoF(self):
		self.dFoF_open = not self.dFoF_open
		return self.dFoF_open

	def toggle_histogram(self):
		self.histogram_open = not self.histogram_open
		return self.histogram_open

	def adjust_dynamic_range(self):
		self.normalizeImage = not self.normalizeImage
		return self.normalizeImage

	def get_exp_params(self, exp_name=None, experiment_id=None, mouse_id=None):
		self.exp_name = exp_name
		self.experiment_id = experiment_id
		self.mouse_id = mouse_id
		if self.hardware_trigger_enabled:
			return self._send_command('start_experiment', (exp_name, experiment_id, mouse_id))
		return self._send_command('preview_experiment', (exp_name, experiment_id, mouse_id))

	def stop_stim(self):
		if self.hardware_trigger_enabled:
			self._send_command('stop_experiment')
		else:
			self._send_command('stop_preview')
		return True

	def stop(self):
		self._send_command('stop_camera')
		if self.process is not None:
			self.process.join(timeout=3.0)

class CameraGUI(QMainWindow):
	def __init__(self):
		super().__init__()
		self.config = load_camera_config(str(CONFIG_DIR / 'cam_config.yaml'))
		self.display_target_size = None
		self.camera_app = None
		self.teensy_controller = None
		self.teensy_active = False
		self.hardware_trigger_enabled = False
		self.preview_mode = False
		self.preview_exp_thread = None
		self.current_exp_config = None
		self.current_cam_config = None
		self.current_teensy_config = None
		self.reset_display_average_on_next_frame = False
		self.reset_fps_on_next_frame = False
		self._auto_stop_pending = False
		self.histogram_open = False
		self.histogram_thread_running = False
		self.histogram_thread = None
		self.last_stats_time = None
		self.last_stats_frame_count = 0
		self.fps_samples = deque(maxlen=100)
		self.exp_status_timer = QTimer()
		self.exp_status_timer.timeout.connect(self.check_experiment_status)
		self.exp_status_timer.start(100)
		self.init_ui()
		self.populate_experiment_controls()
		self.setup_timer()

	def resizeEvent(self, event):
		super().resizeEvent(event)
		# Only update display target size when the window size actually changes.
		if event.oldSize() != event.size() and hasattr(self, 'video_label'):
			self.display_target_size = self.video_label.size()

	def check_experiment_status(self):
		if self.camera_app and hasattr(self.camera_app, 'get_exp_status'):
			messages = self.camera_app.get_exp_status()
			for msg in messages:
				if msg[0] == "status":
					self.update_status(msg[1])
				elif msg[0] == "trial":
					self.update_status(msg[3])
				elif msg[0] in ("logic_analyzer_terminated", "experiment_finished"):
					if not self._auto_stop_pending:
						self._auto_stop_pending = True
						from PyQt6.QtCore import QTimer
						QTimer.singleShot(0, self._auto_stop_after_logic_analyzer)

	def init_ui(self):
		self.setWindowTitle('Camera GUI')
		self.setGeometry(100, 100, 1400, 800)

		central_widget = QWidget()
		self.setCentralWidget(central_widget)

		main_layout = QHBoxLayout()
		central_widget.setLayout(main_layout)

		left_panel = self.create_display_panel()
		main_layout.addWidget(left_panel, 2)

		right_panel = self.create_control_panel()
		right_panel.setFixedWidth(420)
		main_layout.addWidget(right_panel, 1)

	def create_display_panel(self):
		panel = QWidget()
		layout = QVBoxLayout()
		panel.setLayout(layout)

		self.video_label = QLabel()
		self.video_label.setText('Camera not started.\n\nClick "Start Camera" to begin.')
		self.video_label.setMinimumSize(640, 480)
		self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
		self.video_label.setStyleSheet('border: 1px solid gray; background-color: #f0f0f0;')

		layout.addWidget(self.video_label)

		configs_layout = QHBoxLayout()

		exp_config_group = QGroupBox("Experiment Configuration")
		exp_config_group.setFixedWidth(300)
		exp_config_group.setFixedHeight(250)
		exp_config_layout = QVBoxLayout()
		self.config_display = QTextEdit()
		self.config_display.setFixedHeight(200)
		self.config_display.setReadOnly(True)
		self.config_display.setText("No experiment running. \n\nPreview/start an experiment to load configuration.")
		exp_config_layout.addWidget(self.config_display)
		exp_config_group.setLayout(exp_config_layout)
		configs_layout.addWidget(exp_config_group)

		cam_config_group = QGroupBox("Camera Configuration")
		cam_config_group.setFixedWidth(300)
		cam_config_group.setFixedHeight(250)
		cam_config_layout = QVBoxLayout()
		self.cam_config_display = QTextEdit()
		self.cam_config_display.setFixedHeight(200)
		self.cam_config_display.setReadOnly(True)
		self.cam_config_display.setText("Camera configuration will display here.")
		cam_config_layout.addWidget(self.cam_config_display)
		cam_config_group.setLayout(cam_config_layout)
		configs_layout.addWidget(cam_config_group)

		teensy_config_group = QGroupBox("Teensy Configuration")
		teensy_config_group.setFixedWidth(300)
		teensy_config_group.setFixedHeight(250)
		teensy_config_layout = QVBoxLayout()
		self.teensy_config_display = QTextEdit()
		self.teensy_config_display.setFixedHeight(200)
		self.teensy_config_display.setReadOnly(True)
		self.teensy_config_display.setText("Teensy configuration will display here.")
		teensy_config_layout.addWidget(self.teensy_config_display)
		teensy_config_group.setLayout(teensy_config_layout)
		configs_layout.addWidget(teensy_config_group)

		layout.addLayout(configs_layout)

		return panel

	def create_control_panel(self):
		panel = QWidget()
		layout = QVBoxLayout()
		panel.setLayout(layout)

		# Camera control
		cam_group = QGroupBox("Camera Controls")
		cam_layout = QVBoxLayout()

		self.start_btn = QPushButton("Start Camera")
		self.start_btn.clicked.connect(self.start_camera)
		cam_layout.addWidget(self.start_btn)

		self.stop_btn = QPushButton("Stop Camera")
		self.stop_btn.clicked.connect(self.stop_camera)
		self.stop_btn.setEnabled(False)
		cam_layout.addWidget(self.stop_btn)

		self.trigger_btn = QPushButton("Enable Hardware Trigger")
		self.trigger_btn.clicked.connect(self.toggle_trigger_mode)
		self.trigger_btn.setEnabled(False)
		cam_layout.addWidget(self.trigger_btn)

		self.teensy_toggle_btn = QPushButton("Enable Teensy")
		self.teensy_toggle_btn.clicked.connect(self.toggle_teensy_mode)
		self.teensy_toggle_btn.setEnabled(False)
		cam_layout.addWidget(self.teensy_toggle_btn)

		cam_group.setLayout(cam_layout)
		layout.addWidget(cam_group)

		# Camera settings
		settings_group = QGroupBox("Camera Settings")
		settings_layout = QVBoxLayout()

		exp_layout = QHBoxLayout()
		exp_layout.addWidget(QLabel("Exposure (ms):"))
		self.exposure_spin = QDoubleSpinBox()
		self.exposure_spin.setRange(0.1, 1000)
		self.exposure_spin.setValue(self.config['EXPOSURE_TIME'])
		self.exposure_spin.valueChanged.connect(self.update_exposure)
		exp_layout.addWidget(self.exposure_spin)
		settings_layout.addLayout(exp_layout)

		gain_layout = QHBoxLayout()
		gain_layout.addWidget(QLabel("Gain:"))
		self.gain_spin = QSpinBox()
		self.gain_spin.setRange(1, 100)
		self.gain_spin.setValue(self.config['ANALOG_GAIN'])
		self.gain_spin.valueChanged.connect(self.update_gain)
		gain_layout.addWidget(self.gain_spin)
		settings_layout.addLayout(gain_layout)

		settings_group.setLayout(settings_layout)
		layout.addWidget(settings_group)

		# Experiment controls
		exp_group = QGroupBox("Experiment Controls")
		exp_layout = QVBoxLayout()

		exp_select_layout = QHBoxLayout()
		exp_select_layout.addWidget(QLabel("Experiment:"))
		self.exp_combo = QComboBox()
		self.exp_combo.setEnabled(False)
		exp_select_layout.addWidget(self.exp_combo)
		exp_layout.addLayout(exp_select_layout)

		mouse_id_layout = QHBoxLayout()
		mouse_id_layout.addWidget(QLabel("Mouse ID:"))
		self.mouse_id_input = QLineEdit()
		self.mouse_id_input.setPlaceholderText("Enter mouse ID")
		mouse_id_layout.addWidget(self.mouse_id_input)
		exp_layout.addLayout(mouse_id_layout)

		exp_id_layout = QHBoxLayout()
		exp_id_layout.addWidget(QLabel("Experiment ID:"))
		self.experiment_id_input = QLineEdit()
		self.experiment_id_input.setPlaceholderText("Enter experiment ID")
		exp_id_layout.addWidget(self.experiment_id_input)
		exp_layout.addLayout(exp_id_layout)

		self.preview_btn = QPushButton("Preview Experiment")
		self.preview_btn.clicked.connect(self.preview_experiment)
		self.preview_btn.setEnabled(False)
		exp_layout.addWidget(self.preview_btn)

		self.stop_preview_btn = QPushButton("Stop Preview")
		self.stop_preview_btn.clicked.connect(self.stop_preview)
		self.stop_preview_btn.setEnabled(False)
		exp_layout.addWidget(self.stop_preview_btn)

		self.exp_btn = QPushButton("Start Experiment")
		self.exp_btn.clicked.connect(self.start_experiment)
		self.exp_btn.setEnabled(False)
		exp_layout.addWidget(self.exp_btn)

		self.stop_exp_btn = QPushButton("Stop Experiment")
		self.stop_exp_btn.clicked.connect(self.stop_experiment)
		self.stop_exp_btn.setEnabled(False)
		exp_layout.addWidget(self.stop_exp_btn)

		exp_group.setLayout(exp_layout)
		layout.addWidget(exp_group)

		# Live video processing
		proc_group = QGroupBox("Image Processing")
		proc_layout = QVBoxLayout()

		self.normalize_cb = QCheckBox("Normalize Image")
		self.normalize_cb.stateChanged.connect(self.toggle_normalize)
		proc_layout.addWidget(self.normalize_cb)

		self.background_cb = QCheckBox("Remove Background")
		self.background_cb.stateChanged.connect(self.toggle_background)
		proc_layout.addWidget(self.background_cb)

		self.speckle_cb = QCheckBox("Live Speckle")
		self.speckle_cb.stateChanged.connect(self.toggle_speckle)
		proc_layout.addWidget(self.speckle_cb)

		self.dfof_cb = QCheckBox("Enable dFoF")
		self.dfof_cb.stateChanged.connect(self.toggle_dfof)
		proc_layout.addWidget(self.dfof_cb)

		self.histogram_cb = QCheckBox("Show Histogram")
		self.histogram_cb.stateChanged.connect(self.toggle_histogram)
		proc_layout.addWidget(self.histogram_cb)

		proc_group.setLayout(proc_layout)
		layout.addWidget(proc_group)

		# Status
		status_group = QGroupBox("Status")
		status_layout = QVBoxLayout()

		self.status_text = QTextEdit()
		self.status_text.setMaximumHeight(200)
		self.status_text.setReadOnly(True)
		status_layout.addWidget(self.status_text)

		self.stats_label = QLabel("FPS: 0 | Frames: 0 | Saved: 0 | Save Queue: 0")
		status_layout.addWidget(self.stats_label)

		status_group.setLayout(status_layout)
		layout.addWidget(status_group)

		layout.addStretch()

		return panel

	def populate_experiment_controls(self):
		self.exp_combo.clear()
		# Keep index 0 empty so the user must explicitly choose an experiment.
		self.exp_combo.addItem("")
		experiment_list = []
		if self.camera_app and getattr(self.camera_app, 'exp_list', None):
			experiment_list = self.camera_app.exp_list
		else:
			try:
				experiment_list = get_experiment_list()
			except Exception as e:
				self.update_status(f"Error loading experiment list: {e}.")

		if not experiment_list:
			self.exp_combo.setCurrentIndex(0)
			self.exp_combo.setEnabled(False)
			return

		self.exp_combo.addItems(experiment_list)
		self.exp_combo.setCurrentIndex(0)
		self.exp_combo.setEnabled(True)

	def get_selected_experiment_params(self):
		exp_name = self.exp_combo.currentText().strip()
		experiment_id = self.experiment_id_input.text().strip()
		mouse_id = self.mouse_id_input.text().strip()

		if not exp_name:
			self.update_status("Error: Select an experiment from the dropdown.")
			return None

		if not experiment_id:
			self.update_status("Error: Enter an experiment ID.")
			return None

		if not mouse_id:
			self.update_status("Error: Enter a mouse ID.")
			return None

		return exp_name, experiment_id, mouse_id

	def setup_timer(self):
		self.timer = QTimer()
		self.timer.timeout.connect(self.update_display)
		self.stats_timer = QTimer()
		self.stats_timer.timeout.connect(self.update_stats)

	def start_camera(self):
		try:
			self.camera_app = CameraProcessClient(self.config)
			self.camera_app.start()
			self.display_target_size = self.video_label.size()
			self.last_stats_time = time.time()
			self.last_stats_frame_count = 0
			self.fps_samples.clear()

			def _on_logic_analyzer_terminated():
				from PyQt6.QtCore import QTimer
				QTimer.singleShot(0, self._auto_stop_after_logic_analyzer)
			self.camera_app.on_logic_analyzer_terminated = _on_logic_analyzer_terminated
			self.camera_app.on_experiment_finished = _on_logic_analyzer_terminated

			self.timer.start(30)
			self.stats_timer.start(100)

			self.start_btn.setEnabled(False)
			self.stop_btn.setEnabled(True)
			self.trigger_btn.setEnabled(True)
			self.teensy_toggle_btn.setEnabled(True)
			self.teensy_active = False
			self.teensy_toggle_btn.setText("Enable Teensy")
			self.exp_btn.setEnabled(False)
			self.preview_btn.setEnabled(True)
			self.populate_experiment_controls()

			self.load_and_display_camera_config()
			self.load_and_display_teensy_config()

			self.update_status("Camera started successfully in continuous mode.")
		except Exception as e:
			self.update_status(f"Error starting camera: {str(e)}.")

	def _auto_stop_after_logic_analyzer(self):
		if self._auto_stop_pending:
			self._auto_stop_pending = False
		self.disable_hardware_trigger()
		self.stop_experiment()
		self.stop_camera()

	def disable_hardware_trigger(self):
		if self.camera_app:
			try:
				self.camera_app.set_trigger_mode(False)
			except Exception as e:
				self.update_status(f"Error disabling hardware trigger: {e}.")
			self.camera_app.set_saving(False)

		self.hardware_trigger_enabled = False
		self.trigger_btn.setText("Enable Hardware Trigger")
		self.exp_btn.setEnabled(False)

	def stop_camera(self):
		if self.camera_app:
			self.disable_hardware_trigger()
			if self.teensy_controller is not None:
				try:
					if self.teensy_active:
						self.teensy_controller.stop_teensy()
				except Exception as e:
					self.update_status(f"Error stopping Teensy: {e}.")
				try:
					if hasattr(self.teensy_controller, 'ser') and self.teensy_controller.ser and self.teensy_controller.ser.is_open:
						self.teensy_controller.ser.close()
				except Exception:
					pass
				self.teensy_controller = None
				self.teensy_active = False
			self.camera_app.stop()
			self.timer.stop()
			self.stats_timer.stop()
			self.camera_app = None

		self.start_btn.setEnabled(True)
		self.stop_btn.setEnabled(False)
		self.trigger_btn.setEnabled(False)
		self.teensy_toggle_btn.setEnabled(False)
		self.teensy_toggle_btn.setText("Enable Teensy")
		self.hardware_trigger_enabled = False
		self.exp_btn.setEnabled(False)
		self.preview_btn.setEnabled(False)
		self.video_label.setText("Camera Stopped.")
		self.display_target_size = None
		self.cam_config_display.setText("Camera stopped. \n\n Start camera to begin.")
		self.current_cam_config = None
		self.populate_experiment_controls()
		self.update_status("Camera Stopped.")

	def update_display(self):
		if self.camera_app:
			try:
				frame_data = self.camera_app.get_frame_for_display()
				if frame_data is None:
					return
				frame = np.frombuffer(frame_data, dtype=self.camera_app.dtype)

				if self.reset_display_average_on_next_frame:
					if hasattr(self.camera_app, 'circular_buffer'):
						self.camera_app.circular_buffer.fill(0)
					if hasattr(self.camera_app, 'current_buffer_item'):
						self.camera_app.current_buffer_item = 0
					if hasattr(self.camera_app, 'buffer_loop_reached'):
						self.camera_app.buffer_loop_reached = False
					self.reset_display_average_on_next_frame = False

				if self.reset_fps_on_next_frame:
					self.last_stats_time = time.time()
					self.last_stats_frame_count = self.camera_app.frame_count
					self.fps_samples.clear()
					self.reset_fps_on_next_frame = False

				if hasattr(self.camera_app, 'height') and hasattr(self.camera_app, 'width'):
					frame = frame.reshape((self.camera_app.height, self.camera_app.width))
					frame = cv2.flip(frame, 1)
				
				if self.camera_app.bin_exp:
					bs = self.camera_app.bin_size
					frame = frame.astype(np.float32)
					if bs in (2, 4, 8, 16):
						for _ in range(bs.bit_length() - 1):
							frame = (frame[0::2, 0::2] + frame[1::2, 0::2] + frame[0::2, 1::2] + frame[1::2, 1::2])
					else:
						h, w = frame.shape
						frame = frame.reshape(h // bs, bs, w // bs, bs).sum(axis=(1, 3))
					frame /= (bs * bs)

				display_frame = self.process_frame_for_display(frame)
				self.display_processed_frame(display_frame)
			except Exception as e:
				print(f"Display error: {e}.")

	def process_frame_for_display(self, frame):
		if self.camera_app.dtype == 'uint16':
			frame = frame.astype(np.float32)

		if self.camera_app.live_speck and hasattr(self.camera_app, 'enable_live_speckle') and self.camera_app.enable_live_speckle:
			frame = self.camera_app.std_filter_frame(frame)
			if hasattr(self.camera_app, 'circular_buffer'):
				self.camera_app.circular_buffer[self.camera_app.current_buffer_item, :, :] = frame
				self.camera_app.current_buffer_item += 1
				self.camera_app.current_buffer_item %= self.camera_app.buffer_size
				if (not self.camera_app.buffer_loop_reached) and self.camera_app.current_buffer_item == 0:
					self.camera_app.buffer_loop_reached = True

				if self.camera_app.buffer_loop_reached:
					frame = self.camera_app.circular_buffer.mean(axis=0)
				else:
					filled = max(1, self.camera_app.current_buffer_item)
					frame = self.camera_app.circular_buffer[:filled, :, :].mean(axis=0)
				clipped = np.clip(frame, self.camera_app.vmin, self.camera_app.vmax)
				frame = ((clipped - self.camera_app.vmin) / (self.camera_app.vmax - self.camera_app.vmin)) * 255

		if self.camera_app.removeBackground and hasattr(self.camera_app, 'backgroundImg'):
			frame = frame - self.camera_app.backgroundImg
			frame = np.clip(frame, 0, 255)

		if self.camera_app.normalizeImage:
			clipped = np.clip(frame, self.camera_app.minI, self.camera_app.maxI)
			frame = ((clipped - self.camera_app.minI) / (self.camera_app.maxI - self.camera_app.minI)) * 255

		if self.camera_app.dFoF_open and self.camera_app.F0 is not None:
			dfof = (frame.astype(np.float32) - self.camera_app.F0) / self.camera_app.F0
			dfof = np.nan_to_num(dfof, nan=0.0)

			if self.camera_app.normalizeImage:
				clipped = np.clip(dfof, self.camera_app.minI, self.camera_app.maxI)
				frame = ((clipped - self.camera_app.minI) / (self.camera_app.maxI - self.camera_app.minI)) * 255
			else:
				fmin, fmax = dfof.min(), dfof.max()
				if fmax > fmin:
					frame = ((dfof - fmin) / (fmax - fmin)) * 255
				else:
					frame = np.zeros_like(dfof)

		if hasattr(self.camera_app, 'height') and hasattr(self.camera_app, 'width'):
			frame = cv2.resize(frame, (self.camera_app.width//2, self.camera_app.height//2), interpolation=cv2.INTER_NEAREST)

		return frame.astype(np.uint8)

	def display_processed_frame(self, frame):
		try:
			h, w = frame.shape
			bytes_per_line = w
			q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
			pixmap = QPixmap.fromImage(q_img)
			target_size = self.display_target_size if self.display_target_size is not None else self.video_label.size()
			if target_size.width() <= 0 or target_size.height() <= 0:
				target_size = self.video_label.size()
			self.video_label.setPixmap(pixmap.scaled(target_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation))
		except Exception as e:
			print(f"Display frame error: {e}.")

	def update_stats(self):
		if self.camera_app:
			try:
				if hasattr(self.camera_app, 'poll_messages'):
					self.camera_app.poll_messages()
				current_time = time.time()
				if self.last_stats_time is None:
					self.last_stats_time = current_time
					self.last_stats_frame_count = self.camera_app.frame_count

				delta_time = current_time - self.last_stats_time
				delta_frames = self.camera_app.frame_count - self.last_stats_frame_count
				if delta_time > 0 and delta_frames >= 0:
					self.fps_samples.append(delta_frames / delta_time)

				self.last_stats_time = current_time
				self.last_stats_frame_count = self.camera_app.frame_count
				average_fps = sum(self.fps_samples) / len(self.fps_samples) if self.fps_samples else 0

				save_queue_size = getattr(self.camera_app, 'save_queue_size', 0)
				stats_text = f"FPS: {average_fps: .1f} | Frames: {self.camera_app.frame_count} | Saved: {self.camera_app.frames_written} | Save Queue: {save_queue_size}"
				self.stats_label.setText(stats_text)
			except:
				pass

	def toggle_trigger_mode(self):
		if self.camera_app:
			try:
				if not self.hardware_trigger_enabled:
					print("\n Switching to hardware trigger mode.")
					self.camera_app.set_trigger_mode(True)
					self.hardware_trigger_enabled = True
					
					self.trigger_btn.setText("Disable Hardware Trigger")
					self.exp_btn.setEnabled(True)
					self.update_status("Hardware trigger mode enabled - ready for experiment.")
				else:
					self.hardware_trigger_enabled = False
					self.camera_app.set_saving(False)
					self.camera_app.set_trigger_mode(False)
					print("\n Switching to continuous mode.")

					self.trigger_btn.setText("Enable Hardware Trigger")
					self.exp_btn.setEnabled(False)
					self.update_status("Continuous mode enabled.")
			except Exception as e:
				self.update_status(f"Error toggling trigger mode: {e}.")

	def toggle_teensy_mode(self):
		if not self.camera_app:
			self.update_status("Error: Camera must be started first.")
			return

		try:
			if self.teensy_controller is None:
				self.teensy_controller = TeensyController("gui", True, str(CONFIG_DIR / "teensyParams.yaml"))

			if not self.teensy_active:
				self.teensy_controller.start_teensy()
				self.teensy_active = True
				self.teensy_toggle_btn.setText("Disable Teensy")
				self.update_status("Teensy started.")
			else:
				self.teensy_controller.stop_teensy()
				self.teensy_active = False
				self.teensy_toggle_btn.setText("Enable Teensy")
				self.update_status("Teensy stopped.")
		except Exception as e:
			self.update_status(f"Error toggling Teensy: {e}.")

	def preview_experiment(self):
		if not self.camera_app:
			self.update_status('Error: Camera must be started first.')
			return

		params = self.get_selected_experiment_params()
		if params is None:
			return

		exp_name, experiment_id, mouse_id = params

		self.preview_mode = True
		started = self.camera_app.get_exp_params(exp_name=exp_name, experiment_id=experiment_id, mouse_id=mouse_id)
		if not started:
			self.update_status('Error: Failed to start experiment preview subprocess.')
			return
		self.load_and_display_exp_config()
		self.preview_btn.setEnabled(False)
		self.stop_preview_btn.setEnabled(True)
		self.exp_btn.setEnabled(False)
		self.trigger_btn.setEnabled(False)

		self.update_status(f'Starting experiment preview: {exp_name}.')

	def stop_preview(self):
		if self.camera_app:
			self.preview_mode = False
			self.camera_app.stop_stim()
			self.preview_btn.setEnabled(True)
			self.stop_preview_btn.setEnabled(False)
			self.trigger_btn.setEnabled(True)
			self.update_status("Preview stopped.")
			self.config_display.setText("Experiment stopped. \n\nStart/preview experiment to load configuration.")
			self.current_exp_config = None

	def start_experiment(self):
		if self.camera_app and self.hardware_trigger_enabled:
			params = self.get_selected_experiment_params()
			if params is None:
				return

			exp_name, experiment_id, mouse_id = params
			try:
				# Ensure GUI-side Teensy is fully stopped and released before
				# the experiment process (simple_cam_mx) initializes Teensy.
				if self.teensy_controller is not None:
					self.teensy_controller.stop_teensy()
					if hasattr(self.teensy_controller, 'ser') and self.teensy_controller.ser and self.teensy_controller.ser.is_open:
						self.teensy_controller.ser.close()
					self.teensy_controller = None
				self.teensy_active = False
				self.teensy_toggle_btn.setText("Enable Teensy")
			except Exception as e:
				self.update_status(f"Error stopping Teensy before experiment start: {e}.")
				return

			self.camera_app.set_saving(True)
			started = self.camera_app.get_exp_params(exp_name=exp_name, experiment_id=experiment_id, mouse_id=mouse_id)
			if not started:
				self.camera_app.set_saving(False)
				self.update_status('Error: Failed to start experiment subprocess.')
				return
			self.load_and_display_exp_config()
			self.reset_display_average_on_next_frame = True
			self.reset_fps_on_next_frame = True
			self.exp_btn.setEnabled(False)
			self.stop_exp_btn.setEnabled(True)
			self.trigger_btn.setEnabled(False)
			self.update_status(f"Experiment started: {exp_name}.")
		else:
			self.update_status("Error: Must enable hardware trigger mode first.")

	def stop_experiment(self):
		if self.camera_app:
			self.camera_app.set_saving(False)
			self.camera_app.stop_stim()
			self.exp_btn.setEnabled(True)
			self.stop_exp_btn.setEnabled(False)
			self.trigger_btn.setEnabled(True)
			self.update_status("Experiment stopped.")
			self.config_display.setText("Experiment stopped. \n\nStart/preview experiment to load configuration.")
			self.current_exp_config = None

	def load_and_display_exp_config(self):
		try:
			exp_name = getattr(self.camera_app, 'exp_name', None)
	        
			if exp_name is None:
				self.config_display.setText("No experiment selected.\nPlease start an experiment first.")
				return

			config_file = CONFIG_DIR / f"{exp_name.replace(' ', '_').lower()}_config.yaml"
	        
			try:
				with open(config_file, 'r') as f:
					config_content = f.read()

				try:
					config_data = yaml.safe_load(config_content)
	                	           
					formatted_text = f"{exp_name} Configuration\n\n"
	                
					if isinstance(config_data, dict):
	                    # Find longest key for alignment
						max_key_len = max(len(str(k)) for k in config_data.keys())
	                    
						for key, value in config_data.items():
	                        # Format the value nicely
							if isinstance(value, list):
								list_str = str(value)
	                            # Remove brackets for cleaner look
								list_str = list_str.strip('[]')
								formatted_text += f"{key:<{max_key_len}} : {list_str}\n"
							elif isinstance(value, dict):
								formatted_text += f"{key:<{max_key_len}} : [nested parameters]\n"
							else:
								formatted_text += f"{key:<{max_key_len}} : {value}\n"
	                
					self.config_display.setText(formatted_text)
					self.current_exp_config = config_data
	                
				except Exception as e:
	                # If YAML parsing fails, show raw content but remove dashes
					lines = config_content.split('\n')
					clean_lines = []
					for line in lines:
						if line.strip().startswith('-'):
	                        # Remove the dash and any following space
							line = line.replace('-', '', 1).lstrip()
						clean_lines.append(line)
	                
					self.config_display.setText('\n'.join(clean_lines))
	        
			except Exception as e:
				self.config_display.setText(f"Error loading experiment configuration file '{config_file}': \n{str(e)}.")

		except Exception as e:
			self.config_display.setText(f"Error loading experiment configuration: \n{str(e)}.")

	def load_and_display_camera_config(self):
		try:
			config_file = CONFIG_DIR / 'cam_config.yaml'

			try:
				with open(config_file, 'r') as f:
					config_content = f.read()

					try:
						config_data = yaml.safe_load(config_content)

						formatted_text = "Camera Settings\n\n"

						if isinstance(config_data, dict):
	                    # Find longest key for alignment
							max_key_len = max(len(str(k)) for k in config_data.keys())
	                    
							for key, value in config_data.items():
		                        # Format the value nicely
								if isinstance(value, list):
									list_str = str(value)
		                            # Remove brackets for cleaner look
									list_str = list_str.strip('[]')
									formatted_text += f"{key:<{max_key_len}} : {list_str}\n"
								elif isinstance(value, dict):
									formatted_text += f"{key:<{max_key_len}} : [nested parameters]\n"
								else:
									formatted_text += f"{key:<{max_key_len}} : {value}\n"
	                
						self.cam_config_display.setText(formatted_text)
						self.current_cam_config = config_data

					except Exception as e:
		                # If YAML parsing fails, show raw content but remove dashes
						lines = config_content.split('\n')
						clean_lines = []
						for line in lines:
							if line.strip().startswith('-'):
		                        # Remove the dash and any following space
								line = line.replace('-', '', 1).lstrip()
							clean_lines.append(line)
		                
						self.cam_config_display.setText('\n'.join(clean_lines))
	        
			except Exception as e:
				self.cam_config_display.setText(f"Error loading camera configuration file '{config_file}': \n{str(e)}.")

		except Exception as e:
			self.cam_config_display.setText(f"Error loading camera configuration: \n{str(e)}.")

	def load_and_display_teensy_config(self):
		try:
			config_file = CONFIG_DIR / 'teensyParams.yaml'

			try:
				with open(config_file, 'r') as f:
					config_content = f.read()

					try:
						config_data = yaml.safe_load(config_content)

						formatted_text = "Teensy Parameters\n\n"

						if isinstance(config_data, dict):
	                    # Find longest key for alignment
							max_key_len = max(len(str(k)) for k in config_data.keys())
	                    
							for key, value in config_data.items():
		                        # Format the value nicely
								if isinstance(value, list):
									list_str = str(value)
		                            # Remove brackets for cleaner look
									list_str = list_str.strip('[]')
									formatted_text += f"{key:<{max_key_len}} : {list_str}\n"
								elif isinstance(value, dict):
									formatted_text += f"{key:<{max_key_len}} : [nested parameters]\n"
								else:
									formatted_text += f"{key:<{max_key_len}} : {value}\n"
	                
						self.teensy_config_display.setText(formatted_text)
						self.current_teensy_config = config_data

					except Exception as e:
		                # If YAML parsing fails, show raw content but remove dashes
						lines = config_content.split('\n')
						clean_lines = []
						for line in lines:
							if line.strip().startswith('-'):
		                        # Remove the dash and any following space
								line = line.replace('-', '', 1).lstrip()
							clean_lines.append(line)
		                
						self.teensy_config_display.setText('\n'.join(clean_lines))
	        
			except Exception as e:
				self.teensy_config_display.setText(f"Error loading teensy configuration file '{config_file}': \n{str(e)}.")

		except Exception as e:
			self.teensy_config_display.setText(f"Error loading teensy configuration: \n{str(e)}.")

	def update_exposure(self, value):
		if self.camera_app:
			try:
				self.camera_app.exposure = value
				self.camera_app.set_exposure(value)
				self.load_and_display_camera_config()
				self.update_status(f"Exposure set to {value}ms.")
			except Exception as e:
				self.update_status(f'Error setting exposure: {e}.')

	def update_gain(self, value):
		if self.camera_app:
			try:
				self.camera_app.analog_gain = value
				self.camera_app.set_gain(value)
				self.load_and_display_camera_config()
				self.update_status(f"Gain set to {value}.")
			except Exception as e:
				self.update_status(f'Error setting gain: {e}.')

	def toggle_normalize(self, state):
		if self.camera_app:
			if state == Qt.CheckState.Checked.value:
				self.camera_app.adjust_dynamic_range()
				self.update_status("Normalization enabled.")

			else:
				if self.camera_app.normalizeImage:
					self.camera_app.adjust_dynamic_range()
				self.update_status('Normalization disabled.')

	def toggle_background(self, state):
		if self.camera_app:
			if state == Qt.CheckState.Checked.value:
				self.camera_app.toggle_background_removal()
				self.update_status("Background subtraction enabled.")

			else:
				if self.camera_app.removeBackground:
					self.camera_app.toggle_background_removal()
				self.update_status('Background subtraction disabled.')
 
	def toggle_speckle(self, state):
		if self.camera_app:
			if state == Qt.CheckState.Checked.value:
				self.camera_app.toggle_speckle()
				self.update_status("Live speckle imaging enabled.")

			else:
				if self.camera_app.enable_live_speckle:
					self.camera_app.toggle_speckle()
				self.update_status('Live speckle imaging disabled.')			

	def toggle_dfof(self, state):
		if self.camera_app:
			if state == Qt.CheckState.Checked.value:
				self.camera_app.toggle_dFoF()
				self.update_status('dFoF enabled.')
			else:
				self.camera_app.dFoF_open = False
				self.update_status('dFoF disabled.')

	def toggle_histogram(self, state):
		if not self.camera_app:
			return

		if state == Qt.CheckState.Checked.value:
			if self.histogram_open:
				return
			self.histogram_open = True
			self.histogram_thread_running = True
			self.histogram_thread = threading.Thread(target=self.update_histogram, daemon=True)
			self.histogram_thread.start()
			self.update_status('Histogram enabled.')
		else:
			self.histogram_open = False
			self.histogram_thread_running = False
			self.update_status('Histogram disabled.')

	def update_histogram(self):
		hist_width = 512
		hist_height = 400
		bin_width = 2
		max_history = 50
		intensity_history = []

		while self.histogram_open and self.histogram_thread_running and self.camera_app:
			frame_data = self.camera_app.get_frame_for_display()
			if frame_data is not None:
				try:
					frame = np.frombuffer(frame_data, dtype=self.camera_app.dtype)
					if hasattr(self.camera_app, 'height') and hasattr(self.camera_app, 'width') and self.camera_app.height and self.camera_app.width:
						frame = frame.reshape((self.camera_app.height, self.camera_app.width))

					if self.camera_app.dFoF_open and self.camera_app.F0 is not None:
						dfof = (frame.astype(np.float32) - self.camera_app.F0) / self.camera_app.F0
						dfof = np.nan_to_num(dfof, nan=0.0)

						if self.camera_app.normalizeImage:
							clipped = np.clip(dfof, self.camera_app.minI, self.camera_app.maxI)
							display_frame = ((clipped - self.camera_app.minI) / (self.camera_app.maxI - self.camera_app.minI)) * 255
						else:
							fmin, fmax = dfof.min(), dfof.max()
							if fmax > fmin:
								display_frame = ((dfof - fmin) / (fmax - fmin)) * 255
							else:
								display_frame = np.zeros_like(dfof)

						display_frame = display_frame.astype(np.uint8)
					else:
						if self.camera_app.normalizeImage:
							clipped = np.clip(frame, self.camera_app.minI, self.camera_app.maxI)
							display_frame = ((clipped - self.camera_app.minI) / (self.camera_app.maxI - self.camera_app.minI)) * 255
							display_frame = display_frame.astype(np.uint8)
						else:
							display_frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

					intensity_history.extend(display_frame.ravel().tolist())
					if len(intensity_history) > max_history * display_frame.size:
						intensity_history = intensity_history[-max_history * display_frame.size:]

					hist = cv2.calcHist([np.array(intensity_history, dtype=np.uint8)], [0], None, [256], [0, 256])
					cv2.normalize(hist, hist, 0, hist_height, cv2.NORM_MINMAX)

					hist_image = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)
					for i in range(256):
						intensity = int(hist[i, 0])
						cv2.rectangle(hist_image, (i * bin_width, hist_height - intensity), ((i + 1) * bin_width - 1, hist_height), (255, 255, 255), -1)

					if self.camera_app.normalizeImage:
						min_x = int(self.camera_app.minI * bin_width)
						max_x = int(self.camera_app.maxI * bin_width)
						cv2.line(hist_image, (min_x, 0), (min_x, hist_height), (0, 0, 255), 2)
						cv2.line(hist_image, (max_x, 0), (max_x, hist_height), (255, 0, 0), 2)

					cv2.putText(hist_image, f'Frame {self.camera_app.frame_count}', (hist_width // 2 - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
					cv2.imshow('Live Histogram', hist_image)
					cv2.waitKey(1)
				except Exception as e:
					print(f"\nHistogram error: {e}.")

			time.sleep(0.01)

		cv2.destroyWindow('Live Histogram')

	def update_trigger_mode(self, index):
		if self.camera_app:
			try:
				self.camera_app.set_trigger_mode(index != 0)
				mode_name = "Continuous" if index == 0 else "Hardware Trigger"
				self.update_status(f"Trigger mode set to: {mode_name}.")
			except Exception as e:
				self.update_status(f"Error setting trigger mode: {e}.")

	def update_status(self, message):
		timestamp = time.strftime("%H:%M:%S")
		self.status_text.append(f"[{timestamp}] {message}")
		self.status_text.verticalScrollBar().setValue(self.status_text.verticalScrollBar().maximum())

	def closeEvent(self, event):
		self.stop_camera()
		super().closeEvent(event)

def main():
	app = QApplication(sys.argv)
	window = CameraGUI()
	window.show()
	app.exec()

if __name__ == '__main__':
	main()