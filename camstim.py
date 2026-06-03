import sys
import os
import numpy as np
import multiprocessing as mp
import queue
try:
	from setproctitle import setproctitle as _setproctitle
except ImportError:
	_setproctitle = None
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
							QWidget, QPushButton, QLabel, QDoubleSpinBox,
							QGridLayout,
							QGroupBox, QTextEdit, QCheckBox, QComboBox, QLineEdit, QSizePolicy,
							QFileDialog, QInputDialog, QMessageBox, QSplitter)
from PyQt6.QtCore import QTimer, Qt, QRect, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QAction, QPainter, QPen, QColor
import time
from collections import deque
import threading
import re
import subprocess
import tempfile
from core.TeensyController import TeensyController
import cv2
import yaml
from pathlib import Path
try:
	import matplotlib.pyplot as plt
except Exception:
	plt = None

from utils.simple_cam_mx import load_camera_config, load_save_root, run_camera_worker
from core.experiment_discovery import get_experiment_list, resolve_experiment_config_file, discover_experiment_types


CONFIG_DIR = Path(__file__).resolve().parent / 'config_files'
VERSION_FILE = Path(__file__).resolve().parent / 'VERSION'
REPO_ROOT = Path(__file__).resolve().parent


def load_app_version(default='0.0.0'):
	try:
		version_text = VERSION_FILE.read_text(encoding='utf-8').strip()
		return version_text if version_text else default
	except OSError:
		return default


class RoiSelectableLabel(QLabel):
	def __init__(self, parent=None):
		super().__init__(parent)
		self._roi_mode_enabled = False
		self._drag_start = None
		self._drag_current = None
		self._roi_image_rect = QRect()
		self.selection_finished_callback = None

	def set_roi_mode_enabled(self, enabled):
		self._roi_mode_enabled = bool(enabled)
		self._drag_start = None
		self._drag_current = None
		self._roi_image_rect = self._pixmap_rect() if self._roi_mode_enabled else QRect()
		self.setCursor(Qt.CursorShape.CrossCursor if self._roi_mode_enabled else Qt.CursorShape.ArrowCursor)
		self.update()

	def _active_image_rect(self):
		if self._roi_mode_enabled and not self._roi_image_rect.isNull():
			return self._roi_image_rect
		return self._pixmap_rect()

	def _pixmap_rect(self):
		pixmap = self.pixmap()
		if pixmap is None or pixmap.isNull():
			return QRect()
		pixmap_size = pixmap.size()
		x = (self.width() - pixmap_size.width()) // 2
		y = (self.height() - pixmap_size.height()) // 2
		return QRect(x, y, pixmap_size.width(), pixmap_size.height())

	def _clamp_to_pixmap(self, pos):
		pixmap_rect = self._active_image_rect()
		if pixmap_rect.isNull():
			return pos
		x = min(max(pos.x(), pixmap_rect.left()), pixmap_rect.right())
		y = min(max(pos.y(), pixmap_rect.top()), pixmap_rect.bottom())
		return pos.__class__(x, y)

	def mousePressEvent(self, event):
		if self._roi_mode_enabled and event.button() == Qt.MouseButton.LeftButton:
			pixmap_rect = self._active_image_rect()
			point = event.position().toPoint()
			if pixmap_rect.contains(point):
				self._drag_start = self._clamp_to_pixmap(point)
				self._drag_current = self._drag_start
				self.update()
				return
		super().mousePressEvent(event)

	def mouseMoveEvent(self, event):
		if self._roi_mode_enabled and self._drag_start is not None:
			self._drag_current = self._clamp_to_pixmap(event.position().toPoint())
			self.update()
			return
		super().mouseMoveEvent(event)

	def mouseReleaseEvent(self, event):
		if self._roi_mode_enabled and event.button() == Qt.MouseButton.LeftButton and self._drag_start is not None:
			self._drag_current = self._clamp_to_pixmap(event.position().toPoint())
			selection_rect = QRect(self._drag_start, self._drag_current).normalized()
			image_rect = self._active_image_rect()
			self._drag_start = None
			self._drag_current = None
			self.update()
			if selection_rect.width() > 4 and selection_rect.height() > 4 and callable(self.selection_finished_callback):
				self.selection_finished_callback(selection_rect, image_rect)
			return
		super().mouseReleaseEvent(event)

	def paintEvent(self, event):
		super().paintEvent(event)
		if self._drag_start is None or self._drag_current is None:
			return
		painter = QPainter(self)
		painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
		painter.setPen(QPen(QColor(255, 64, 64), 2, Qt.PenStyle.SolidLine))
		painter.drawRect(QRect(self._drag_start, self._drag_current).normalized())

class HistogramWindow(QWidget):
	closed = pyqtSignal()

	def __init__(self, parent=None):
		super().__init__(parent)
		self.setWindowTitle('Live Histogram')
		self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
		self.setWindowFlag(Qt.WindowType.Window, True)

	def closeEvent(self, event):
		self.closed.emit()
		event.accept()


class HistogramLabel(QLabel):
	limitsChanged = pyqtSignal(float, float)

	def __init__(self, parent=None):
		super().__init__(parent)
		self._limit_min = 0.0
		self._limit_max = 255.0
		self._data_min = 0.0
		self._data_max = 255.0
		self._drag_target = None
		self._drag_margin_px = 8
		self.setMouseTracking(True)

	def set_histogram_limits(self, limit_min, limit_max, data_min=0.0, data_max=255.0):
		self._limit_min = float(limit_min)
		self._limit_max = float(limit_max)
		self._data_min = float(data_min)
		self._data_max = float(data_max)
		if self._data_max <= self._data_min:
			self._data_max = self._data_min + 1.0

	def _x_to_value(self, x_pos):
		width = max(1, self.width() - 1)
		fraction = min(max(float(x_pos) / float(width), 0.0), 1.0)
		return self._data_min + fraction * (self._data_max - self._data_min)

	def _value_to_x(self, value):
		width = max(1, self.width() - 1)
		fraction = (float(value) - self._data_min) / (self._data_max - self._data_min)
		fraction = min(max(fraction, 0.0), 1.0)
		return int(round(fraction * width))

	def _pick_drag_target(self, x_pos):
		min_x = self._value_to_x(self._limit_min)
		max_x = self._value_to_x(self._limit_max)
		min_dist = abs(x_pos - min_x)
		max_dist = abs(x_pos - max_x)
		if min_dist <= self._drag_margin_px or max_dist <= self._drag_margin_px:
			return 'min' if min_dist <= max_dist else 'max'
		return None

	def _update_cursor_for_x(self, x_pos):
		if self._drag_target is not None or self._pick_drag_target(x_pos) is not None:
			self.setCursor(Qt.CursorShape.SizeHorCursor)
		else:
			self.unsetCursor()

	def _emit_limits_for_x(self, x_pos):
		new_value = self._x_to_value(x_pos)
		if self._drag_target == 'min':
			new_min = min(new_value, self._limit_max - 1.0)
			new_min = max(new_min, self._data_min)
			self._limit_min = new_min
		elif self._drag_target == 'max':
			new_max = max(new_value, self._limit_min + 1.0)
			new_max = min(new_max, self._data_max)
			self._limit_max = new_max
		self.limitsChanged.emit(self._limit_min, self._limit_max)
		self.update()

	def mousePressEvent(self, event):
		if event.button() == Qt.MouseButton.LeftButton:
			self._drag_target = self._pick_drag_target(int(event.position().x()))
			if self._drag_target is not None:
				self.setCursor(Qt.CursorShape.SizeHorCursor)
				self._emit_limits_for_x(int(event.position().x()))
				return
		super().mousePressEvent(event)

	def mouseMoveEvent(self, event):
		if self._drag_target is not None:
			self._emit_limits_for_x(int(event.position().x()))
			return
		self._update_cursor_for_x(int(event.position().x()))
		super().mouseMoveEvent(event)

	def mouseReleaseEvent(self, event):
		if self._drag_target is not None and event.button() == Qt.MouseButton.LeftButton:
			self._emit_limits_for_x(int(event.position().x()))
			self._drag_target = None
			self._update_cursor_for_x(int(event.position().x()))
			return
		super().mouseReleaseEvent(event)

	def leaveEvent(self, event):
		if self._drag_target is None:
			self.unsetCursor()
		super().leaveEvent(event)

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
		self.removeBackground = False
		self.saving = False
		self.hardware_trigger_enabled = False
		self.normalizeImage = False
		self.dFoF_open = False
		self.F0 = None
		self.minI = 0
		self.maxI = 255
		self.autoI = 0.05
		self.backgroundImg = None
		self.histogram_open = False
		self.frame_count = 0
		self.frames_written = 0
		self.save_queue_size = 0
		self.display_queue_size = 0
		self.average_fps = None
		self.sensor_temperature = None
		self.exp_status_queue = queue.Queue()
		self.analog_gain = float(config.get('ANALOG_GAIN', 1.0))
		self.analog_gain_step = 0.1
		self.analog_gain_min = 0.1
		self.analog_gain_max = 100.0
		self.exp_name = None
		self.experiment_id = None
		self.mouse_id = None
		self.on_logic_analyzer_terminated = None
		self.on_experiment_finished = None
		self.active_save_filename = None
		use_cgrab_cfg = config.get('USE_CGRABCALLBACK', True)
		if isinstance(use_cgrab_cfg, str):
			self.use_c_framegrab = use_cgrab_cfg.strip().lower() in ('1', 'true', 'yes', 'on')
		else:
			self.use_c_framegrab = bool(use_cgrab_cfg)
		self.display_output_enabled = True

	def start(self):
		ctx = mp.get_context('spawn')
		self.frame_queue = ctx.Queue(maxsize=1)
		self.command_queue = ctx.Queue()
		self.status_queue = ctx.Queue()
		self.process = ctx.Process(
			target=run_camera_worker,
			args=(self.config, self.frame_queue, self.command_queue, self.status_queue),
			daemon=False,
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
		if self.process is None or not self.process.is_alive():
			self.exp_status_queue.put(("status", "Error: camera worker process is not running."))
			return False
		try:
			self.command_queue.put_nowait({'name': name, 'payload': payload})
			return True
		except Exception as e:
			self.exp_status_queue.put(("status", f"Error sending command '{name}': {e}"))
			return False

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
				self.analog_gain_step = float(msg.get('analog_gain_step', self.analog_gain_step))
				self.analog_gain_min = float(msg.get('analog_gain_min', self.analog_gain_min))
				self.analog_gain_max = float(msg.get('analog_gain_max', self.analog_gain_max))
				self.analog_gain = float(msg.get('analog_gain', self.analog_gain))
			elif msg_type == 'stats':
				self.frame_count = int(msg.get('frame_count', self.frame_count))
				self.frames_written = int(msg.get('frames_written', self.frames_written))
				self.save_queue_size = int(msg.get('save_queue_size', self.save_queue_size))
				self.display_queue_size = int(msg.get('display_queue_size', self.display_queue_size))
				if 'average_fps' in msg:
					self.average_fps = float(msg['average_fps'])
				if 'sensor_temperature' in msg and msg['sensor_temperature'] is not None:
					self.sensor_temperature = msg['sensor_temperature']
			elif msg_type == 'trigger_mode':
				self.hardware_trigger_enabled = msg.get('mode') == 2
			elif msg_type == 'exposure':
				self.exposure = msg.get('value', getattr(self, 'exposure', None))
			elif msg_type == 'gain':
				self.analog_gain = msg.get('value', getattr(self, 'analog_gain', None))
			elif msg_type == 'roi_applied':
				self.width = int(msg.get('width', self.width or 0)) or self.width
				self.height = int(msg.get('height', self.height or 0)) or self.height
				with self.latest_frame_lock:
					self.latest_frame_data = None
				self.exp_status_queue.put(("status", f"ROI applied: {msg.get('width')}x{msg.get('height')} at ({msg.get('x')}, {msg.get('y')})."))
			elif msg_type == 'framegrab_backend':
				self.use_c_framegrab = bool(msg.get('using_c', self.use_c_framegrab))
				self.exp_status_queue.put(("status", msg.get('message', 'Framegrab backend updated.')))
			elif msg_type == 'display_output_enabled':
				self.display_output_enabled = bool(msg.get('value', self.display_output_enabled))
			elif msg_type == 'status':
				self.exp_status_queue.put(("status", msg.get('message', '')))
			elif msg_type == 'saving_file':
				filename = msg.get('filename')
				if filename:
					self.active_save_filename = str(filename)
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
		if not self.display_output_enabled:
			return None
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

	def set_roi(self, x, y, width, height):
		return self._send_command('set_roi', {'x': x, 'y': y, 'width': width, 'height': height})

	def set_framegrab_backend(self, use_c_backend):
		self.use_c_framegrab = bool(use_c_backend)
		return self._send_command('set_framegrab_backend', bool(use_c_backend))

	def reset_roi(self):
		return self._send_command('reset_roi')

	def set_saving(self, enabled):
		self.saving = bool(enabled)
		return self._send_command('set_saving', bool(enabled))

	def set_display_output_enabled(self, enabled):
		self.display_output_enabled = bool(enabled)
		if not self.display_output_enabled:
			with self.latest_frame_lock:
				self.latest_frame_data = None
		return self._send_command('set_display_output_enabled', bool(enabled))

	def toggle_background_removal(self):
		if self.removeBackground:
			self.removeBackground = False
			return self.removeBackground

		frame_data = self.get_frame_for_display()
		if frame_data is None or not self.height or not self.width:
			self.removeBackground = False
			return self.removeBackground

		frame = np.frombuffer(frame_data, dtype=self.dtype)
		expected_pixels = int(self.height) * int(self.width)
		if frame.size < expected_pixels:
			self.removeBackground = False
			return self.removeBackground

		frame = frame[:expected_pixels].reshape((self.height, self.width)).astype(np.float32)
		kernel = np.ones((10, 10), np.float32) / 100.0
		self.backgroundImg = cv2.filter2D(frame, -1, kernel)
		self.removeBackground = True
		return self.removeBackground

	def toggle_dFoF(self):
		if self.dFoF_open:
			self.dFoF_open = False
			return self.dFoF_open

		if not self.height or not self.width:
			self.dFoF_open = False
			return self.dFoF_open

		baseline_frames = []
		f0_n_frames = 50
		expected_pixels = int(self.height) * int(self.width)
		for _ in range(f0_n_frames):
			frame_data = self.get_frame_for_display()
			if frame_data is None:
				time.sleep(0.02)
				continue
			frame = np.frombuffer(frame_data, dtype=self.dtype)
			if frame.size < expected_pixels:
				continue
			frame = frame[:expected_pixels].reshape((self.height, self.width))
			baseline_frames.append(frame.astype(np.float32))
			time.sleep(0.01)

		if len(baseline_frames) < f0_n_frames:
			self.dFoF_open = False
			return self.dFoF_open

		sampled_stack = np.stack(baseline_frames, axis=0)
		self.F0 = np.percentile(sampled_stack, 10, axis=0)
		epsilon = 1e-6
		self.F0[self.F0 == 0] = epsilon

		if not self.normalizeImage:
			test_dfof = []
			for frame in baseline_frames[:10]:
				dfof_test = (frame - self.F0) / self.F0
				test_dfof.append(dfof_test)

			test_dfof_stack = np.stack(test_dfof, axis=0)
			valid_values = test_dfof_stack[~np.isnan(test_dfof_stack)]
			if valid_values.size > 0:
				self.minI = float(np.percentile(valid_values, 1))
				self.maxI = float(np.percentile(valid_values, 99))
				if self.maxI <= self.minI:
					self.maxI = self.minI + 1.0
				self.normalizeImage = True

		self.dFoF_open = True
		return self.dFoF_open

	def toggle_histogram(self):
		self.histogram_open = not self.histogram_open
		return self.histogram_open

	def enable_normalize(self):
		self.normalizeImage = True
		self.autoI *= 2
		if self.autoI > 49:
			self.autoI = 0.05

		frame_data = self.get_frame_for_display()
		if frame_data is None or not self.height or not self.width:
			return self.normalizeImage

		frame = np.frombuffer(frame_data, dtype=self.dtype)
		expected_pixels = int(self.height) * int(self.width)
		if frame.size < expected_pixels:
			return self.normalizeImage
		frame = frame[:expected_pixels].reshape((self.height, self.width))

		if self.dFoF_open and self.F0 is not None and getattr(self.F0, 'shape', None) == frame.shape:
			data = (frame.astype(np.float32) - self.F0) / self.F0
			data = np.nan_to_num(data, nan=0.0)
		else:
			data = frame.astype(np.float32)

		self.minI = float(np.percentile(data, self.autoI))
		self.maxI = float(np.percentile(data, 100 - self.autoI))
		if self.maxI <= self.minI:
			self.maxI = self.minI + 1.0
		return self.normalizeImage

	def adjust_dynamic_range(self):
		if self.normalizeImage:
			self.normalizeImage = False
			return self.normalizeImage

		return self.enable_normalize()

	def get_exp_params(self, exp_name=None, experiment_id=None, mouse_id=None, preview=False, save_outputs=False):
		self.exp_name = exp_name
		self.experiment_id = experiment_id
		self.mouse_id = mouse_id
		if preview:
			return self._send_command('preview_experiment', (exp_name, experiment_id, mouse_id, bool(save_outputs)))
		return self._send_command('start_experiment', (exp_name, experiment_id, mouse_id))

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
			if self.process.is_alive():
				self.process.terminate()
				self.process.join(timeout=2.0)
			self.process = None

class CameraGUI(QMainWindow):
	histogram_frame_ready = pyqtSignal(object)
	histogram_close_requested = pyqtSignal()

	def __init__(self):
		super().__init__()
		self.app_version = load_app_version()
		self.config = load_camera_config(str(CONFIG_DIR / 'cam_config.yaml'))
		self.plot_experiment_qc = self._load_plot_experiment_qc_flag()
		self.display_target_size = None
		self.camera_app = None
		self.teensy_controller = None
		self.teensy_active = False
		self.hardware_trigger_enabled = False
		self.preview_mode = False
		self.preview_exp_thread = None
		self.preview_process = None
		self.preview_status_queue = queue.Queue()
		self.current_exp_config = None
		self.current_cam_config = None
		self.current_teensy_config = None
		self.reset_display_average_on_next_frame = False
		self.reset_fps_on_next_frame = False
		self._auto_stop_pending = False
		self.highlight_special_pixels = True
		self.histogram_open = False
		self.histogram_thread_running = False
		self.histogram_thread = None
		self.histogram_window = None
		self.histogram_label = None
		self.histogram_info_label = None
		self.last_stats_time = None
		self.last_stats_frame_count = 0
		self.fps_samples = deque(maxlen=100)
		self.experiment_list_cache = None
		self.last_display_frame = None
		self.frc_result_windows = []
		self.select_roi_action = None
		self.reset_roi_action = None
		self.framegrab_backend_cb = None
		self.exp_status_timer = QTimer()
		self.exp_status_timer.timeout.connect(self.check_experiment_status)
		self.exp_status_timer.start(100)
		self.histogram_frame_ready.connect(self._on_histogram_frame_ready)
		self.histogram_close_requested.connect(self._close_histogram_window)
		self._last_experiment_display_time = 0.0
		self._last_plotted_metadata_key = None
		refresh_rate_cfg = self.config.get('DISPLAY_REFRESH_RATE', 10)
		try:
			self.display_refresh_rate = float(refresh_rate_cfg)
		except (TypeError, ValueError):
			self.display_refresh_rate = 10.0
		if self.display_refresh_rate <= 0:
			self.display_refresh_rate = 10.0
		self.display_refresh_interval_s = 1.0 / self.display_refresh_rate
		self.init_ui()
		self.load_and_display_camera_config()
		self.load_and_display_teensy_config()
		self.populate_experiment_controls()
		self.load_and_display_exp_config()
		self.setup_timer()

	def _load_plot_experiment_qc_flag(self):
		cfg_path = CONFIG_DIR / 'config.yaml'
		try:
			with open(cfg_path, 'r', encoding='utf-8') as file:
				cfg = yaml.safe_load(file) or {}
		except Exception:
			return False

		value = cfg.get('PLOT_EXPERIMENT_QC', False)
		if isinstance(value, str):
			return value.strip().lower() in ('1', 'true', 'yes', 'on')
		return bool(value)

	def _schedule_auto_stop(self):
		if self._auto_stop_pending:
			return
		self._auto_stop_pending = True
		QTimer.singleShot(0, self._auto_stop_after_logic_analyzer)

	def resizeEvent(self, event):
		super().resizeEvent(event)
		# Only update display target size when the window size actually changes.
		if event.oldSize() != event.size() and hasattr(self, 'video_label'):
			self.display_target_size = self.video_label.size()

	def check_experiment_status(self):
		self._drain_preview_status_queue()
		if self.camera_app and hasattr(self.camera_app, 'get_exp_status'):
			messages = self.camera_app.get_exp_status()
			for msg in messages:
				if msg[0] == "status":
					status_text = str(msg[1])
					if self.plot_experiment_qc and status_text.startswith("Timestamp gap summary:"):
						self._plot_latest_experiment_metadata()
					self.update_status(msg[1])
				elif msg[0] == "trial":
					self.update_status(msg[3])
				elif msg[0] in ("logic_analyzer_terminated", "experiment_finished"):
					self._schedule_auto_stop()

	def _drain_preview_status_queue(self):
		while True:
			try:
				msg = self.preview_status_queue.get_nowait()
			except queue.Empty:
				break

			if not msg:
				continue

			if msg[0] == 'status':
				self.update_status(msg[1])
			elif msg[0] == 'frc_saved_figures':
				self._show_frc_saved_figures(msg[1])
			elif msg[0] == 'finished':
				self._finish_preview_without_camera()

	def _show_frc_saved_figures(self, image_paths):
		valid_paths = [Path(p) for p in (image_paths or []) if Path(p).is_file()]
		if not valid_paths:
			self.update_status('FRC completed, but no saved figure files were found.')
			return

		viewer = QWidget(self, Qt.WindowType.Window)
		viewer.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
		viewer.setWindowTitle('FRC Results')

		layout = QVBoxLayout()
		viewer.setLayout(layout)

		for image_path in valid_paths:
			label_title = QLabel(image_path.name)
			layout.addWidget(label_title)

			image_label = QLabel()
			image_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
			pixmap = QPixmap(str(image_path))
			if pixmap.isNull():
				image_label.setText(f'Could not load image: {image_path}')
			else:
				if pixmap.width() > 900:
					pixmap = pixmap.scaledToWidth(900, Qt.TransformationMode.SmoothTransformation)
				image_label.setPixmap(pixmap)
			layout.addWidget(image_label)

		viewer.resize(980, 780)
		viewer.show()
		self.frc_result_windows.append(viewer)

	def _plot_latest_experiment_metadata(self):
		if self.camera_app is None:
			return
		if plt is None:
			self.update_status('Metadata plot skipped: matplotlib is not available in this environment.')
			return

		mouse_id = (getattr(self.camera_app, 'mouse_id', None) or '').strip()
		experiment_id = (getattr(self.camera_app, 'experiment_id', None) or '').strip()
		exp_name = (getattr(self.camera_app, 'exp_name', None) or '').strip()
		filename = (getattr(self.camera_app, 'active_save_filename', None) or '').strip()

		if not mouse_id or not experiment_id:
			return
		if not filename:
			filename = f"{mouse_id}_{experiment_id}"

		plot_key = (mouse_id, experiment_id, filename)
		if self._last_plotted_metadata_key == plot_key:
			return

		try:
			save_root = load_save_root()
		except Exception as exc:
			self.update_status(f'Could not resolve save directory for metadata plot: {exc}.')
			return

		meta_path = Path(save_root) / mouse_id / experiment_id / f"{filename}.npy"
		if not meta_path.is_file():
			self.update_status(f'Metadata plot skipped: metadata file not found ({meta_path}).')
			return

		try:
			metadata = np.load(str(meta_path), allow_pickle=True).item()
			if not isinstance(metadata, dict):
				raise ValueError('metadata payload is not a dict')
		except Exception as exc:
			self.update_status(f'Failed to load metadata for plotting ({meta_path}): {exc}.')
			return

		sys_clock_timestamps = np.asarray(metadata.get('sys_clock_timestamps', []), dtype=np.float64)
		frame_timestamps = np.asarray(metadata.get('frame_timestamps', []), dtype=np.float64)
		temp = np.asarray(metadata.get('sensor_temperatures_c', []), dtype=np.float64)
		callback_timing = metadata.get('callback_timing', {}) if isinstance(metadata.get('callback_timing', {}), dict) else {}
		tsamps = np.asarray(callback_timing.get('samples_ms', []), dtype=np.float64)

		title_text = f"{mouse_id}_{exp_name or experiment_id}"
		fig, axes = plt.subplots(2, 2, num=f"Experiment Summary - {title_text}")

		axes[0, 0].plot(np.diff(sys_clock_timestamps), '.-')
		axes[0, 0].set_xlabel('frame')
		axes[0, 0].set_ylabel('sys clock timestamp diff')

		axes[0, 1].plot(np.diff(frame_timestamps), '.-')
		axes[0, 1].set_xlabel('frame')
		axes[0, 1].set_ylabel('frame timestamps diff')

		axes[1, 0].plot(temp, '.-')
		axes[1, 0].set_xlabel('frame')
		axes[1, 0].set_ylabel('Temperature (c)')

		axes[1, 1].plot(tsamps, '.-')
		axes[1, 1].set_xlabel('frame')
		axes[1, 1].set_ylabel('framegrabber time (ms)')

		fig.suptitle(title_text)
		fig.tight_layout()
		plt.show(block=False)
		try:
			fig.canvas.draw_idle()
			plt.pause(0.001)
		except Exception:
			pass

		self._last_plotted_metadata_key = plot_key
		self.update_status(f'Opened metadata summary plot from {meta_path}.')

	def init_ui(self):
		self.setWindowTitle(f'camstim {self.app_version}')
		self.setGeometry(100, 100, 1400, 800)
		self._create_menu_bar()

		central_widget = QWidget()
		self.setCentralWidget(central_widget)

		main_layout = QHBoxLayout()
		central_widget.setLayout(main_layout)

		self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
		self.main_splitter.setChildrenCollapsible(False)
		self._splitter_adjust_guard = False
		self._main_splitter_last_sizes = []

		left_panel = self.create_display_panel()
		self.main_splitter.addWidget(left_panel)

		controls_panel = self.create_control_panel()
		controls_panel.setMinimumWidth(420)
		self.main_splitter.addWidget(controls_panel)

		status_panel = self.create_status_panel()
		status_panel.setMinimumWidth(320)
		self.main_splitter.addWidget(status_panel)

		# Column 1 and 3 are flexible; column 2 keeps at least its current width.
		self.main_splitter.setStretchFactor(0, 2)
		self.main_splitter.setStretchFactor(1, 0)
		self.main_splitter.setStretchFactor(2, 1)
		self.main_splitter.setSizes([900, 420, 540])
		self._main_splitter_last_sizes = self.main_splitter.sizes()
		self.main_splitter.splitterMoved.connect(self._on_main_splitter_moved)

		main_layout.addWidget(self.main_splitter)

	def _on_main_splitter_moved(self, pos, index):
		if self._splitter_adjust_guard or not hasattr(self, 'main_splitter'):
			return

		sizes = self.main_splitter.sizes()
		if len(sizes) != 3:
			self._main_splitter_last_sizes = sizes
			return

		# When dragging handle between column 2 and 3, keep column 2 width stable
		# and let column 1 absorb the delta so column 3 resize affects column 1.
		if index == 2 and len(self._main_splitter_last_sizes) == 3:
			prev_col2 = self._main_splitter_last_sizes[1]
			delta_col2 = sizes[1] - prev_col2
			if delta_col2 != 0:
				new_col1 = max(200, sizes[0] + delta_col2)
				new_col2 = max(420, prev_col2)
				new_col3 = max(220, sizes[2])
				self._splitter_adjust_guard = True
				try:
					self.main_splitter.setSizes([new_col1, new_col2, new_col3])
				finally:
					self._splitter_adjust_guard = False
				sizes = self.main_splitter.sizes()

		self._main_splitter_last_sizes = sizes

	def _create_menu_bar(self):
		menu_bar = self.menuBar()
		file_menu = menu_bar.addMenu('File')
		image_menu = menu_bar.addMenu('Image')

		quit_action = QAction('Quit', self)
		quit_action.triggered.connect(self.close)
		file_menu.addAction(quit_action)

		save_menu = image_menu.addMenu('Save')

		save_current_action = QAction('Current frame', self)
		save_current_action.triggered.connect(self.save_current_frame_image)
		save_menu.addAction(save_current_action)

		save_average_action = QAction('Average frame', self)
		save_average_action.triggered.connect(self.save_average_frame_image)
		save_menu.addAction(save_average_action)

		save_snapshot_action = QAction('Snapshot', self)
		save_snapshot_action.triggered.connect(self.save_snapshot_image)
		save_menu.addAction(save_snapshot_action)

		self.select_roi_action = QAction('Select ROI', self)
		self.select_roi_action.setEnabled(False)
		self.select_roi_action.triggered.connect(self.start_roi_selection)
		image_menu.addAction(self.select_roi_action)

		self.reset_roi_action = QAction('Reset ROI', self)
		self.reset_roi_action.setEnabled(False)
		self.reset_roi_action.triggered.connect(self.reset_roi)
		image_menu.addAction(self.reset_roi_action)

	def _prompt_image_save_path(self, title):
		default_dir = self._default_image_save_dir()
		file_path, _ = QFileDialog.getSaveFileName(
			self,
			title,
			default_dir,
			'TIFF (*.tiff *.tif);;PNG (*.png);;BMP (*.bmp);;All Files (*)',
		)
		if not file_path:
			return None

		path = Path(file_path)
		if path.suffix == '':
			path = path.with_suffix('.tiff')
		return str(path)

	def _default_image_save_dir(self):
		try:
			save_root = Path(load_save_root())
		except Exception:
			save_root = CONFIG_DIR

		mouse_id = ''
		experiment_id = ''

		if self.camera_app is not None:
			mouse_id = (getattr(self.camera_app, 'mouse_id', '') or '').strip()
			experiment_id = (getattr(self.camera_app, 'experiment_id', '') or '').strip()

		if not mouse_id and hasattr(self, 'mouse_id_input'):
			mouse_id = self.mouse_id_input.text().strip()
		if not experiment_id and hasattr(self, 'experiment_id_input'):
			experiment_id = self.experiment_id_input.text().strip()

		default_dir = save_root
		if mouse_id and experiment_id:
			default_dir = save_root / mouse_id / experiment_id

		try:
			default_dir.mkdir(parents=True, exist_ok=True)
		except OSError:
			pass

		return str(default_dir)

	def _save_image_to_path(self, frame, file_path):
		ok = cv2.imwrite(file_path, frame)
		if not ok:
			raise OSError(f'cv2.imwrite returned False for {file_path}')

	def save_current_frame_image(self):
		if self.camera_app is None:
			QMessageBox.warning(self, 'Save Current Frame', 'Camera is not running.')
			return

		frame = self.last_display_frame
		if frame is None:
			try:
				frame_data = self.camera_app.get_frame_for_display()
				if frame_data is not None:
					frame = self._frame_data_to_display_frame(frame_data)
			except Exception:
				frame = None

		if frame is None:
			QMessageBox.warning(self, 'Save Current Frame', 'No frame is currently available to save.')
			return

		file_path = self._prompt_image_save_path('Save Current Frame')
		if not file_path:
			return

		try:
			self._save_image_to_path(frame, file_path)
			self.update_status(f'Saved current frame: {file_path}')
		except Exception as e:
			QMessageBox.critical(self, 'Save Current Frame', f'Failed to save image: {e}')

	def _capture_next_display_frames(self, frame_count, timeout_s=10.0):
		if self.camera_app is None or frame_count <= 0:
			return []

		frames = []
		timer_was_active = hasattr(self, 'timer') and self.timer.isActive()
		timer_interval_ms = self.timer.interval() if timer_was_active else 0

		if timer_was_active:
			self.timer.stop()

		deadline = time.time() + max(timeout_s, frame_count * 0.15)
		try:
			while len(frames) < frame_count and time.time() < deadline:
				frame_data = self.camera_app.get_frame_for_display()
				if frame_data is None:
					QApplication.processEvents()
					time.sleep(0.005)
					continue

				display_frame = self._frame_data_to_display_frame(frame_data)
				if display_frame is None:
					continue

				frames.append(display_frame.astype(np.float32))
				QApplication.processEvents()
		finally:
			if timer_was_active:
				self.timer.start(timer_interval_ms)

		return frames

	def save_average_frame_image(self):
		if self.camera_app is None:
			QMessageBox.warning(self, 'Save Average Frame', 'Camera is not running.')
			return

		frame_count, ok = QInputDialog.getInt(
			self,
			'Save Average Frame',
			'Number of frames to average:',
			10,
			1,
			10000,
			1,
		)
		if not ok:
			return

		file_path = self._prompt_image_save_path('Save Average Frame')
		if not file_path:
			return

		frames = self._capture_next_display_frames(frame_count)
		if len(frames) < frame_count:
			QMessageBox.warning(
				self,
				'Save Average Frame',
				f'Only captured {len(frames)} of {frame_count} frame(s). Try again.',
			)
			return

		avg_frame = np.mean(np.stack(frames, axis=0), axis=0)
		avg_frame = np.clip(np.rint(avg_frame), 0, 255).astype(np.uint8)

		try:
			self._save_image_to_path(avg_frame, file_path)
			self.update_status(f'Saved average frame ({frame_count}): {file_path}')
		except Exception as e:
			QMessageBox.critical(self, 'Save Average Frame', f'Failed to save image: {e}')

	def save_snapshot_image(self):
		pixmap = self.video_label.pixmap() if hasattr(self, 'video_label') else None
		if pixmap is None or pixmap.isNull():
			QMessageBox.warning(self, 'Save Snapshot', 'No displayed image is available to snapshot.')
			return

		file_path = self._prompt_image_save_path('Save Snapshot')
		if not file_path:
			return

		try:
			if not pixmap.save(file_path):
				raise OSError(f'QPixmap.save returned False for {file_path}')
			self.update_status(f'Saved snapshot: {file_path}')
		except Exception as e:
			QMessageBox.critical(self, 'Save Snapshot', f'Failed to save snapshot: {e}')

	def start_roi_selection(self):
		if self.camera_app is None:
			QMessageBox.warning(self, 'Select ROI', 'Camera is not running.')
			return
		if self.camera_app.saving or self.hardware_trigger_enabled:
			QMessageBox.warning(self, 'Select ROI', 'ROI selection is only available in live view when not saving.')
			return
		if self.last_display_frame is None:
			QMessageBox.warning(self, 'Select ROI', 'No displayed frame is available yet.')
			return
		self.video_label.set_roi_mode_enabled(True)
		self.update_status('ROI selection enabled. Drag a rectangle on the image to apply it.')

	def apply_selected_roi(self, selection_rect, image_rect):
		self.video_label.set_roi_mode_enabled(False)
		if self.camera_app is None or self.last_display_frame is None:
			return
		if image_rect.isNull() or image_rect.width() <= 0 or image_rect.height() <= 0:
			return

		camera_width = int(getattr(self.camera_app, 'width', 0) or 0)
		camera_height = int(getattr(self.camera_app, 'height', 0) or 0)
		if camera_width <= 0 or camera_height <= 0:
			QMessageBox.warning(self, 'Select ROI', 'Camera resolution is unavailable.')
			return

		x = round((selection_rect.left() - image_rect.left()) * camera_width / image_rect.width())
		y = round((selection_rect.top() - image_rect.top()) * camera_height / image_rect.height())
		width = round(selection_rect.width() * camera_width / image_rect.width())
		height = round(selection_rect.height() * camera_height / image_rect.height())

		x = max(0, min(x, camera_width - 2))
		y = max(0, min(y, camera_height - 2))
		width = max(2, min(width, camera_width - x))
		height = max(2, min(height, camera_height - y))
		requested_width = width
		requested_height = height

		# When live binning is enabled, ROI dimensions must be multiples of BIN_SIZE.
		if bool(getattr(self.camera_app, 'bin_exp', False)):
			bs = int(getattr(self.camera_app, 'bin_size', 1) or 1)
			if bs > 1:
				width -= (width % bs)
				height -= (height % bs)
				if width < bs or height < bs:
					QMessageBox.warning(
						self,
						'Select ROI',
						f'Selected ROI is too small for BIN_SIZE={bs}. Increase ROI size.',
					)
					return
				if width != requested_width or height != requested_height:
					self.update_status(
						f'Adjusted ROI to {width}x{height} to match BIN_SIZE={bs} '
						f'(requested {requested_width}x{requested_height}).'
					)

		if width <= 1 or height <= 1:
			QMessageBox.warning(self, 'Select ROI', 'Selected ROI is too small.')
			return

		self.reset_display_average_on_next_frame = True
		self.reset_fps_on_next_frame = True
		self.last_display_frame = None
		self.camera_app.set_roi(x, y, width, height)
		self.update_status(f'Requested ROI: {width}x{height} at ({x}, {y}).')

	def reset_roi(self):
		if self.camera_app is None:
			QMessageBox.warning(self, 'Reset ROI', 'Camera is not running.')
			return
		if self.camera_app.saving or self.hardware_trigger_enabled:
			QMessageBox.warning(self, 'Reset ROI', 'ROI reset is only available in live view when not saving.')
			return
		self.video_label.set_roi_mode_enabled(False)
		self.reset_display_average_on_next_frame = True
		self.reset_fps_on_next_frame = True
		self.last_display_frame = None
		self.camera_app.reset_roi()
		self.update_status('Requested ROI reset to full frame.')

	def create_display_panel(self):
		panel = QWidget()
		layout = QVBoxLayout()
		panel.setLayout(layout)

		self.video_label = RoiSelectableLabel()
		self.video_label.selection_finished_callback = self.apply_selected_roi
		self.video_label.setText(f'camstim {self.app_version}\n\nCamera not started.\n\nClick "Start Camera" to begin.')
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
		self.exposure_spin.setFixedWidth(90)
		self.exposure_spin.setValue(self.config['EXPOSURE_TIME'])
		self.exposure_spin.valueChanged.connect(self.update_exposure)
		exp_layout.addWidget(self.exposure_spin)

		gain_layout = QHBoxLayout()
		gain_layout.addWidget(QLabel("Gain:"))
		self.gain_spin = QDoubleSpinBox()
		self.gain_spin.setRange(0.1, 100.0)
		self.gain_spin.setDecimals(3)
		self.gain_spin.setSingleStep(0.1)
		self.gain_spin.setFixedWidth(90)
		self.gain_spin.setKeyboardTracking(False)
		self.gain_spin.setValue(float(self.config['ANALOG_GAIN']))
		self.gain_spin.setEnabled(False)
		self.gain_spin.valueChanged.connect(self.update_gain)
		gain_layout.addWidget(self.gain_spin)

		camera_row_layout = QHBoxLayout()
		camera_row_layout.addLayout(exp_layout)
		camera_row_layout.addSpacing(12)
		camera_row_layout.addLayout(gain_layout)
		camera_row_layout.addStretch()
		settings_layout.addLayout(camera_row_layout)

		settings_group.setLayout(settings_layout)
		layout.addWidget(settings_group)

		# Experiment controls
		exp_group = QGroupBox("Experiment Controls")
		exp_layout = QVBoxLayout()

		exp_select_layout = QHBoxLayout()
		exp_select_layout.addWidget(QLabel("Experiment:"))
		self.exp_combo = QComboBox()
		self.exp_combo.setEnabled(False)
		self.exp_combo.currentTextChanged.connect(self.on_experiment_selection_changed)
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
		proc_layout = QGridLayout()

		self.autonorm_btn = QPushButton("Auto Norm")
		self.autonorm_btn.clicked.connect(self.apply_auto_norm)
		proc_layout.addWidget(self.autonorm_btn, 0, 0)

		self.disablenorm_btn = QPushButton("Disable Norm")
		self.disablenorm_btn.clicked.connect(self.disable_norm)
		proc_layout.addWidget(self.disablenorm_btn, 0, 1)

		self.histogram_btn = QPushButton("Show Histogram")
		self.histogram_btn.clicked.connect(self.toggle_histogram)
		proc_layout.addWidget(self.histogram_btn, 1, 0)

		self.frc_btn = QPushButton("Fourier Ring Correlation")
		self.frc_btn.clicked.connect(self.show_fourier_ring_correlation)
		proc_layout.addWidget(self.frc_btn, 1, 1)

		self.background_cb = QCheckBox("Remove Background")
		self.background_cb.stateChanged.connect(self.toggle_background)
		proc_layout.addWidget(self.background_cb, 2, 0)

		self.dfof_cb = QCheckBox("Enable dFoF")
		self.dfof_cb.stateChanged.connect(self.toggle_dfof)
		proc_layout.addWidget(self.dfof_cb, 2, 1)

		self.highlight_pixels_cb = QCheckBox("Highlight 0/255 Pixels")
		self.highlight_pixels_cb.setChecked(True)
		self.highlight_pixels_cb.stateChanged.connect(self.toggle_special_pixel_highlight)
		proc_layout.addWidget(self.highlight_pixels_cb, 3, 0)

		self.display_output_cb = QCheckBox("Live Display Updates")
		self.display_output_cb.setChecked(True)
		self.display_output_cb.stateChanged.connect(self.toggle_display_output)
		proc_layout.addWidget(self.display_output_cb, 3, 1)

		proc_group.setLayout(proc_layout)
		layout.addWidget(proc_group)

		layout.addStretch()

		return panel

	def create_status_panel(self):
		panel = QWidget()
		layout = QVBoxLayout()
		panel.setLayout(layout)
		panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

		status_group = QGroupBox("Status")
		status_layout = QVBoxLayout()
		status_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

		self.status_text = QTextEdit()
		self.status_text.setReadOnly(True)
		self.status_text.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
		status_layout.addWidget(self.status_text)

		self.stats_label = QLabel(f"FPS: 0 | Frames: 0 | Saved: 0 | Save Queue: 0")
		status_layout.addWidget(self.stats_label)

		status_group.setLayout(status_layout)
		layout.addWidget(status_group)
		self.statusBar().showMessage(f"camstim {self.app_version}")
		return panel

	def populate_experiment_controls(self, force_refresh=False):
		previous_selection = self.exp_combo.currentText().strip()

		if force_refresh or self.experiment_list_cache is None:
			experiment_list = []
			if self.camera_app and getattr(self.camera_app, 'exp_list', None):
				experiment_list = self.camera_app.exp_list
			else:
				try:
					experiment_list = get_experiment_list()
				except Exception as e:
					self.update_status(f"Error loading experiment list: {e}.")
			if experiment_list:
				self.experiment_list_cache = list(experiment_list)

		experiment_list = self.experiment_list_cache or []

		self.exp_combo.blockSignals(True)
		self.exp_combo.clear()
		# Keep index 0 empty so the user must explicitly choose an experiment.
		self.exp_combo.addItem("")

		if not experiment_list:
			self.exp_combo.setCurrentIndex(0)
			self.exp_combo.setEnabled(False)
			self.exp_combo.blockSignals(False)
			self.load_and_display_exp_config()
			return

		self.exp_combo.addItems(experiment_list)
		if previous_selection and previous_selection in experiment_list:
			self.exp_combo.setCurrentText(previous_selection)
		else:
			self.exp_combo.setCurrentIndex(0)
		self.exp_combo.setEnabled(True)
		self.exp_combo.blockSignals(False)
		self.load_and_display_exp_config()

	def _should_enable_experiment_button(self):
		"""Check if experiment button should be enabled based on selected experiment and trigger mode."""
		exp_name = self.exp_combo.currentText().strip()
		if not exp_name:
			return False
		
		# Check if this experiment type requires hardware trigger
		exp_types = discover_experiment_types()
		exp_class = exp_types.get(exp_name)
		requires_hw_trigger = getattr(exp_class, 'requires_hardware_trigger', True)
		
		# Enable button if:
		# 1. Experiment doesn't require hardware trigger (like Continuous), OR
		# 2. Hardware trigger is enabled
		return not requires_hw_trigger or self.hardware_trigger_enabled

	def _should_enable_preview_button(self):
		if self.preview_mode:
			return False
		return bool(self.exp_combo.currentText().strip())

	def on_experiment_selection_changed(self, _text):
		self.load_and_display_exp_config()
		# Update button enablement when experiment selection changes
		self.preview_btn.setEnabled(self._should_enable_preview_button())
		if self.camera_app:
			self.exp_btn.setEnabled(self._should_enable_experiment_button())

	def _track_preview_process_output(self, process):
		for line in process.stdout:
			line = line.strip()
			if line:
				self.preview_status_queue.put(('status', line))

		if process.poll() not in (0, None):
			self.preview_status_queue.put(('status', f'Preview subprocess exited with code {process.returncode}.'))

		self.preview_status_queue.put(('finished',))

	def _track_frc_worker(self, process, log_path):
		process.wait()
		all_lines = []
		log_tail = []
		try:
			with open(log_path, 'r', encoding='utf-8', errors='replace') as fh:
				all_lines = [line.strip() for line in fh.readlines() if line.strip()]
				log_tail = all_lines[-20:]
		except OSError:
			all_lines = []
			log_tail = []

		if process.returncode == 0:
			saved_paths = [line for line in all_lines if line.startswith('/') and line.lower().endswith('.png')]
			if saved_paths:
				self.preview_status_queue.put(('status', f'FRC completed. Figure saved to {saved_paths[0]}'))
				self.preview_status_queue.put(('frc_saved_figures', saved_paths))
			else:
				self.preview_status_queue.put(('status', 'FRC completed.'))
			return

		msg = f'FRC worker exited with code {process.returncode}. See log: {log_path}'
		if log_tail:
			msg = f"{msg} | Last log line: {log_tail[-1]}"
		self.preview_status_queue.put(('status', msg))

	def _plot_frc_numpy(self, img1, img2):
		if plt is None:
			raise RuntimeError('matplotlib is not available in this environment')

		try:
			import frc
		except Exception as exc:
			raise RuntimeError(f'frc module import failed: {exc}')

		img1_proc = frc.util.square_image(np.array(img1), add_padding=False)
		img2_proc = frc.util.square_image(np.array(img2), add_padding=False)
		img1_proc = frc.util.apply_tukey(img1_proc)
		img2_proc = frc.util.apply_tukey(img2_proc)

		two_frc_curve = frc.two_frc(img1_proc, img2_proc)
		one_frc_curve = frc.one_frc(img1_proc)
		img_size = int(img1_proc.shape[0])
		xs_freq = np.arange(len(two_frc_curve), dtype=np.float64) / float(img_size)

		def _draw_curve(curve, title):
			frc_res = None
			threshold_fn = None
			try:
				frc_res, _, threshold_fn = frc.frc_res(xs_freq, curve, img_size)
			except Exception:
				frc_res = None
				threshold_fn = None

			fig, ax = plt.subplots(num=title)
			ax.clear()
			ax.plot(xs_freq, curve, label='FRC')
			if threshold_fn is not None:
				ax.plot(xs_freq, threshold_fn(xs_freq), label='Threshold')
			if frc_res is not None:
				ax.axvline(float(frc_res), color='r', linestyle='--', label=f'Resolution {float(frc_res):.4f}')
			ax.set_xlabel('Spatial Frequency (pixel^-1)')
			ax.set_ylabel('FRC')
			ax.set_title(title)
			ax.legend()
			fig.tight_layout()
			return fig

		fig1 = _draw_curve(two_frc_curve, 'Fourier Ring Correlation (two consecutive frames)')
		fig2 = _draw_curve(one_frc_curve, 'Fourier Ring Correlation (single frame)')

		plt.show(block=False)
		for fig in (fig1, fig2):
			try:
				fig.canvas.draw_idle()
			except Exception:
				pass
		try:
			plt.pause(0.001)
		except Exception:
			pass

	def _finish_preview_without_camera(self):
		self.preview_mode = False
		self.preview_process = None
		self.preview_btn.setEnabled(self._should_enable_preview_button())
		self.stop_preview_btn.setEnabled(False)
		self.config_display.setText("Experiment stopped. \n\nStart/preview experiment to load configuration.")
		self.current_exp_config = None

	def _start_preview_without_camera(self, exp_name, experiment_id, mouse_id, save_outputs=False):
		if self.preview_process is not None and self.preview_process.poll() is None:
			self.update_status('Error: Experiment preview is already running.')
			return False

		skip_teensy_enabled = bool(self.config.get('DEBUG_SKIP_TEENSY', False))
		try:
			exp_types = discover_experiment_types()
			exp_class = exp_types.get(exp_name)
			skip_teensy_enabled = skip_teensy_enabled or getattr(exp_class, 'skip_teensy', False)
		except Exception:
			pass

		wf_script = REPO_ROOT / 'utils' / 'wf_main.py'
		cmd = [sys.executable, '-u', str(wf_script), exp_name, experiment_id, mouse_id]
		cmd.append('--preview')
		if save_outputs:
			cmd.append('--save-preview')
		if skip_teensy_enabled:
			cmd.append('--skip-teensy')

		self.preview_process = subprocess.Popen(
			cmd,
			stdout=subprocess.PIPE,
			stderr=subprocess.STDOUT,
			stdin=subprocess.PIPE,
			text=True,
			cwd=str(REPO_ROOT),
		)
		threading.Thread(target=self._track_preview_process_output, args=(self.preview_process,), daemon=True).start()
		return True

	def _preview_should_save_outputs(self):
		experiment_id = self.experiment_id_input.text().strip()
		mouse_id = self.mouse_id_input.text().strip()
		return bool(experiment_id and mouse_id)

	def _normalize_preview_identity(self, experiment_id, mouse_id):
		experiment_id = (experiment_id or '').strip() or 'preview'
		mouse_id = (mouse_id or '').strip() or 'preview'
		return experiment_id, mouse_id

	def get_selected_experiment_params(self, allow_empty_ids=False):
		exp_name = self.exp_combo.currentText().strip()
		experiment_id = self.experiment_id_input.text().strip()
		mouse_id = self.mouse_id_input.text().strip()

		if not exp_name:
			self.update_status("Error: Select an experiment from the dropdown.")
			return None

		if allow_empty_ids:
			experiment_id, mouse_id = self._normalize_preview_identity(experiment_id, mouse_id)
			return exp_name, experiment_id, mouse_id

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

	def _validate_exposure_timing_before_start(self):
		teensy_path = CONFIG_DIR / 'teensyParams.yaml'
		try:
			with open(teensy_path, 'r') as f:
				teensy_cfg = yaml.safe_load(f) or {}
		except Exception as e:
			msg = f"Could not read Teensy timing config ({teensy_path}): {e}."
			QMessageBox.warning(self, "Start Camera Blocked", msg)
			self.update_status(msg)
			return False

		try:
			f_led_hz = float(teensy_cfg['F_LED'])
			o_cam_us = float(teensy_cfg['O_CAM'])
			exposure_ms = float(self.exposure_spin.value())
		except (KeyError, TypeError, ValueError) as e:
			msg = (
				"Invalid Teensy timing parameters. Ensure F_LED (Hz) and O_CAM (us) are valid numeric values in "
				f"{teensy_path}."
			)
			QMessageBox.warning(self, "Start Camera Blocked", msg)
			self.update_status(msg)
			return False

		if f_led_hz <= 0:
			msg = f"Invalid F_LED={f_led_hz}. F_LED must be > 0 Hz."
			QMessageBox.warning(self, "Start Camera Blocked", msg)
			self.update_status(msg)
			return False

		max_exposure_ms = (1000.0 / f_led_hz) - (o_cam_us / 1000.0)
		if max_exposure_ms <= 0:
			msg = (
				"Invalid Teensy timing: computed max exposure is <= 0 ms. "
				f"(F_LED={f_led_hz} Hz, O_CAM={o_cam_us} us, max={max_exposure_ms:.3f} ms)"
			)
			QMessageBox.warning(self, "Start Camera Blocked", msg)
			self.update_status(msg)
			return False

		if exposure_ms >= max_exposure_ms:
			msg = (
				"Exposure timing check failed. "
				f"Exposure ({exposure_ms:.3f} ms) must be smaller than (1/F_LED - O_CAM) = {max_exposure_ms:.3f} ms "
				f"(F_LED={f_led_hz:g} Hz, O_CAM={o_cam_us:g} us)."
			)
			QMessageBox.warning(self, "Start Camera Blocked", msg)
			self.update_status(msg)
			return False

		return True

	def start_camera(self):
		if not self._validate_exposure_timing_before_start():
			return

		try:
			selected_experiment_before_start = self.exp_combo.currentText().strip()
			self.camera_app = CameraProcessClient(self.config)
			self.camera_app.start()
			self.display_target_size = self.video_label.size()
			self.last_stats_time = time.time()
			self.last_stats_frame_count = 0
			self.fps_samples.clear()

			def _on_logic_analyzer_terminated():
				self._schedule_auto_stop()
			self.camera_app.on_logic_analyzer_terminated = _on_logic_analyzer_terminated
			self.camera_app.on_experiment_finished = _on_logic_analyzer_terminated

			self.timer.start(30)
			self.stats_timer.start(100)
			self._update_autonorm_button_text()

			self.start_btn.setEnabled(False)
			self.stop_btn.setEnabled(True)
			if self.select_roi_action is not None:
				self.select_roi_action.setEnabled(True)
			if self.reset_roi_action is not None:
				self.reset_roi_action.setEnabled(True)
			self.trigger_btn.setEnabled(True)
			self.teensy_toggle_btn.setEnabled(True)
			self.teensy_active = False
			self.teensy_toggle_btn.setText("Enable Teensy")
			self.exp_btn.setEnabled(False)
			self.preview_btn.setEnabled(self._should_enable_preview_button())
			self.populate_experiment_controls(force_refresh=False)
			if selected_experiment_before_start and selected_experiment_before_start in (self.experiment_list_cache or []):
				self.exp_combo.setCurrentText(selected_experiment_before_start)

			# Preserve the GUI exposure value across camera restarts.
			exposure_value = float(self.exposure_spin.value())
			self.camera_app.exposure = exposure_value
			self.camera_app.set_exposure(exposure_value)
			if self.framegrab_backend_cb is not None:
				self.camera_app.set_framegrab_backend(self.framegrab_backend_cb.isChecked())
			# Always force continuous mode at startup so preview does not depend on
			# stale external-trigger state inside the camera SDK.
			self.camera_app.set_trigger_mode(False)
			self.hardware_trigger_enabled = False
			self.trigger_btn.setText("Enable Hardware Trigger")
			self.exp_btn.setEnabled(self._should_enable_experiment_button())
			if hasattr(self, 'display_output_cb'):
				self.camera_app.set_display_output_enabled(self.display_output_cb.isChecked())
			self._sync_gain_spinner_with_camera()

			self.load_and_display_camera_config()
			self.load_and_display_teensy_config()

			self.update_status("Camera started successfully in continuous mode.")
		except Exception as e:
			self.update_status(f"Error starting camera: {str(e)}.")

	def update_framegrab_backend(self, state):
		use_c_backend = bool(state == Qt.CheckState.Checked.value)
		self.config['USE_CGRABCALLBACK'] = use_c_backend
		if self.camera_app:
			self.camera_app.set_framegrab_backend(use_c_backend)
		self.update_status(
			f"Requested framegrab backend: {'C extension' if use_c_backend else 'Python fallback'}."
		)

	def _sync_gain_spinner_with_camera(self):
		if not self.camera_app or not hasattr(self, 'gain_spin'):
			return

		step = float(getattr(self.camera_app, 'analog_gain_step', 0.1))
		gmin = float(getattr(self.camera_app, 'analog_gain_min', 0.1))
		gmax = float(getattr(self.camera_app, 'analog_gain_max', 100.0))
		gval = float(getattr(self.camera_app, 'analog_gain', self.gain_spin.value()))

		if step <= 0:
			step = 0.1
		if gmin > gmax:
			gmin, gmax = gmax, gmin

		self.gain_spin.blockSignals(True)
		self.gain_spin.setEnabled(bool(getattr(self.camera_app, 'ready', False)))
		self.gain_spin.setRange(gmin, gmax)
		self.gain_spin.setSingleStep(step)
		decimals = 0
		tmp = step
		while decimals < 6 and abs(tmp - round(tmp)) > 1e-9:
			tmp *= 10.0
			decimals += 1
		self.gain_spin.setDecimals(max(1, min(6, decimals)))
		self.gain_spin.setValue(self._quantize_gain_value(gval, gmin, gmax, step))
		self.gain_spin.blockSignals(False)

	def _quantize_gain_value(self, value, gmin=None, gmax=None, step=None):
		if gmin is None:
			gmin = float(self.gain_spin.minimum())
		if gmax is None:
			gmax = float(self.gain_spin.maximum())
		if step is None:
			step = float(self.gain_spin.singleStep())

		if step <= 0:
			step = 0.1
		clamped = max(gmin, min(gmax, float(value)))
		steps_from_min = round((clamped - gmin) / step)
		quantized = gmin + (steps_from_min * step)
		return max(gmin, min(gmax, quantized))

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
			self.video_label.set_roi_mode_enabled(False)
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
		if self.select_roi_action is not None:
			self.select_roi_action.setEnabled(False)
		if self.reset_roi_action is not None:
			self.reset_roi_action.setEnabled(False)
		self.trigger_btn.setEnabled(False)
		self.teensy_toggle_btn.setEnabled(False)
		self._update_autonorm_button_text()
		self.teensy_toggle_btn.setText("Enable Teensy")
		self.hardware_trigger_enabled = False
		self.exp_btn.setEnabled(False)
		self.exp_combo.setEnabled(bool(self.experiment_list_cache))
		self.preview_btn.setEnabled(self._should_enable_preview_button())
		if hasattr(self, 'gain_spin'):
			self.gain_spin.setEnabled(False)
		self.video_label.setText(f"camstim {self.app_version}\n\nCamera Stopped.")
		self.display_target_size = None
		self.cam_config_display.setText("Camera stopped. \n\n Start camera to begin.")
		self.current_cam_config = None
		self.update_status("Camera Stopped.")

	def update_display(self):
		if self.camera_app:
			try:
				# Limit preview refresh during experiment (saving or hardware trigger enabled).
				experiment_active = getattr(self.camera_app, 'saving', False) or self.hardware_trigger_enabled
				if experiment_active:
					now = time.time()
					if now - self._last_experiment_display_time < self.display_refresh_interval_s:
						return
					self._last_experiment_display_time = now

				frame_data = self.camera_app.get_frame_for_display()
				if frame_data is None:
					return

				if self.reset_display_average_on_next_frame:
					self.reset_display_average_on_next_frame = False

				if self.reset_fps_on_next_frame:
					self.last_stats_time = time.time()
					self.last_stats_frame_count = self.camera_app.frame_count
					self.fps_samples.clear()
					self.reset_fps_on_next_frame = False

				display_frame = self._frame_data_to_display_frame(frame_data)
				if display_frame is None:
					return
				self.last_display_frame = np.ascontiguousarray(display_frame.copy())
				self.display_processed_frame(display_frame)
			except Exception as e:
				print(f"Display error: {e}.")

	def _frame_data_to_display_frame(self, frame_data):
		if self.camera_app is None or frame_data is None:
			return None

		frame = np.frombuffer(frame_data, dtype=self.camera_app.dtype)

		if hasattr(self.camera_app, 'height') and hasattr(self.camera_app, 'width'):
			expected_pixels = int(self.camera_app.height) * int(self.camera_app.width)
			if frame.size < expected_pixels:
				return None
			frame = frame[:expected_pixels].reshape((self.camera_app.height, self.camera_app.width))
			# Keep GUI display orientation identical to saved raw frames.
			# Prefer hardware mirror; fall back to software mirror if needed.
			if getattr(self.camera_app, 'software_mirror_horizontal', False):
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

		frame = self._prepare_display_frame(frame)
		return self.process_frame_for_display(frame)

	def _prepare_display_frame(self, frame):
		if self.camera_app.dtype == 'uint16':
			frame = frame.astype(np.float32)
		else:
			frame = frame.astype(np.float32, copy=False)
		return frame

	def process_frame_for_display(self, frame):
		if self.camera_app.removeBackground and getattr(self.camera_app, 'backgroundImg', None) is not None:
			bg_ref = self._align_reference_to_frame(self.camera_app.backgroundImg, frame.shape, avoid_zero=False)
			frame = frame - bg_ref
			frame = np.clip(frame, 0, 255)

		limit_min = float(getattr(self.camera_app, 'minI', 0.0))
		limit_max = float(getattr(self.camera_app, 'maxI', 255.0))
		if limit_max <= limit_min:
			limit_max = limit_min + 1.0
		clipped = np.clip(frame, limit_min, limit_max)
		frame = ((clipped - limit_min) / (limit_max - limit_min)) * 255

		if self.camera_app.dFoF_open and self.camera_app.F0 is not None:
			f0_ref = self._align_reference_to_frame(self.camera_app.F0, frame.shape, avoid_zero=True)
			dfof = (frame.astype(np.float32) - f0_ref) / f0_ref
			dfof = np.nan_to_num(dfof, nan=0.0)

			limit_min = float(getattr(self.camera_app, 'minI', 0.0))
			limit_max = float(getattr(self.camera_app, 'maxI', 255.0))
			if limit_max <= limit_min:
				limit_max = limit_min + 1.0
			clipped = np.clip(dfof, limit_min, limit_max)
			frame = ((clipped - limit_min) / (limit_max - limit_min)) * 255

		if hasattr(self.camera_app, 'height') and hasattr(self.camera_app, 'width'):
			frame = cv2.resize(frame, (self.camera_app.width//2, self.camera_app.height//2), interpolation=cv2.INTER_NEAREST)

		return frame.astype(np.uint8)

	def _align_reference_to_frame(self, ref, target_shape, avoid_zero=False):
		if ref is None:
			return ref
		if tuple(ref.shape) == tuple(target_shape):
			aligned = ref.astype(np.float32, copy=False)
			if avoid_zero:
				aligned = aligned.copy()
				aligned[aligned == 0] = 1e-6
			return aligned
		target_h, target_w = int(target_shape[0]), int(target_shape[1])
		aligned = cv2.resize(ref.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_AREA)
		if avoid_zero:
			aligned[aligned == 0] = 1e-6
		return aligned

	def display_processed_frame(self, frame):
		try:
			h, w = frame.shape
			if self.highlight_special_pixels:
				rgb_frame = np.stack([frame, frame, frame], axis=-1)

				# Highlight special values in the GUI only: 0 -> blue, 255 -> red.
				zero_mask = (frame == 0)
				sat_mask = (frame == 255)
				rgb_frame[zero_mask] = [0, 0, 255]
				rgb_frame[sat_mask] = [255, 0, 0]
				rgb_frame = np.ascontiguousarray(rgb_frame)
			else:
				rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

			bytes_per_line = 3 * w
			q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
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
				self._sync_gain_spinner_with_camera()
				worker_average_fps = getattr(self.camera_app, 'average_fps', None)
				if worker_average_fps is None:
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
				else:
					average_fps = float(worker_average_fps)

				save_queue_size = getattr(self.camera_app, 'save_queue_size', 0)
				sensor_temp = getattr(self.camera_app, 'sensor_temperature', None)
				temp_str = f" | Temp: {sensor_temp:.1f}°C" if sensor_temp is not None else ""
				stats_text = (
					f"FPS: {average_fps: .1f}{temp_str}\n"
					f"Frames: {self.camera_app.frame_count} | Saved: {self.camera_app.frames_written} | Save Queue: {save_queue_size}"
				)
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
					# Update button based on whether selected experiment needs hardware trigger
					self.exp_btn.setEnabled(self._should_enable_experiment_button())
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
		params = self.get_selected_experiment_params(allow_empty_ids=True)
		if params is None:
			return

		exp_name, experiment_id, mouse_id = params
		save_outputs = self._preview_should_save_outputs()

		self.preview_mode = True
		if self.camera_app:
			started = self.camera_app.get_exp_params(
				exp_name=exp_name,
				experiment_id=experiment_id,
				mouse_id=mouse_id,
				preview=True,
				save_outputs=save_outputs,
			)
		else:
			started = self._start_preview_without_camera(
				exp_name=exp_name,
				experiment_id=experiment_id,
				mouse_id=mouse_id,
				save_outputs=save_outputs,
			)
		if not started:
			self.preview_mode = False
			self.update_status('Error: Failed to start experiment preview subprocess.')
			return
		self.load_and_display_exp_config()
		self.preview_btn.setEnabled(False)
		self.stop_preview_btn.setEnabled(True)
		self.exp_btn.setEnabled(False)
		self.trigger_btn.setEnabled(False)

		self.update_status(f'Starting experiment preview: {exp_name}.')

	def stop_preview(self):
		self.preview_mode = False
		if self.camera_app:
			self.camera_app.stop_stim()
			self.preview_btn.setEnabled(self._should_enable_preview_button())
			self.stop_preview_btn.setEnabled(False)
			self.trigger_btn.setEnabled(True)
			self.update_status("Preview stopped.")
			self.config_display.setText("Experiment stopped. \n\nStart/preview experiment to load configuration.")
			self.current_exp_config = None
			return

		if self.preview_process is not None:
			try:
				if self.preview_process.poll() is None:
					self.preview_process.stdin.write("STOP\n")
					self.preview_process.stdin.flush()
					self.preview_process.terminate()
					self.preview_process.wait(timeout=2.0)
			except (OSError, subprocess.TimeoutExpired) as e:
				self.update_status(f"Error stopping preview: {e}.")
				if self.preview_process.poll() is None:
					self.preview_process.kill()
			finally:
				self.preview_process = None

		self.preview_btn.setEnabled(self._should_enable_preview_button())
		self.stop_preview_btn.setEnabled(False)
		self.update_status("Preview stopped.")
		self.config_display.setText("Experiment stopped. \n\nStart/preview experiment to load configuration.")
		self.current_exp_config = None

	def start_experiment(self):
		if not self.camera_app:
			self.update_status("Error: Camera app not initialized.")
			return

		# Check if the selected experiment requires hardware trigger
		params = self.get_selected_experiment_params()
		if params is None:
			return

		exp_name, experiment_id, mouse_id = params
		
		# Check if this experiment type requires hardware trigger
		exp_types = discover_experiment_types()
		exp_class = exp_types.get(exp_name)
		requires_hw_trigger = getattr(exp_class, 'requires_hardware_trigger', True)
		
		if not self.hardware_trigger_enabled and requires_hw_trigger:
			self.update_status("Error: Must enable hardware trigger mode first.")
			return

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

		started = self.camera_app.get_exp_params(exp_name=exp_name, experiment_id=experiment_id, mouse_id=mouse_id)
		if not started:
			self.update_status('Error: Failed to start experiment subprocess.')
			return
		self._last_plotted_metadata_key = None
		self.camera_app.active_save_filename = None
		self.load_and_display_exp_config()
		self.reset_display_average_on_next_frame = True
		self.reset_fps_on_next_frame = True
		self.exp_btn.setEnabled(False)
		self.stop_exp_btn.setEnabled(True)
		self.trigger_btn.setEnabled(False)
		self.update_status(f"Experiment start requested: {exp_name}. Waiting for worker confirmation...")

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
			exp_name = ""
			if hasattr(self, 'exp_combo') and self.exp_combo is not None:
				exp_name = self.exp_combo.currentText().strip()
			if not exp_name:
				exp_name = (getattr(self.camera_app, 'exp_name', None) or "").strip()

			if not exp_name:
				self.config_display.setText("No experiment selected.\nSelect an experiment from the dropdown.")
				return

			config_file = resolve_experiment_config_file(exp_name)
	        
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
				quantized = self._quantize_gain_value(value)
				if abs(quantized - value) > 1e-9:
					self.gain_spin.blockSignals(True)
					self.gain_spin.setValue(quantized)
					self.gain_spin.blockSignals(False)
				self.camera_app.analog_gain = quantized
				self.camera_app.set_gain(quantized)
				self.load_and_display_camera_config()
				self.update_status(f"Gain set to {quantized}.")
			except Exception as e:
				self.update_status(f'Error setting gain: {e}.')

	def apply_auto_norm(self):
		if self.camera_app:
			self.camera_app.enable_normalize()
			self._update_autonorm_button_text()
			self.update_status(f"Auto norm applied [{self.camera_app.minI:.3f}, {self.camera_app.maxI:.3f}].")

	def disable_norm(self):
		if self.camera_app:
			self.camera_app.normalizeImage = False
			self.camera_app.minI = 0.0
			self.camera_app.maxI = 255.0
			if self.histogram_label is not None:
				self.histogram_label.set_histogram_limits(0.0, 255.0, 0.0, 255.0)
				self.histogram_label.update()
			self._update_autonorm_button_text()
			self.update_status('Normalization disabled.')

	def _update_autonorm_button_text(self):
		if self.camera_app:
			self.autonorm_btn.setText(f"Auto Norm ({self.camera_app.autoI:.2f})")
		else:
			self.autonorm_btn.setText("Auto Norm")

	def toggle_background(self, state):
		if self.camera_app:
			if state == Qt.CheckState.Checked.value:
				enabled = self.camera_app.toggle_background_removal()
				if enabled:
					self.update_status("Background subtraction enabled.")
				else:
					self.background_cb.blockSignals(True)
					self.background_cb.setChecked(False)
					self.background_cb.blockSignals(False)
					self.update_status('Background subtraction unavailable (no valid frame yet).')

			else:
				if self.camera_app.removeBackground:
					self.camera_app.toggle_background_removal()
				self.update_status('Background subtraction disabled.')
 
	def toggle_dfof(self, state):
		if self.camera_app:
			if state == Qt.CheckState.Checked.value:
				enabled = self.camera_app.toggle_dFoF()
				if enabled:
					self.update_status('dFoF enabled.')
				else:
					self.dfof_cb.blockSignals(True)
					self.dfof_cb.setChecked(False)
					self.dfof_cb.blockSignals(False)
					self.update_status('dFoF unavailable (need baseline frames).')
			else:
				self.camera_app.dFoF_open = False
				self.update_status('dFoF disabled.')

	def toggle_histogram(self):
		if not self.camera_app:
			return

		if self.histogram_open:
			self._stop_histogram()
		else:
			self.histogram_open = True
			self.histogram_thread_running = True
			self.histogram_btn.setText("Hide Histogram")
			self.histogram_thread = threading.Thread(target=self.update_histogram, daemon=True)
			self.histogram_thread.start()

	def _stop_histogram(self):
		self.histogram_open = False
		self.histogram_thread_running = False
		self.histogram_btn.setText("Show Histogram")
		self._close_histogram_window()

	def _on_histogram_window_closed(self):
		"""Called from the histogram window's closeEvent — does NOT call .close() again."""
		self.histogram_open = False
		self.histogram_thread_running = False
		self.histogram_btn.setText("Show Histogram")
		self.histogram_window = None
		self.histogram_label = None
		self.histogram_info_label = None

	def _capture_two_consecutive_display_frames(self, timeout_s=5.0):
		if self.camera_app is None:
			return None

		frames = []
		last_frame_data = None
		deadline = time.time() + float(max(0.5, timeout_s))

		while len(frames) < 2 and time.time() < deadline:
			frame_data = self.camera_app.get_frame_for_display()
			if frame_data is None:
				QApplication.processEvents()
				time.sleep(0.005)
				continue

			# Skip duplicates so we collect consecutive displayed frames after the button press.
			if last_frame_data is not None and frame_data == last_frame_data:
				QApplication.processEvents()
				time.sleep(0.005)
				continue

			display_frame = self._frame_data_to_display_frame(frame_data)
			if display_frame is None:
				QApplication.processEvents()
				time.sleep(0.005)
				continue

			frames.append(display_frame.astype(np.float32, copy=True))
			last_frame_data = frame_data
			QApplication.processEvents()
			time.sleep(0.005)

		return frames if len(frames) == 2 else None

	def show_fourier_ring_correlation(self):
		if self.camera_app is None:
			self.update_status("Error: Camera must be running for Fourier Ring Correlation.")
			return
		self.update_status("Starting Fourier Ring Correlation: capturing two consecutive frames...")

		frames = self._capture_two_consecutive_display_frames(timeout_s=5.0)
		if not frames:
			self.update_status("Error: Could not capture two consecutive frames for FRC.")
			return

		img1, img2 = frames

		# Prefer the original in-process numpy/matplotlib plotting path.
		try:
			self._plot_frc_numpy(img1, img2)
			self.update_status('Opened Fourier Ring Correlation plots.')
			return
		except Exception as exc:
			self.update_status(f'Local FRC plot failed ({exc}); launching worker fallback.')

		try:
			with tempfile.NamedTemporaryFile(suffix="_frc_img1.npy", delete=False) as f1:
				img1_path = f1.name
			with tempfile.NamedTemporaryFile(suffix="_frc_img2.npy", delete=False) as f2:
				img2_path = f2.name

			np.save(img1_path, np.asarray(img1, dtype=np.float32))
			np.save(img2_path, np.asarray(img2, dtype=np.float32))

			# Use the same interpreter that launched camstim.
			python_exec = sys.executable

			env = os.environ.copy()
			env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
			log_path = str(Path(tempfile.gettempdir()) / f"camstim_frc_worker_{int(time.time() * 1000)}.log")
			log_file = open(log_path, 'w', encoding='utf-8')
			worker_script = str((REPO_ROOT / 'utils' / 'frc_worker.py').resolve())
			if not Path(worker_script).exists():
				raise RuntimeError(f"FRC worker script not found: {worker_script}")

			proc = subprocess.Popen(
				[python_exec, worker_script, img1_path, img2_path],
				env=env,
				cwd=str(REPO_ROOT),
				stdout=log_file,
				stderr=subprocess.STDOUT,
			)
			log_file.close()

			# Detect immediate startup failures.
			time.sleep(0.05)
			exit_code = proc.poll()
			if exit_code is not None and exit_code != 0:
				raise RuntimeError(f"FRC worker exited immediately with code {exit_code}. See log: {log_path}")

			self.update_status(
				f"Launched Fourier Ring Correlation worker (PID {proc.pid}) using {python_exec}."
			)
			threading.Thread(target=self._track_frc_worker, args=(proc, log_path), daemon=True).start()
		except Exception as e:
			try:
				if 'log_file' in locals() and not log_file.closed:
					log_file.close()
			except Exception:
				pass
			for p in (locals().get('img1_path'), locals().get('img2_path')):
				if p:
					try:
						os.remove(p)
					except OSError:
						pass
			self.update_status(f"Error computing Fourier Ring Correlation: {e}.")

	def toggle_special_pixel_highlight(self, state):
		self.highlight_special_pixels = (state == Qt.CheckState.Checked.value)
		if self.highlight_special_pixels:
			self.update_status('Special pixel highlight enabled (0->blue, 255->red).')
		else:
			self.update_status('Special pixel highlight disabled.')

	def toggle_display_output(self, state):
		enabled = (state == Qt.CheckState.Checked.value)
		if self.camera_app:
			self.camera_app.set_display_output_enabled(enabled)
		if enabled:
			self.update_status('Live display updates enabled.')
		else:
			self.update_status('Live display updates disabled (screen refresh paused).')

	def update_histogram(self):
		hist_width = 512
		hist_height = 400
		hist_update_interval_s = 0.10
		last_draw_time = 0.0
		last_frame_count = -1

		while self.histogram_open and self.histogram_thread_running and self.camera_app:
			now = time.time()
			if now - last_draw_time < hist_update_interval_s:
				time.sleep(0.005)
				continue

			if self.camera_app.frame_count == last_frame_count:
				time.sleep(0.01)
				continue

			frame_data = self.camera_app.get_frame_for_display()
			if frame_data is not None:
				try:
					frame = np.frombuffer(frame_data, dtype=self.camera_app.dtype)
					if hasattr(self.camera_app, 'height') and hasattr(self.camera_app, 'width') and self.camera_app.height and self.camera_app.width:
						frame = frame.reshape((self.camera_app.height, self.camera_app.width))

					if getattr(self.camera_app, 'software_mirror_horizontal', False):
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

					history_frame = np.ascontiguousarray(frame.copy())

					# Use the pre-display frame so histogram values are independent of the color scale.
					history_arr = history_frame.ravel()

					# Keep full 8-bit domain on x-axis; adapt bin count to observed data spread.
					if history_arr.size > 1:
						q25 = np.percentile(history_arr, 25)
						q75 = np.percentile(history_arr, 75)
						iqr = float(q75 - q25)
						n = float(history_arr.size)
						if iqr > 0 and n > 1:
							bin_width_fd = 2.0 * iqr / (n ** (1.0 / 3.0))
							bin_count = int(np.clip(np.ceil(256.0 / max(bin_width_fd, 1e-6)), 16, 256))
						else:
							span = int(history_arr.max()) - int(history_arr.min()) + 1
							bin_count = int(np.clip(span, 16, 256))
					else:
						bin_count = 16

					counts, edges = np.histogram(history_arr, bins=bin_count, range=(0, 256))
					hist = counts.astype(np.float32)
					if hist.max() > 0:
						hist = (hist / hist.max()) * hist_height

					hist_image = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)
					for i in range(bin_count):
						intensity = int(hist[i])
						x1 = int((edges[i] / 256.0) * hist_width)
						x2 = int((edges[i + 1] / 256.0) * hist_width) - 1
						if x2 < x1:
							x2 = x1
						cv2.rectangle(hist_image, (x1, hist_height - intensity), (x2, hist_height), (255, 255, 255), -1)

					line_min = float(getattr(self.camera_app, 'minI', 0.0))
					line_max = float(getattr(self.camera_app, 'maxI', 255.0))
					if line_max <= line_min:
						line_max = line_min + 1.0
					x_scale = (hist_width - 1) / 255.0
					min_x = int(np.clip(line_min, 0, 255) * x_scale)
					max_x = int(np.clip(line_max, 0, 255) * x_scale)
					cv2.line(hist_image, (min_x, 0), (min_x, hist_height), (0, 0, 255), 2)
					cv2.line(hist_image, (max_x, 0), (max_x, hist_height), (255, 0, 0), 2)

					hist_info = (
						f"Frame {self.camera_app.frame_count} | Bins {bin_count} | Range 0-255 | "
						f"Limits min={line_min:.2f}, max={line_max:.2f}"
					)
					self.histogram_frame_ready.emit({'image': hist_image.copy(), 'info': hist_info})
					last_draw_time = now
					last_frame_count = self.camera_app.frame_count
				except Exception as e:
					print(f"\nHistogram error: {e}.")

			time.sleep(0.01)

		self.histogram_close_requested.emit()

	def _on_histogram_frame_ready(self, payload):
		if payload is None:
			return
		hist_image = payload.get('image') if isinstance(payload, dict) else payload
		hist_info = payload.get('info', '') if isinstance(payload, dict) else ''
		if hist_image is None:
			return
		if self.histogram_window is None:
			self.histogram_window = HistogramWindow()
			self.histogram_window.closed.connect(self._on_histogram_window_closed)
			layout = QVBoxLayout()
			self.histogram_label = HistogramLabel()
			self.histogram_label.limitsChanged.connect(self._update_histogram_limits)
			self.histogram_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
			self.histogram_label.setScaledContents(True)
			self.histogram_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
			layout.addWidget(self.histogram_label, 1)
			self.histogram_info_label = QLabel()
			self.histogram_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
			layout.addWidget(self.histogram_info_label)
			self.histogram_window.setLayout(layout)
			self.histogram_window.resize(560, 460)

		if not self.histogram_window.isVisible():
			self.histogram_window.show()

		rgb = cv2.cvtColor(hist_image, cv2.COLOR_BGR2RGB)
		h, w, ch = rgb.shape
		q_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
		pixmap = QPixmap.fromImage(q_img.copy())
		line_min = float(getattr(self.camera_app, 'minI', 0.0))
		line_max = float(getattr(self.camera_app, 'maxI', 255.0))
		if line_max <= line_min:
			line_max = line_min + 1.0
		if self.histogram_label is not None:
			self.histogram_label.set_histogram_limits(line_min, line_max, 0.0, 255.0)
		self.histogram_label.setPixmap(pixmap)
		if self.histogram_info_label is not None:
			self.histogram_info_label.setText(hist_info)

	def _update_histogram_limits(self, min_value, max_value):
		if not self.camera_app:
			return
		self.camera_app.minI = float(min_value)
		self.camera_app.maxI = float(max_value)

	def _close_histogram_window(self):
		if self.histogram_window is not None:
			self.histogram_window.close()
		self.histogram_window = None
		self.histogram_label = None
		self.histogram_info_label = None

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
		line = f"[{timestamp}] {message}"
		self.status_text.append(line)
		self.status_text.verticalScrollBar().setValue(self.status_text.verticalScrollBar().maximum())
		print(line, flush=True)

	def closeEvent(self, event):
		self.histogram_open = False
		self.histogram_thread_running = False
		self._close_histogram_window()
		self.stop_camera()
		super().closeEvent(event)

def main():
	if _setproctitle is not None:
		_setproctitle('camstim')
	app = QApplication(sys.argv)
	window = CameraGUI()
	window.show()
	app.exec()

if __name__ == '__main__':
	main()
