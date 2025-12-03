import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
							QWidget, QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
							QGroupBox, QTextEdit, QCheckBox, QComboBox)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
import time
from simple_cam_mx import App, load_camera_config
import threading
import mvsdk
import cv2

class CameraGUI(QMainWindow):
	def __init__(self):
		super().__init__()
		self.config = load_camera_config('cam_config.yml')
		self.camera_app = None
		self.init_ui()
		self.setup_timer()

	def init_ui(self):
		self.setWindowTitle('Camera GUI')
		self.setGeometry(100, 100, 1200, 800)

		central_widget = QWidget()
		self.setCentralWidget(central_widget)

		main_layout = QHBoxLayout()
		central_widget.setLayout(main_layout)

		left_panel = self.create_display_panel()
		main_layout.addWidget(left_panel, 1)

		right_panel = self.create_control_panel()
		main_layout.addWidget(right_panel, 2)

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

		self.stats_label = QLabel("FPS: 0 | Frames: 0 | Saved: 0 | Queue: 0")
		status_layout.addWidget(self.stats_label)

		status_group.setLayout(status_layout)
		layout.addWidget(status_group)

		layout.addStretch()

		return panel

	def setup_timer(self):
		self.timer = QTimer()
		self.timer.timeout.connect(self.update_display)
		self.stats_timer = QTimer()
		self.stats_timer.timeout.connect(self.update_stats)

	def start_camera(self):
		try:
			self.camera_app = App(self.config, gui_mode=True)

			self.camera_app.save_thread.start()
			self.camera_app.display_thread.start()

			self.camera_thread = threading.Thread(target=self.camera_app.main)
			self.camera_thread.daemon = True
			self.camera_thread.start()

			self.timer.start(30)
			self.stats_timer.start(100)

			self.start_btn.setEnabled(False)
			self.stop_btn.setEnabled(True)
			self.trigger_btn.setEnabled(True)
			self.exp_btn.setEnabled(False)

			self.update_status("Camera started successfully in continuous mode.")
		except Exception as e:
			self.update_status(f"Error starting camera: {str(e)}.")

	def stop_camera(self):
		if self.camera_app:
			self.camera_app.quit = True
			self.timer.stop()
			self.stats_timer.stop()
			self.camera_app.cleanup_udp()

			if hasattr(self.camera_app, 'save_thread'):
				self.camera_app.save_thread.join(timeout=2.0)
			if hasattr(self.camera_app, 'display_thread'):
				self.camera_app.display_thread.join(timeout=2.0)
			if hasattr(self, 'camera_thread'):
				self.camera_thread.join(timeout=2.0)

		self.start_btn.setEnabled(True)
		self.stop_btn.setEnabled(False)
		self.exp_btn.setEnabled(False)
		self.video_label.setText("Camera Stopped.")
		self.update_status("Camera Stopped.")

	def update_display(self):
		if self.camera_app and not self.camera_app.display_queue.empty():
			try:
				frame_data = self.camera_app.display_queue.get_nowait()
				frame = np.frombuffer(frame_data, dtype=self.camera_app.dtype)
				if hasattr(self.camera_app, 'height') and hasattr(self.camera_app, 'width'):
					frame = frame.reshape((self.camera_app.height, self.camera_app.width))
					frame = cv2.flip(frame, 1)
				
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
				frame = self.camera_app.circular_buffer.mean(axis=0)

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
			self.video_label.setPixmap(pixmap.scaled(self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio))
		except Exception as e:
			print(f"Display frame error: {e}.")

	def update_stats(self):
		if self.camera_app:
			try:
				current_time = time.time()
				elapsed_time = current_time - self.camera_app.t_start if self.camera_app.t_start else 0
				average_fps = self.camera_app.frame_count / elapsed_time if elapsed_time > 0 else 0

				stats_text = f"FPS: {average_fps: .1f} | Frames: {self.camera_app.frame_count} | Saved: {self.camera_app.frames_written} | Queue: {self.camera_app.display_queue.qsize()}"
				self.stats_label.setText(stats_text)
			except:
				pass

	def toggle_trigger_mode(self):
		if self.camera_app and self.camera_app.hCamera:
			try:
				if not self.camera_app.saving:
					print("\n Switching to hardware trigger mode.")
					mvsdk.CameraSetTriggerMode(self.camera_app.hCamera, 2)
					self.camera_app.saving = True
					
					self.trigger_btn.setText("Disable Hardware Trigger")
					self.exp_btn.setEnabled(True)
					self.update_status("Hardware trigger mode enabled - ready for experiment.")
				else:
					self.camera_app.saving = False
					print("\n Switching to continuous mode.")
					mvsdk.CameraSetTriggerMode(self.camera_app.hCamera, 0)

					self.trigger_btn.setText("Enable Hardware Trigger")
					self.exp_btn.setEnabled(False)
					self.update_status("Continuous mode enabled.")
			except Exception as e:
				self.update_status(f"Error toggling trigger mode: {e}.")

	def start_experiment(self):
		if self.camera_app and self.camera_app.saving:
			self.camera_app.get_exp_params()
			self.exp_btn.setEnabled(False)
			self.stop_exp_btn.setEnabled(True)
			self.trigger_btn.setEnabled(False)
			self.update_status("Experiment started.")
		else:
			self.update_status("Error: Must enable hardware trigger mode first.")

	def stop_experiment(self):
		if self.camera_app:
			self.camera_app.stop_stim()
			self.exp_btn.setEnabled(True)
			self.stop_exp_btn.setEnabled(False)
			self.trigger_btn.setEnabled(True)
			self.update_status("Experiment stopped.")

	def update_exposure(self, value):
		if self.camera_app and self.camera_app.hCamera:
			try:
				self.camera_app.exposure = value
				mvsdk.CameraSetExposureTime(self.camera_app.hCamera, value * 1000)
				self.update_status(f"Exposure set to {value}ms.")
			except Exception as e:
				self.update_status(f'Error setting exposure: {e}.')

	def update_gain(self, value):
		if self.camera_app and self.camera_app.hCamera:
			try:
				self.camera_app.analog_gain = value
				mvsdk.CameraSetAnalogGain(self.camera_app.hCamera, value)
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

### need to fix this 
	def toggle_speckle(self, state):
		if self.camera_app:
			enabled = (state == Qt.CheckState.Checked.value)

			if enabled and not hasattr(self.camera_app, 'circular_buffer'):
				if self.camera_app.live_speck:
					self.camera_app.setup_live_speckle_variables()

			self.camera_app.enable_live_speckle = enabled

			if enabled:
				self.update_status('Live speckle imaging enabled.')
			else:
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
		if self.camera_app:
			if state == Qt.CheckState.Checked.value:
				self.camera_app.toggle_histogram()
				self.update_status('Histogram enabled.')
			else:
				if self.camera_app.histogram_open:
					self.camera_app.toggle_histogram()
				self.update_status('Histogram disabled.')

	def update_trigger_mode(self, index):
		if self.camera_app and self.camera_app.hCamera:
			try:
				mvsdk.CameraSetTriggerMode(self.camera_app.hCamera, index)
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

