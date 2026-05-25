import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import warnings



class ROIDrawer:
    def __init__(self):
        self.top_left_pt = (-1, -1)
        self.bottom_right_pt = (-1, -1)
        self.drawing = False
        self.current_mouse_position = None
        self.active_roi = False

    def handle_mouse_events(self, event, x, y, flags, param):
        self.current_mouse_position = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.top_left_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.bottom_right_pt = (x, y)
            self.active_roi = True

    def draw_rectangle(self, frame):
        if self.drawing:
            cv2.rectangle(frame, self.top_left_pt, self.current_mouse_position, (0, 0, 255), 2)
        elif self.active_roi:
            cv2.rectangle(frame, self.top_left_pt, self.bottom_right_pt, (0, 0, 255), 2)
        return frame


class ROIPlotter:
    def __init__(self):
        self.times = []
        self.intensities = []
        self.initialized = False
        warnings.filterwarnings("ignore")

    def initialize_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(10,6))
        plt.ion()
        plt.show()
        self.t_start = time.time()

    def deinitialize_plot(self):
        if self.initialized:
            plt.close()
            self.times = []
            self.intensities = []
            self.initialized = False

    def update_plot(self, average_intensity):
        if not self.initialized:
            self.initialize_plot()
            self.initialized = True
        # if top_left_pt != (-1, -1) and bottom_right_pt != (-1, -1):
            #roi = frame[top_left_pt[1]:bottom_right_pt[1], top_left_pt[0]:bottom_right_pt[0]]
            #average_intensity = np.mean(roi)
        self.intensities.append(average_intensity)
        self.times.append(time.time()-self.t_start)  # Assuming each update is a new time point

        # Limit the number of points to the last 100
        if len(self.times) > 300:
            self.times = self.times[-300:]
            self.intensities = self.intensities[-300:]

        self.ax.clear()
        self.ax.plot(self.times, self.intensities)
        self.ax.set_title("Average Intensity Over Time")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Average Intensity (a.u.)")
        plt.draw()
