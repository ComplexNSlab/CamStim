from abc import ABC, abstractmethod
from core.ExperimentLogger import ExperimentLogger

import psychopy.visual
import psychopy.event
import psychopy.monitors
import yaml
from sys import platform
import os
from pathlib import Path
from time import sleep
import numpy as np
from psychopy.visual.windowwarp import Warper

class BaseExperiment(ABC):
    def __init__(self, experiment_id, mouse_id, daq, monitor_config_filename, save_settings_config_filename, exp_config_filename, debug):
        self.experiment_id = experiment_id
        self.mouse_id = mouse_id
        self.debug = debug

        self.status_callback = None
        self.trial_callback = None

        self.acquisition_running = False
        self.experiment_running = False

        # the NI daq logging class 
        self.daq = daq
        
        self.monitor = None
        self.monitor_settings = None
        self.window = None

        self.exp_parameters = None
        self.exp_parameters_filename = self.resolve_config_path(exp_config_filename)
        
        self.clock = psychopy.core.Clock()
        self.master_clock = psychopy.core.Clock()
        self.absolute_total_time = 0

        self.load_monitor(monitor_config_filename)
        self.load_window()

        self.save_dir = None
        self.experiment_log_filename = None
        self.ni_log_filename = None
        self.data_log_dir = None
        self.create_save_directories(save_settings_config_filename)

        self.experiment_settings_filenames = [
            self.resolve_config_path(monitor_config_filename),
            self.resolve_config_path(save_settings_config_filename),
            self.exp_parameters_filename,
        ]

        # experiment trial params logger
        self.exp_log = ExperimentLogger(self.experiment_log_filename, self.experiment_id, self.mouse_id, self.data_log_dir, self.experiment_settings_filenames) 
        self.daq.ni_log_filename = self.ni_log_filename

        self.create_photodiode_square()

        # Common exp log    
        self.exp_log.log['daq_sampling_rate'] = self.daq.sampling_rate

    def set_status_callback(self, callback):
        self.status_callback = callback

    def set_trial_callback(self, callback):
        self.trial_callback = callback

    def update_status(self, message):
        if self.status_callback:
            self.status_callback(message)
        else:
            print(message)

    def update_trial_progress(self, current, total):
        message = f"Trial {current} out of {total}."
        if self.trial_callback:
            self.trial_callback(current, total, message)
        elif self.status_callback:
            self.status_callback(message)
        else:
            print(message)

    def __del__(self):
        self.window.close()

    @staticmethod
    def resolve_config_path(config_filename):
        config_path = Path(config_filename)
        if config_path.is_absolute():
            return str(config_path)

        repo_root = Path(__file__).resolve().parent.parent
        # Accept either "monitor_config.yaml" or "config_files/monitor_config.yaml".
        candidate_paths = [
            repo_root / 'config_files' / config_path,
            repo_root / config_path,
        ]
        for candidate in candidate_paths:
            if candidate.is_file():
                return str(candidate)

        # Backward-compatible fallback for callers that pass core-relative paths.
        return str(Path(__file__).resolve().parent / config_path)
   
    def load_monitor(self, monitor_config_filename):
        monitor_path = self.resolve_config_path(monitor_config_filename)
        with open(monitor_path, 'r') as file:
            self.monitor_settings = yaml.load(file, Loader=yaml.FullLoader)
        
        monitor_name = self.monitor_settings['monitor_name']
        monitor_width_pixels = self.monitor_settings['monitor_width_pixels']
        monitor_height_pixels = self.monitor_settings['monitor_height_pixels']
        monitor_width_cm = self.monitor_settings['monitor_width_cm']
        viewing_distance_cm = self.monitor_settings['viewing_distance_cm']
        monitor_gamma = self.monitor_settings['monitor_gamma']
        self.refresh_rate = self.monitor_settings['monitor_refresh_rate']
        
        self.gamma = self.monitor_settings['monitor_gamma']
 

        self.monitor = psychopy.monitors.Monitor(monitor_name, width=monitor_width_cm, distance=viewing_distance_cm, gamma=1)
        self.monitor.setSizePix((monitor_width_pixels, monitor_height_pixels))


        #TODO FB additional fields that are sometimes used (i.e LocallySparseNoise)
        # AWFUL idea but not sure what is the right way to do this, should I write a FBMonitor class with the additional fields...?
        monitor_height_cm = monitor_width_cm*(monitor_height_pixels/monitor_width_pixels)
        
        # Following coordinate system assumes (0, 0) is the center of the monitor
        self.monitor.azi_min_coord = -np.arctan((monitor_width_cm/2)/viewing_distance_cm)*180/np.pi
        self.monitor.azi_max_coord = np.arctan((monitor_width_cm/2)/viewing_distance_cm)*180/np.pi
        self.monitor.alt_min_coord = -np.arctan((monitor_height_cm/2)/viewing_distance_cm)*180/np.pi
        self.monitor.alt_max_coord = np.arctan((monitor_height_cm/2)/viewing_distance_cm)*180/np.pi
        
        # TODO  check if monitor needs to get saved in win
        #self.monitor.save()
        
    def linearize_image(self, im):
        #return (self.a + self.k*im)**self.gamma
        ## im must be between 0 and 1
        return im**(1/self.gamma)

    def load_window(self):
        #monitor_gamma_lut = np.load(self.monitor_settings['monitor_gamma_lut'])
        
        # true half max luminance
        gray = 255*(0.5 ** (1/self.gamma))
        gray -= 128
        gray /= 128

        self.gray = gray
        
        print("linearized gray applied!")

        self.window = psychopy.visual.Window(monitor=self.monitor, 
                                            size=(self.monitor_settings['monitor_width_pixels'],
                                                  self.monitor_settings['monitor_height_pixels']),
                                            color=gray,
                                            colorSpace='rgb',
                                            units='deg',
                                            screen=self.monitor_settings['screen_id'],
                                            allowGUI=False, #False for real experiment
                                            fullscr=False, #True " "
                                            waitBlanking=True, #True " "
                                            useFBO=True) #True " "


        if self.monitor_settings['use_spherical_warp']:
            self.warper = Warper(self.window,
                    warp='spherical',
                    warpfile = "",
                        warpGridsize = 128,
                    eyepoint = (0.5, 0.5),
                    flipHorizontal = False,
                    flipVertical = False)

            print("Using spherical warping!")

        
        #self.window.gammaRamp = monitor_gamma_lut


    def create_photodiode_square(self):
        # Load the settings of the square that goes on the photodiode
        self.square_size = self.monitor_settings['square_size']
        self.square_position = self.monitor_settings['square_position']
        self.square_color_off = self.monitor_settings['square_color_off']
        self.square_color_on = self.monitor_settings['square_color_on']

        # Since 2020 psychopy argument 'color' was deprecated.
        self.photodiode_square = psychopy.visual.Rect(win=self.window, pos=self.square_position, width=self.square_size[0], height=self.square_size[1], 
                                                      units='pix', fillColor=self.square_color_off, lineColor=self.square_color_off)


    def create_save_directories(self, save_settings_config_filename):
        save_settings_path = self.resolve_config_path(save_settings_config_filename)
        with open(save_settings_path, 'r') as file:
            save_settings = yaml.load(file, Loader=yaml.FullLoader)

        save_root = save_settings.get('SAVE_DIR')
        if not save_root:
            raise Exception("SAVE_DIR missing in {}".format(save_settings_path))

        self.save_dir = os.path.join(save_root, self.mouse_id, self.experiment_id)

        self.data_log_dir = self.save_dir

        # set the log filenames for the NI log and the exp log
        self.experiment_log_filename = os.path.join(self.data_log_dir, "{}_{}".format(self.mouse_id, self.experiment_id))
        self.ni_log_filename = os.path.join(self.data_log_dir, "{}_ni_log.npy".format(self.experiment_id))


        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        else:
            if not self.debug:
                raise Exception("Experiment ID: {} already exists make new ID...".format(self.experiment_id))



    def start_data_acquisition(self,):
        if self.daq is None:
            raise Exception("Please set the daq object, it has not been set.")

        if platform == "win32" and not self.debug:
            self.acquisition_running = True
            self.daq.start_everything()


    def stop_data_acquisition(self,):
        if platform == "win32" and not self.debug:
            self.daq.stop_everything()
            self.acquisition_running = False


    @abstractmethod
    def load_experiment_config(self, ):
        pass

    @abstractmethod
    def run_experiment(self,):
        pass

