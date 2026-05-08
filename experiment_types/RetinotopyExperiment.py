from core.BaseExperiment import BaseExperiment

import psychopy.visual
import psychopy.event
import psychopy.monitors
import numpy as np
import yaml

import WarpedVisualStim.StimulusRoutines as stim
from WarpedVisualStim.MonitorSetup import Monitor, Indicator
from WarpedVisualStim.DisplayStimulus import DisplaySequence

class RetinotopyExperiment(BaseExperiment):
    def load_experiment_config(self, ):
        with open (self.exp_parameters_filename, 'r') as file:
            self.exp_parameters = yaml.load(file, Loader=yaml.FullLoader)

        # Name of experiment params
        self.exp_protocol = self.exp_parameters['name']
        self.experiment_delay = self.exp_parameters['experiment_delay']


        # Smelly code but I need my retinotopies quickly :(
        monitor_name = self.monitor_settings['monitor_name']
        monitor_width_pixels = self.monitor_settings['monitor_width_pixels']
        monitor_height_pixels = self.monitor_settings['monitor_height_pixels']
        monitor_width_cm = self.monitor_settings['monitor_width_cm']
        monitor_height_cm = self.monitor_settings['monitor_height_cm']
        viewing_distance_cm = self.monitor_settings['viewing_distance_cm']
        monitor_gamma = self.monitor_settings['monitor_gamma']
        mon_resolution = (monitor_height_pixels, monitor_width_pixels)

        mon_C2T_cm = monitor_height_cm / 2.
        mon_C2A_cm = monitor_width_cm / 2.
        mon_center_coordinates = self.monitor_settings['mon_center_coordinates']
        mon_downsample_rate = self.monitor_settings['mon_downsample_rate']
        
        self.junmon = JunMonitor(resolution=mon_resolution, dis=viewing_distance_cm, mon_width_cm=monitor_width_cm, mon_height_cm=monitor_height_cm, 
                                C2T_cm=mon_C2T_cm, C2A_cm=mon_C2A_cm,
                                center_coordinates=mon_center_coordinates, downsample_rate=mon_downsample_rate, gamma=monitor_gamma,
                                refresh_rate=self.monitor_settings['monitor_refresh_rate'])

        ind_width_cm = self.monitor_settings['ind_width_cm']
        ind_height_cm = self.monitor_settings['ind_height_cm']
        ind_position = self.monitor_settings['ind_position']
        ind_is_sync = self.monitor_settings['ind_is_sync']
        ind_freq    = self.monitor_settings['ind_freq']

        self.ind = Indicator(self.junmon, width_cm=ind_width_cm, height_cm=ind_height_cm,
                position=ind_position, is_sync=ind_is_sync, freq=ind_freq)


        self.pregap_dur = self.exp_parameters['generic_stimulus_parameters']['pre_gap_dur']
        self.postgap_dur = self.exp_parameters['generic_stimulus_parameters']['post_gap_dur']
        self.background = self.exp_parameters['generic_stimulus_parameters']['background']
        self.coordinate = self.exp_parameters['generic_stimulus_parameters']['coordinate']
        self.block_iterations = self.exp_parameters['generic_stimulus_parameters']['block_iterations']

        self.ks_square_size = self.exp_parameters['KSstimAllDir']['ks_square_size']
        self.ks_square_center = self.exp_parameters['KSstimAllDir']['ks_square_center']
        self.ks_flicker_frame = self.exp_parameters['KSstimAllDir']['ks_flicker_frame']
        self.ks_sweep_width = self.exp_parameters['KSstimAllDir']['ks_sweep_width']
        self.ks_step_width = self.exp_parameters['KSstimAllDir']['ks_step_width']
        self.ks_sweep_frame = self.exp_parameters['KSstimAllDir']['ks_sweep_frame']
        self.ks_iteration = self.exp_parameters['KSstimAllDir']['ks_iteration']


        self.ks = stim.KSstimAllDir(monitor=self.junmon, indicator=self.ind, pregap_dur=self.pregap_dur, postgap_dur=self.postgap_dur,
                       background=self.background, coordinate=self.coordinate, square_size=self.ks_square_size,
                       square_center=self.ks_square_center, flicker_frame=self.ks_flicker_frame,
                       sweep_width=self.ks_sweep_width, step_width=self.ks_step_width, sweep_frame=self.ks_sweep_frame,
                       iteration=self.ks_iteration)

        print("Created JunMon, Indicator and KSstim obj.")
        self.generate_stimuli()
        

        # log the experiment parameters
        #self.exp_log.log.create_dataset("exp_parameters", data=self.exp_protocol)
        #self.exp_log.trial_params_columns = self.exp_parameters['trial_params_columns']
        self.exp_log.log['exp_parameters'] = self.exp_protocol
        self.exp_log.log['trial_params_columns'] = self.exp_parameters['trial_params_columns']
        self.exp_log.log['ks_iteration'] = self.ks_iteration
        self.exp_log.log['block_iterations'] = self.block_iterations
    


        # Save DAQ params into the log
        # TODO improve, I feel like calls to create_dset should be within DAQ class
        #self.exp_log.log.create_dataset("daq_sampling_rate", data=self.daq.sampling_rate)
        self.exp_log.log['daq_sampling_rate'] = self.daq.sampling_rate
        
    def load_window(self):
        # true half max luminance
        gray = 255*(0.5 ** (1/self.gamma))
        gray -= 128
        gray /= 128

        self.gray = gray
        print("No window will be made for OG simple-vs")
        
    def create_photodiode_square(self):
        print("No OG simple-vs photodiode square because Allen Institute already has one")

    def generate_stimuli(self):
        self.keep_display = True
        self.sequence, self.seq_log = self.ks.generate_movie()
        resolution = self.seq_log['monitor']['resolution'][::-1]
        # Override the old window obj due to wrong units
        self.window = psychopy.visual.Window(monitor=self.monitor, 
                                            size=(self.monitor_settings['monitor_width_pixels'],
                                                  self.monitor_settings['monitor_height_pixels']),
                                            color=self.gray,
                                            colorSpace='rgb',
                                            units='pix',
                                            screen=self.monitor_settings['screen_id'],
                                            allowGUI=True,
                                            fullscr=False,
                                            waitBlanking=False,
                                            useFBO=False)
        
        self.stim = psychopy.visual.ImageStim(self.window, size=resolution, interpolate=False)


        display_time = (float(self.sequence.shape[0]) *
                            self.block_iterations / self.refresh_rate)
        print('\nExpected display time: {} seconds.\n'.format(display_time))



    def _update_display_status(self):
        if self.keep_display is None:
            raise LookupError('self.keep_display should start as True')

        # check keyboard input 'q' or 'escape'
        keyList = psychopy.event.getKeys(['q', 'escape'])
        if len(keyList) > 0:
            self.keep_display = False
            print("Keyboard interrupting signal detected. Stop displaying. \n")



    def run_experiment(self, ):
        
        self.experiment_running = True
        bool_logged_start = False
        bool_logged_end = False
        # pre experiment delay
        self.clock.reset()
        self.master_clock.reset()

        # Experiment starts after experiment_delay time
        while self.clock.getTime() < self.experiment_delay:
            self.window.flip()

        self.exp_log.log_exp_start(self.master_clock.getTime())
        self.absolute_total_time += self.experiment_delay


        n_block = 0 
        f = 0
        while self.keep_display and f < self.sequence.shape[0]*self.block_iterations:
            frame_num = f % self.sequence.shape[0]

            self.stim.setImage(self.sequence[frame_num][::-1])

            self.stim.draw()

            self.window.flip()
            
            if f % self.sequence.shape[0] == 0:
                print(f, self.sequence.shape[0])
                self.exp_log.log_stimulus(self.master_clock.getTime(), n_block, 0, 0)
                n_block += 1

            self._update_display_status()

            f += 1


        self.exp_log.log_exp_end(self.master_clock.getTime(), self.sequence.shape[0]*self.block_iterations)
        self.exp_log.log['jun_log'] = self.seq_log
        # Half of the experiment delay there IS black square and then we stop drawing it, that's when the experiment ends.
        self.clock.reset()
        while self.clock.getTime() < self.experiment_delay:
            self.window.flip()
        
        self.exp_log.save_log()
        self.experiment_running = False

