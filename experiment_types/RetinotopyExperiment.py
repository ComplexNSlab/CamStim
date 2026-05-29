from core.BaseExperiment import BaseExperiment

import sys
from pathlib import Path
import psychopy.visual
import psychopy.event
import yaml

# Add WarpedVisualStim directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'external' / 'WarpedVisualStim'))

import WarpedVisualStim.StimulusRoutines as stim
from WarpedVisualStim.MonitorSetup import Monitor, Indicator

class RetinotopyExperiment(BaseExperiment):
    def load_experiment_config(self):
        with open(self.exp_parameters_filename, 'r') as file:
            self.exp_parameters = yaml.load(file, Loader=yaml.FullLoader)

        # Name of experiment params
        self.exp_protocol = self.exp_parameters['name']
        self.experiment_delay = float(self.exp_parameters['experiment_delay'])

        monitor_width_pixels = self.monitor_settings['monitor_width_pixels']
        monitor_height_pixels = self.monitor_settings['monitor_height_pixels']
        monitor_width_cm = self.monitor_settings['monitor_width_cm']
        monitor_height_cm = self.monitor_settings['monitor_height_cm']
        viewing_distance_cm = self.monitor_settings['viewing_distance_cm']
        monitor_gamma = self.monitor_settings['monitor_gamma']
        mon_resolution = (monitor_height_pixels, monitor_width_pixels)

        mon_C2T_cm = monitor_height_cm / 2.0
        mon_C2A_cm = monitor_width_cm / 2.0
        mon_center_coordinates = self.monitor_settings['mon_center_coordinates']
        mon_downsample_rate = self.monitor_settings['mon_downsample_rate']

        self.junmon = Monitor(
            resolution=mon_resolution,
            dis=viewing_distance_cm,
            mon_width_cm=monitor_width_cm,
            mon_height_cm=monitor_height_cm,
            C2T_cm=mon_C2T_cm,
            C2A_cm=mon_C2A_cm,
            center_coordinates=mon_center_coordinates,
            downsample_rate=mon_downsample_rate,
            gamma=monitor_gamma,
            refresh_rate=self.monitor_settings['monitor_refresh_rate'],
        )

        ind_width_cm = self.monitor_settings['ind_width_cm']
        ind_height_cm = self.monitor_settings['ind_height_cm']
        ind_position = self.monitor_settings['ind_position']
        ind_is_sync = self.monitor_settings['ind_is_sync']
        ind_freq = self.monitor_settings['ind_freq']

        self.ind = Indicator(
            self.junmon,
            width_cm=ind_width_cm,
            height_cm=ind_height_cm,
            position=ind_position,
            is_sync=ind_is_sync,
            freq=ind_freq,
        )

        generic = self.exp_parameters['generic_stimulus_parameters']
        ks_cfg = self.exp_parameters['KSstimAllDir']
        self.pregap_dur = generic['pre_gap_dur']
        self.postgap_dur = generic['post_gap_dur']
        self.background = generic['background']
        self.coordinate = generic['coordinate']
        self.block_iterations = int(generic['block_iterations'])

        self.ks_square_size = ks_cfg['ks_square_size']
        self.ks_square_center = ks_cfg['ks_square_center']
        self.ks_flicker_frame = ks_cfg['ks_flicker_frame']
        self.ks_sweep_width = ks_cfg['ks_sweep_width']
        self.ks_step_width = ks_cfg['ks_step_width']
        self.ks_sweep_frame = ks_cfg['ks_sweep_frame']
        self.ks_iteration = ks_cfg['ks_iteration']

        self.ks = stim.KSstimAllDir(
            monitor=self.junmon,
            indicator=self.ind,
            pregap_dur=self.pregap_dur,
            postgap_dur=self.postgap_dur,
            background=self.background,
            coordinate=self.coordinate,
            square_size=self.ks_square_size,
            square_center=self.ks_square_center,
            flicker_frame=self.ks_flicker_frame,
            sweep_width=self.ks_sweep_width,
            step_width=self.ks_step_width,
            sweep_frame=self.ks_sweep_frame,
            iteration=self.ks_iteration,
        )

        print('Created Monitor, Indicator and KSstimAllDir objects.')
        self.generate_stimuli()

        self.exp_log.log['exp_parameters'] = self.exp_protocol
        self.exp_log.log['trial_params_columns'] = self.exp_parameters['trial_params_columns']
        self.exp_log.log['ks_iteration'] = self.ks_iteration
        self.exp_log.log['block_iterations'] = self.block_iterations
        self.exp_log.log['daq_sampling_rate'] = self.daq.sampling_rate

    def load_window(self):
        # true half max luminance
        gray = 255 * (0.5 ** (1 / self.gamma))
        gray -= 128
        gray /= 128

        self.gray = gray
        print('Retinotopy will create a custom pixel-space window in generate_stimuli().')
        
    def create_photodiode_square(self):
        print('Retinotopy uses WarpedVisualStim indicator; skipping BaseExperiment photodiode square.')

    def generate_stimuli(self):
        self.keep_display = True
        self.sequence, self.seq_log = self.ks.generate_movie()
        monitor_width = int(self.monitor_settings['monitor_width_pixels'])
        monitor_height = int(self.monitor_settings['monitor_height_pixels'])

        # Retinotopy sequence is rendered in pixel coordinates.
        self.window = psychopy.visual.Window(monitor=self.monitor, 
                                            size=(monitor_width, monitor_height),
                                            color=self.gray,
                                            colorSpace='rgb',
                                            units='pix',
                                            screen=self.monitor_settings['screen_id'],
                                            allowGUI=False,
                                            fullscr=False,
                                            waitBlanking=True,
                                            useFBO=False)
        
        self.stim = psychopy.visual.ImageStim(
            self.window,
            # KS movie is generated at downsampled resolution; display it
            # stretched to full monitor size for normal retinotopy coverage.
            size=(monitor_width, monitor_height),
            units='pix',
            interpolate=False,
        )

        display_time = float(self.sequence.shape[0]) * self.block_iterations / self.refresh_rate
        print('\nExpected display time: {} seconds.\n'.format(display_time))

    def _update_display_status(self):
        if self.keep_display is None:
            raise LookupError('self.keep_display should start as True')

        if not self.experiment_running:
            self.keep_display = False
            return

        # check keyboard input 'q' or 'escape'
        keyList = psychopy.event.getKeys(['q', 'escape'])
        if len(keyList) > 0:
            self.keep_display = False
            self.experiment_running = False
            self.update_status('Keyboard interrupt detected. Stopping retinotopy display.')

    def stop_experiment(self):
        self.keep_display = False
        self.experiment_running = False
        self.update_status('Retinotopy experiment stop requested.')

    def run_experiment(self):
        self.experiment_running = True
        self.keep_display = True

        self.update_status('Starting Retinotopy experiment.')

        # Pre-experiment delay.
        self.clock.reset()
        self.master_clock.reset()

        while self.keep_display and self.experiment_running and self.clock.getTime() < self.experiment_delay:
            self.window.flip()

        if not self.keep_display or not self.experiment_running:
            self.exp_log.log_exp_end(self.master_clock.getTime(), 0)
            self.exp_log.save_log()
            self.experiment_running = False
            self.update_status('Retinotopy experiment stopped before stimulus presentation.')
            return

        self.exp_log.log_exp_start(self.master_clock.getTime())
        self.absolute_total_time += self.experiment_delay

        n_block = 0
        f = 0
        total_frames = self.sequence.shape[0] * self.block_iterations
        while self.keep_display and self.experiment_running and f < total_frames:
            frame_num = f % self.sequence.shape[0]

            self.stim.setImage(self.sequence[frame_num][::-1])
            self.stim.draw()
            self.window.flip()
            
            if f % self.sequence.shape[0] == 0:
                self.update_trial_progress(n_block + 1, self.block_iterations)
                self.exp_log.log_stimulus(self.master_clock.getTime(), n_block, 0, 0)
                n_block += 1

            self._update_display_status()
            f += 1

        self.exp_log.log_exp_end(self.master_clock.getTime(), total_frames)
        self.exp_log.log['jun_log'] = self.seq_log

        # Post-experiment delay.
        self.clock.reset()
        while self.keep_display and self.experiment_running and self.clock.getTime() < self.experiment_delay:
            self.window.flip()
        
        self.exp_log.save_log()
        self.experiment_running = False
        if self.keep_display:
            self.update_status('Retinotopy experiment completed.')
        else:
            self.update_status('Retinotopy experiment stopped.')

