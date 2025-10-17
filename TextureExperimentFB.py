from BaseExperiment import BaseExperiment

import psychopy.visual
import psychopy.event
import psychopy.monitors
import numpy as np
import yaml
import pandas as pd

class TextureExperimentFB(BaseExperiment):
    def load_experiment_config(self, ):
        with open (self.exp_parameters_filename, 'r') as file:
            self.exp_parameters = yaml.load(file, Loader=yaml.FullLoader)


        self.exp_protocol = self.exp_parameters['name']

        self.images_filename = self.exp_parameters['images_filename']
        self.vignette_filename = self.exp_parameters['vignette_filename']
        self.images_properties_filename = self.exp_parameters['images_properties_filename']
        self.experiment_delay = self.exp_parameters['experiment_delay']

        self.give_blanks = self.exp_parameters['give_blanks']
        self.n_stims_per_condition = self.exp_parameters['n_stims_per_condition']
        
        self.experiment_delay = self.exp_parameters['experiment_delay']
        self.image_repeat_times = self.exp_parameters['image_repeat_times']
        self.image_on_period = self.exp_parameters['image_on_period']
        self.image_off_period = self.exp_parameters['image_off_period']
        self.inter_trial_delay = self.exp_parameters['inter_trial_delay']


        self.chosen_stim_types = self.exp_parameters['chosen_stim_types']
        self.chosen_families = self.exp_parameters['chosen_families']
        self.chosen_subfamilies = self.exp_parameters['chosen_subfamilies']

        self.image_size = self.exp_parameters['image_size']
        self.image_position = self.exp_parameters['image_position']
        self.image_mask = self.exp_parameters['image_mask']
        self.image_mask_sd = self.exp_parameters['image_mask_sd']


        self.images = None
        self.n_images = None
        self.image_properties = None
        self.load_images()

        self.all_possible_stims = []
        self.experiment_stims = []
        self.n_trials = None
        self.create_randomization()


        # additional log info
        self.exp_log.log['exp_parameters'] = self.exp_protocol
        self.exp_log.log['trial_params_columns'] = self.exp_parameters['trial_params_columns']
        self.exp_log.log['all_possible_stims'] = self.all_possible_stims
        self.exp_log.log['experiment_stims'] = self.experiment_stims
 
        self.image_stim = psychopy.visual.ImageStim(win=self.window, image=None, units="deg", pos=self.image_position,
                                                    size=self.image_size, mask=self.image_mask, maskParams={'sd': self.image_mask_sd})

    def load_images(self):
        print("Loading all images to RAM... ")
        self.images = np.load(self.images_filename).astype(np.float32)
        self.n_images = self.images.shape[0]
        self.images -= 128 # images must be between -1 and 1, where 0 is gray, -1 is black, 1 is white
        self.images /= 128

        print("Loading vignette...")
        self.vignette = np.load(self.vignette_filename)
        print(self.vignette.shape)
        self.images *= self.vignette


        print("Loading all image properties...", flush=False)
        self.image_properties = pd.read_hdf(self.images_properties_filename)

        assert (self.image_properties.shape[0] == self.n_images)

        # Let's check to make sure we have the same number of textures and noise 
        assert (self.image_properties.iloc[np.where(self.image_properties['stim_type'] == 'texture')[0]].count() == 
                self.image_properties.iloc[np.where(self.image_properties['stim_type'] == 'noise')[0]].count()).all()

        print("All Done!")


    def create_randomization(self):
        # First we find out all the possible combinations of stimuli
        for type in self.chosen_stim_types:
            for family in self.chosen_families:
                for subfamily in self.chosen_subfamilies:
                    self.all_possible_stims.append([type, family, subfamily])
                
        # ALL possible stims is an array where either it is a blank or the stim type as such:
        # [stim_type, family, subfamily]
        if self.give_blanks:
            self.all_possible_stims.append('blank')

        # now we create an array containing all possible stims multiplied times the number of stims per condition
        self.experiment_stims = self.all_possible_stims * self.n_stims_per_condition
        self.n_trials = len(self.all_possible_stims) * self.n_stims_per_condition
        np.random.shuffle(self.experiment_stims)

        # Now for each stimulus, if it isn't a blank we will convert it to an image index        
        for i, stim in enumerate(self.experiment_stims):
            if stim != 'blank':
                stim_type = stim[0]
                family = stim[1]
                subfamily = stim[2]
                indices = np.where((self.image_properties['stim_type'] == stim_type) &
                                   (self.image_properties['family'] == family) & 
                                   (self.image_properties['subfamily'] == subfamily))[0]
                
                # now we choose one of the indices at random
                index = np.random.choice(indices, 1)[0]
                
                self.experiment_stims[i] = index






    def run_experiment(self, ):
        print("Experiment starting...")
        self.experiment_running = True
        bool_logged_start = False
        bool_logged_end = False
        # pre experiment delay
        self.clock.reset()
        self.master_clock.reset()
        
        # Half of the experiment delay there is no black square and then we draw it, that's when the experiment starts.
        while self.clock.getTime() < self.experiment_delay:
            if self.clock.getTime() >= self.experiment_delay/2:
                if not bool_logged_start:
                    self.exp_log.log_exp_start(self.master_clock.getTime())
                    bool_logged_start = True
                self.photodiode_square.draw()

            self.window.flip()

        self.absolute_total_time += self.experiment_delay

        for i in range(self.n_trials):
            index = self.experiment_stims[i]
            print("Image trial {} out of {}.".format(i+1, self.n_trials))
            if index != 'blank':
                properties = self.image_properties.iloc[index]
                self.image_stim.image = self.images[index]
            else:
                properties = 'blank'

            total_time = 0
            self.clock.reset()
            self.photodiode_square.fillColor = self.photodiode_square.lineColor = self.square_color_on
            # Log stimulus
            self.exp_log.log_stimulus(self.master_clock.getTime(), i, index, properties)
            for j in range(self.image_repeat_times):
                while self.clock.getTime() < self.image_on_period + total_time:
                    if index != 'blank':
                        self.image_stim.draw()

                    self.photodiode_square.draw()

                    self.window.flip()
                total_time += self.image_on_period
                
                if j==self.image_repeat_times-1:
                    # we skip the last image off period because we go straight into ITI
                    break

                while self.clock.getTime() < self.image_off_period + total_time:
                    self.photodiode_square.draw()

                    self.window.flip()

                total_time += self.image_off_period
 
            self.photodiode_square.fillColor = self.photodiode_square.lineColor = self.square_color_off
            while self.clock.getTime() < self.inter_trial_delay + total_time:
                self.photodiode_square.draw()

                self.window.flip()


        # Half of the experiment delay there IS black square and then we stop drawing it, that's when the experiment ends.
        self.clock.reset()
        while self.clock.getTime() < self.experiment_delay:
            if self.clock.getTime() < self.experiment_delay/2:
                self.photodiode_square.draw()
            if self.clock.getTime() >= self.experiment_delay/2:
                if not bool_logged_end:
                    self.exp_log.log_exp_end(self.master_clock.getTime(), self.n_trials)
                    bool_logged_end = True
            self.window.flip()

        self.exp_log.save_log()
        self.experiment_running = False

