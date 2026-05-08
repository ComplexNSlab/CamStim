from BaseExperiment import BaseExperiment

import psychopy.visual
import psychopy.event
import psychopy.monitors
import numpy as np
import yaml
import pandas as pd

# very similar to TextureExperimentFBVGG however in this one we add the parameterization of the stimulus on time.
class TextureExperimentFBVGGMultiTime(BaseExperiment):
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
        self.image_on_periods = self.exp_parameters['image_on_periods']
        self.image_off_period = self.exp_parameters['image_off_period']
        self.inter_trial_delay = self.exp_parameters['inter_trial_delay']

        # Will get updated per trial
        self.image_on_period = None


        self.chosen_stim_types = self.exp_parameters['chosen_stim_types']
        self.chosen_families = self.exp_parameters['chosen_families']

        self.image_sizes = self.exp_parameters['image_sizes']
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
                                                    size=self.image_sizes[0], mask=self.image_mask, maskParams={'sd': self.image_mask_sd},
                                                    interpolate=True)

    def load_images(self):
        print("Loading all images to RAM... ")
        self.images = np.load(self.images_filename).astype(np.float32)
        self.n_images = self.images.shape[0]
        self.images -= 128 # images must be between -1 and 1, where 0 is gray, -1 is black, 1 is white
        self.images /= 128

        # print("Loading vignette...")
        # self.vignette = np.load(self.vignette_filename)
        # print(self.vignette.shape)
        # self.images *= self.vignette


        print("Loading all image properties...", flush=False)
        self.image_properties = pd.read_hdf(self.images_properties_filename)

        assert (self.image_properties.shape[0] == self.n_images)

        # Let's check to make sure we have the same number of textures and low-order 
        # assert (self.image_properties.iloc[np.where(self.image_properties['stim_type'] == 'texture')[0]].count() == 
                # self.image_properties.iloc[np.where(self.image_properties['stim_type'] == 'low-complexity')[0]].count()).all()
                
        # TODO recalculation of number of blanks
        # chosen families is only for texture stimuli. As LC stimuli have less examples due to being redundant
        #self.n_stim_types_size = (np.where(self.image_properties['stim_type'] == 'texture')[0].shape[0]//len(self.chosen_families))*self.n_stims_per_condition
        #self.n_stim_types_size = (np.where(self.image_properties['stim_type'] == 'low-complexity')[0].shape[0]//len(self.chosen_families))*self.n_stims_per_condition

        print("All Done!")



    def create_randomization(self):
        self.experiment_stims = np.repeat(np.arange(self.n_images), self.n_stims_per_condition)
        # the line below seems to be a rather strange way to repeat the whole array...
        # consider a better method...
        self.experiment_stims = np.asarray(list(self.experiment_stims)*len(self.image_sizes)*len(self.image_on_periods))
        self.stim_sizes = np.repeat(self.image_sizes, self.n_images*self.n_stims_per_condition*len(self.image_on_periods))
        self.stim_on_times = np.repeat(self.image_on_periods, self.n_images*self.n_stims_per_condition*len(self.image_sizes))

        if self.give_blanks:
            raise(NotImplementedError("Blanks must be fixed first"))
            #blank_array_size = np.array([-1]*self.n_stim_types_size)
            #blank_on_times    = np.array([self.image_on_periods[0]]*self.n_stim_types_size)
            #self.experiment_stims = np.concatenate((self.experiment_stims, blank_array_size))
            #self.stim_sizes = np.concatenate((self.stim_sizes, blank_array_size))
            # stim on of blank will be the first element in the image_on_periods array
            #self.stim_on_times = np.concatenate((self.stim_on_times, blank_on_times))

            
        # shuffle the image indices and the stimuli sizes with the same pattern
        shuffler = np.arange(np.shape(self.experiment_stims)[0])
        np.random.shuffle(shuffler)

        self.experiment_stims = self.experiment_stims[shuffler]
        self.stim_sizes = self.stim_sizes[shuffler]
        self.stim_on_times = self.stim_on_times[shuffler]

        self.n_trials = self.experiment_stims.shape[0]

        assert(self.experiment_stims.shape[0] == self.stim_sizes.shape[0])
        assert(self.experiment_stims.shape[0] == self.stim_on_times.shape[0])
        print("Total of trials for experiment: ", self.n_trials)

        self.verify_stimuli_generation()
       
    def verify_stimuli_generation(self):
        # check that number of stimuli match all the sizes, periods and repeat of images
        unique_image_indices = np.unique(self.experiment_stims)
        for unique_index in unique_image_indices:
            assert((self.experiment_stims == unique_index).sum() == self.n_stims_per_condition*len(self.image_sizes)*len(self.image_on_periods))

        unique_stim_sizes = np.unique(self.stim_sizes)
        for unique_size in unique_stim_sizes:
            assert((self.stim_sizes == unique_size).sum() == self.n_images*self.n_stims_per_condition*len(self.image_on_periods))

        unique_stim_times = np.unique(self.stim_on_times)
        for unique_stim_time in unique_stim_times:
            assert((self.stim_on_times == unique_stim_time).sum() == self.n_images*self.n_stims_per_condition*len(self.image_sizes))
        return True


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
            if index != -1:
                properties = self.image_properties.iloc[index]
                self.image_stim.image = self.images[index]
                self.image_stim.size = self.stim_sizes[i]
            else:
                properties = 'blank'
                
            self.image_on_period = self.stim_on_times[i]
                


            total_time = 0
            self.clock.reset()
            self.photodiode_square.fillColor = self.photodiode_square.lineColor = self.square_color_on
            # Log stimulus
            self.exp_log.log_stimulus(self.master_clock.getTime(), i, [index, self.stim_sizes[i], self.image_on_period], properties)
            for j in range(self.image_repeat_times):
                while self.clock.getTime() < self.image_on_period + total_time:
                    if index != -1:
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

