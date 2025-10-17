from BaseExperiment import BaseExperiment

import psychopy.visual
import psychopy.event
import psychopy.monitors
import numpy as np
import yaml
import pandas as pd

class TextureExperimentFBVGG(BaseExperiment):
    def load_experiment_config(self, ):
        with open (self.exp_parameters_filename, 'r') as file:
            self.exp_parameters = yaml.load(file, Loader=yaml.FullLoader)


        self.exp_protocol = self.exp_parameters['name']

        self.images_filename = self.exp_parameters['images_filename']
        self.vignette_filename = self.exp_parameters['vignette_filename']
        self.images_properties_filename = self.exp_parameters['images_properties_filename']
        self.experiment_delay = self.exp_parameters['experiment_delay']

        self.give_blanks = self.exp_parameters['give_blanks']
        self.n_stim_repeats = self.exp_parameters['n_stim_repeats']
        
        self.experiment_delay = self.exp_parameters['experiment_delay']
        self.image_repeat_times = self.exp_parameters['image_repeat_times']
        self.image_on_period = self.exp_parameters['image_on_period']
        self.image_off_period = self.exp_parameters['image_off_period']
        self.inter_trial_delay = self.exp_parameters['inter_trial_delay']


        self.chosen_stim_types = self.exp_parameters['chosen_stim_types']

        self.image_crop_size = self.exp_parameters['image_crop_size']
        self.image_rotations = self.exp_parameters['image_rotations']
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
        self.images = np.load(self.images_filename).astype(np.float32) / 255
        self.n_images = self.images.shape[0]
        
        for i in range(self.n_images):
            self.images[i] = self.linearize_image(self.images[i])

        print("Images linearized!")
        self.images *= 255
        self.images -= 128 # images must be between -1 and 1, where 0 is gray, -1 is black, 1 is white
        self.images /= 128

        if self.image_crop_size != -1:
            center_y = self.images[0].shape[0]//2
            center_x = self.images[0].shape[1]//2
            c_x = self.image_crop_size[0]//2
            c_y = self.image_crop_size[1]//2

            self.images = self.images[:, center_y-c_y:center_y+c_y, center_x-c_x:center_x+c_x]
        print("Shape of image tensor", self.images.shape)

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
                
        # chosen families is only for texture stimuli. As LC stimuli have less examples due to being redundant
        self.n_stim_per_family = (np.where(self.image_properties['stim_type'] == self.chosen_stim_types[0])[0].shape[0]//len(self.image_properties['family'].unique()))
        print("All Done!")



    def create_randomization(self):
        self.experiment_stims = np.repeat(np.arange(self.n_images), self.n_stim_repeats)
        # the line below seems to be a rather strange way to repeat the whole array...
        # consider a better method...

        # multiply experiment stimuli by total number of stim sizes
        self.experiment_stims = np.asarray(list(self.experiment_stims)*len(self.image_sizes))
        self.stim_sizes = np.repeat(self.image_sizes, self.n_images*self.n_stim_repeats*len(self.image_rotations))

        # multiply experiment stimuli by total number of rotations
        self.experiment_stims = np.asarray(list(self.experiment_stims)*len(self.image_rotations))
        self.stim_rotations = np.repeat(self.image_rotations, self.n_images*self.n_stim_repeats*len(self.image_sizes))




        if self.give_blanks:
            blank_array_size = np.array([-1]*self.n_stim_repeats*self.n_stim_per_family)
            self.experiment_stims = np.concatenate((self.experiment_stims, blank_array_size))
            self.stim_sizes = np.concatenate((self.stim_sizes, blank_array_size))
            self.stim_rotations = np.concatenate((self.stim_rotations, blank_array_size))\

        assert(len(self.experiment_stims) == len(self.stim_sizes))
        assert(len(self.experiment_stims) == len(self.stim_rotations))

            
        # shuffle the image indices and the stimuli sizes with the same pattern
        shuffler = np.arange(np.shape(self.experiment_stims)[0])
        np.random.shuffle(shuffler)

        self.experiment_stims = self.experiment_stims[shuffler]
        self.stim_sizes = self.stim_sizes[shuffler]
        self.stim_rotations = self.stim_rotations[shuffler]
        self.n_trials = self.experiment_stims.shape[0]

        assert(self.experiment_stims.shape[0] == self.stim_sizes.shape[0])
        
        print("Total of trials for experiment: ", self.n_trials)
       


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
                self.image_stim.ori = self.stim_rotations[i]
            else:
                properties = 'blank'
                
            if self.debug:
                print(self.image_stim.size, self.image_stim.ori, properties)
            

            total_time = 0
            self.clock.reset()
            self.photodiode_square.fillColor = self.photodiode_square.lineColor = self.square_color_on
            # Log stimulus
            self.exp_log.log_stimulus(self.master_clock.getTime(), i, [index, self.stim_sizes[i], self.stim_rotations[i]], properties)
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

