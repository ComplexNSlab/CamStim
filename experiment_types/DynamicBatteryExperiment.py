from BaseExperiment import BaseExperiment

import psychopy.visual
import psychopy.event
import psychopy.monitors
import numpy as np
import yaml

class DynamicBatteryExperiment(BaseExperiment):
    def load_experiment_config(self, ):
        with open (self.exp_parameters_filename, 'r') as file:
            self.exp_parameters = yaml.load(file, Loader=yaml.FullLoader)

        # Name of experiment params
        self.exp_protocol = self.exp_parameters['name']


        # Load n trials and timing lengths
        self.n_trials = self.exp_parameters['n_trials']
        self.experiment_delay = self.exp_parameters['experiment_delay']
        self.stim_length = self.exp_parameters['stim_length']
        self.inter_trial_delay = self.exp_parameters['inter_trial_delay']

        # Load the stim parameters
        self.grating_sfs = self.exp_parameters['grating_sfs']
        self.grating_position = self.exp_parameters['grating_position']
        self.grating_orientations = self.exp_parameters['grating_orientations']
        self.give_blanks = self.exp_parameters['give_blanks']
        #self.grating_phase_temporal_frequency = self.exp_parameters['grating_phase_temporal_frequency']
        self.grating_sizes = self.exp_parameters['grating_sizes']
        self.grating_mask = self.exp_parameters['grating_mask']

        # generate the stimuli, all orientations (+ blanks) will be shown same number of times but randomized
        self.experiment_stims = []
        self.generate_stimuli()

        # ps stands for psychopy... 
        self.ps_grating = psychopy.visual.GratingStim(win=self.window, units="deg", pos=self.grating_position, sf=self.grating_sfs[0],
                                                       size=[self.grating_sizes[0], self.grating_sizes[1]], mask=self.grating_mask)

        # log the experiment parameters
        #self.exp_log.log.create_dataset("exp_parameters", data=self.exp_protocol)
        #self.exp_log.trial_params_columns = self.exp_parameters['trial_params_columns']
        self.exp_log.log['exp_parameters'] = self.exp_protocol
        self.exp_log.log['trial_params_columns'] = self.exp_parameters['trial_params_columns']


        # Save DAQ params into the log
        # TODO improve, I feel like calls to create_dset should be within DAQ class
        #self.exp_log.log.create_dataset("daq_sampling_rate", data=self.daq.sampling_rate)
        self.exp_log.log['daq_sampling_rate'] = self.daq.sampling_rate

    def generate_stimuli(self):
        all_possible_stims = []
        for ori in self.grating_orientations:
            for sf in self.grating_sfs:
                for size in self.grating_sizes:
                    all_possible_stims.append([ori, sf, size])

        if self.give_blanks:
            all_possible_stims.append('blank')
       

        n_stims_per_condition = (self.n_trials//len(all_possible_stims))
        if n_stims_per_condition * len(all_possible_stims) != self.n_trials:
            print(len(all_possible_stims))
            raise Exception("Please make the number of trials divisible by total possible stims")

        self.experiment_stims = all_possible_stims * (self.n_trials//len(all_possible_stims))

        np.random.shuffle(self.experiment_stims)


        


    def run_experiment(self, ):
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

        for trial in range(self.n_trials):
            print("Trial {} out of {}.".format(trial+1, self.n_trials))
            current_orientation = self.experiment_stims[trial][0]
            current_sf = self.experiment_stims[trial][1]
            current_size = self.experiment_stims[trial][2]
            current_phase = np.round(np.random.random(), 2)

            #current_phase = np.random.randint(self.grating_phases_range[0], self.grating_phases_range[1])

            if self.experiment_stims[trial] != 'blank':
                self.ps_grating.ori = current_orientation
                self.ps_grating.sf = current_sf
                self.ps_grating.size = current_size
                self.ps_grating.phase = current_phase#np.round(np.random.random(), 2)
                
            


            total_time = 0
            self.clock.reset()

            self.photodiode_square.fillColor = self.photodiode_square.lineColor = self.square_color_on
            # Log stimulus
            # TODO figure out how to dump all the PS grating information easily....
            self.exp_log.log_stimulus(self.master_clock.getTime(), trial, [current_orientation, current_sf, current_size, current_phase], 0)
            while self.clock.getTime() < self.stim_length:

                # temporal frequency change
                # if self.clock.getTime() < self.stim_length/2:
                #     self.ps_grating.contrast = 1
                # else:
                #     self.ps_grating.contrast = -1
                #self.ps_grating.phase = np.mod(self.clock.getTime(), 1)

                #self.ps_grating.contrast = 1 - (1/(self.stim_length/2))*np.mod(self.clock.getTime(), self.stim_length)

                if self.experiment_stims[trial] != 'blank':
                    self.ps_grating.draw()
                self.photodiode_square.draw()

                self.window.flip()
            self.photodiode_square.fillColor = self.photodiode_square.lineColor = self.square_color_off

            total_time += self.stim_length


            while self.clock.getTime() < total_time+self.inter_trial_delay:
                self.photodiode_square.draw()
                self.window.flip()

            total_time += self.inter_trial_delay

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

