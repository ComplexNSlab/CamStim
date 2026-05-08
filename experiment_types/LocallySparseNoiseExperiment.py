from BaseExperiment import BaseExperiment

import psychopy.visual
import psychopy.event
import psychopy.monitors
import numpy as np
import yaml
import matplotlib.pyplot as plt

from LocallySparseNoise import *
from time import sleep

class LocallySparseNoiseExperiment(BaseExperiment):
    def load_experiment_config(self, ):
        with open (self.exp_parameters_filename, 'r') as file:
            self.exp_parameters = yaml.load(file, Loader=yaml.FullLoader)

        # Name of experiment params
        self.exp_protocol = self.exp_parameters['name']


        # Load n trials and timing lengths
        self.experiment_delay = self.exp_parameters['experiment_delay']
        self.stim_length = self.exp_parameters['stim_length']

        # Load the stim parameters
        self.n_iterations = self.exp_parameters['n_iterations']
        self.n_repeats    = self.exp_parameters['n_repeats']
        self.probe_size   = self.exp_parameters['probe_size']
        self.min_distance = self.exp_parameters['min_distance']
        self.sign         = self.exp_parameters['sign']

        self.lsn = LocallySparseNoise(self.monitor, min_distance=self.min_distance, probe_size=self.probe_size,
                                      sign=self.sign, iteration=self.n_iterations, repeat=self.n_repeats)
        self.generate_stimuli()


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

        self.lsn.generate_randomization()

        if False:
            plt.figure()
            for i in range(len(self.lsn.frames_unique)):
                frame_test = self.lsn.frames_unique[i][1]
                for frame in frame_test:
                    plt.scatter(x=frame[1], y=frame[0])

            plt.show()

        # the sequence of unique frames with randomized probes * n_iterations
        self.experiment_stims = self.lsn.frames_unique

        # the list of indices where each unique frame index is shown n_repeats
        self.experiment_indices = self.lsn.list_of_indices

        max_n_probes = 0
        for i in range(len(self.experiment_stims)):
            tmp = len(self.experiment_stims[i][1])
            if tmp > max_n_probes:
                max_n_probes = tmp

        self.ps_probes = []

        experiment_length = len(self.experiment_indices) * self.stim_length
        print ("Randomization created, total length of experiment {}s, n_probes={}, n_indices={}".format(experiment_length,
                                                                                                         len(self.experiment_stims), 
                                                                                                         len(self.experiment_indices)))
        
        #assert (len(self.experiment_indices) % 2) == 0

        for i in range(max_n_probes):
            probe = psychopy.visual.Rect(win=self.window, pos=(0,0), size=(self.probe_size[1], self.probe_size[0]), 
                                         units='deg', fillColor=self.gray, lineColor=self.gray) # linearized gray
            self.ps_probes.append(probe)




        


    def run_experiment(self, ):
        self.experiment_running = True
        bool_logged_start = False
        bool_logged_end = False

        self.exp_log.log['probe_list'] = self.experiment_stims
        self.exp_log.log['probe_indices'] = self.experiment_indices

        # pre experiment delay
        self.clock.reset()
        self.master_clock.reset()
        
        ## EXPERIMENT STARTS HERE
        # Half of the experiment delay there is no black square and then we draw it, that's when the experiment starts.
        while self.clock.getTime() < self.experiment_delay:
            if self.clock.getTime() >= self.experiment_delay/2:
                if not bool_logged_start:
                    self.exp_log.log_exp_start(self.master_clock.getTime())
                    bool_logged_start = True
                self.photodiode_square.draw()

            self.window.flip()

        self.absolute_total_time += self.experiment_delay



        ## Stimuli start being shown here
        for i, index in enumerate(self.experiment_indices):       
            print("Frame {} out of {}.".format(i+1, len(self.experiment_indices)))
            probes_params = self.experiment_stims[index][1] 

            for j in range(len(probes_params)):
                probe_params = probes_params[j]
                self.ps_probes[j].lineColor = probe_params[2]
                self.ps_probes[j].fillColor = probe_params[2]
                self.ps_probes[j].pos = (probe_params[1], probe_params[0])
                self.ps_probes[j].draw()


            self.photodiode_square.fillColor = self.photodiode_square.lineColor = self.square_color_on
            self.photodiode_square.draw()            
            self.window.flip()
            self.exp_log.log_stimulus(self.master_clock.getTime(), index, len(probes_params), 0)

            # Half the time photodiode square is on 
            self.clock.reset()
            while self.clock.getTime() < self.stim_length/2:
                sleep(0.001)


            for j in range(len(probes_params)):
                probe_params = probes_params[j]
                self.ps_probes[j].lineColor = probe_params[2]
                self.ps_probes[j].fillColor = probe_params[2]
                self.ps_probes[j].pos = (probe_params[1], probe_params[0])
                self.ps_probes[j].draw()


            self.photodiode_square.fillColor = self.photodiode_square.lineColor = self.square_color_off
            self.photodiode_square.draw()
            self.window.flip()

            # second half of the stimulus
            while self.clock.getTime() < self.stim_length:
                sleep(0.001)
                

            self.absolute_total_time += self.stim_length



        ### EXPERIMENT ENDS HERE (FINAL DELAY)
        # Half of the experiment delay there IS black square and then we stop drawing it, that's when the experiment ends.
        self.clock.reset()
        while self.clock.getTime() < self.experiment_delay:
            if self.clock.getTime() < self.experiment_delay/2:
                self.photodiode_square.draw()
            if self.clock.getTime() >= self.experiment_delay/2:
                if not bool_logged_end:
                    self.exp_log.log_exp_end(self.master_clock.getTime(), len(self.experiment_indices))
                    bool_logged_end = True
            self.window.flip()

        self.exp_log.save_log()
        self.experiment_running = False

