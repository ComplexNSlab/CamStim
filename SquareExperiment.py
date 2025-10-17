from BaseExperiment import BaseExperiment

import psychopy.visual
import psychopy.event
import psychopy.monitors
import numpy as np
import yaml

class SquareExperiment(BaseExperiment):
    def load_experiment_config(self, ):
        with open (self.exp_parameters_filename, 'r') as file:
            self.exp_parameters = yaml.load(file, Loader=yaml.FullLoader)

        # Name of experiment params
        self.exp_protocol = self.exp_parameters['name']

        # Load params
        self.stim_position = self.exp_parameters['square_position']
        self.stim_size = self.exp_parameters['square_size']
        
       # Since 2020 psychopy argument 'color' was deprecated.
        self.square = psychopy.visual.Rect(win=self.window, pos=self.stim_position, width=self.stim_size[0], height=self.stim_size[1], 
                                                      units='deg', fillColor=0, lineColor=-1)
        # log the experiment parameters
        #self.exp_log.log.create_dataset("exp_parameters", data=self.exp_protocol)
        #self.exp_log.trial_params_columns = self.exp_parameters['trial_params_columns']
        self.exp_log.log['exp_parameters'] = self.exp_protocol




    def run_experiment(self, ):
        self.experiment_running = True
        
        self.clock.reset()
        self.master_clock.reset()
        
        new_color = 0
        self.square.draw()
        while True:
            if new_color < -1:
                break

            self.square.fillColor = new_color
            self.square.draw()
            self.window.flip()

            new_color = float(input("New Color? "))

      
        self.experiment_running = False

