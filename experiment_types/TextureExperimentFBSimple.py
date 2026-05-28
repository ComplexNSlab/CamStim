from pathlib import Path

from core.BaseExperiment import BaseExperiment

import cv2
import numpy as np
import psychopy.event
import psychopy.monitors
import psychopy.visual
import yaml


class TextureExperimentFBSimple(BaseExperiment):
    def load_experiment_config(self):
        with open(self.exp_parameters_filename, 'r') as file:
            self.exp_parameters = yaml.load(file, Loader=yaml.FullLoader)

        self.exp_protocol = self.exp_parameters['name']

        self.images_folder = self.exp_parameters['images_folder']
        self.vignette_filename = self.exp_parameters.get('vignette_filename', '')
        self.experiment_delay = self.exp_parameters['experiment_delay']

        self.give_blanks = self.exp_parameters['give_blanks']
        self.n_stims_per_condition = self.exp_parameters['n_stims_per_condition']

        self.image_on_period = self.exp_parameters['image_on_period']
        self.inter_trial_delay = self.exp_parameters['inter_trial_delay']

        self.image_size = self.exp_parameters['image_size']
        self.image_position = self.exp_parameters['image_position']
        self.image_mask = self.exp_parameters['image_mask']
        self.image_mask_sd = self.exp_parameters['image_mask_sd']

        self.images = None
        self.image_paths = []
        self.n_images = None
        self.load_images()

        self.experiment_stims = []
        self.n_trials = None
        self.create_randomization()

        self.exp_log.log['exp_parameters'] = self.exp_protocol
        self.exp_log.log['trial_params_columns'] = self.exp_parameters['trial_params_columns']
        self.exp_log.log['image_paths'] = [str(path) for path in self.image_paths]
        self.exp_log.log['experiment_stims'] = self.experiment_stims

        self.image_stim = psychopy.visual.ImageStim(
            win=self.window,
            image=None,
            units='deg',
            pos=self.image_position,
            size=self.image_size,
            mask=self.image_mask,
            maskParams={'sd': self.image_mask_sd},
        )

    def _normalize_image(self, image):
        if image.ndim == 3:
            image = image[:, :, 0]

        orig_dtype = image.dtype
        image = image.astype(np.float32)

        if np.issubdtype(orig_dtype, np.integer):
            info = np.iinfo(orig_dtype)
            image /= float(info.max)
        else:
            max_val = float(np.max(np.abs(image)))
            if max_val > 0:
                image /= max_val

        return (image * 2.0) - 1.0

    def load_images(self):
        print('Loading all TIFF images to RAM...')
        folder = Path(self.images_folder)
        if not folder.is_dir():
            raise FileNotFoundError(f'Image folder does not exist: {folder}')

        self.image_paths = sorted(
            [
                path
                for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in ('.tif', '.tiff')
            ]
        )

        if not self.image_paths:
            raise ValueError(f'No TIFF images found in {folder}')

        loaded_images = []
        expected_shape = None
        for path in self.image_paths:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f'Unable to read image: {path}')

            image = self._normalize_image(image)
            if expected_shape is None:
                expected_shape = image.shape
            elif image.shape != expected_shape:
                raise ValueError(
                    f'Image shape mismatch for {path}: expected {expected_shape}, got {image.shape}'
                )
            loaded_images.append(image)

        self.images = np.stack(loaded_images, axis=0).astype(np.float32)
        self.n_images = int(self.images.shape[0])

        # print('Loading vignette...')
        # self.vignette = np.load(self.vignette_filename)
        # self.images *= self.vignette

        print(f'Loaded {self.n_images} images.')

    def create_randomization(self):
        repeated_images = np.repeat(np.arange(self.n_images, dtype=np.int64), self.n_stims_per_condition)

        if self.give_blanks:
            blanks = np.full(self.n_stims_per_condition, -1, dtype=np.int64)
            trial_order = np.concatenate((repeated_images, blanks), axis=0)
        else:
            trial_order = repeated_images

        permutation = np.random.permutation(trial_order.shape[0])
        self.experiment_stims = trial_order[permutation].tolist()
        self.n_trials = int(len(self.experiment_stims))

        print(f'Total trials for experiment: {self.n_trials}')

    def run_experiment(self):
        print('Experiment starting...')
        self.experiment_running = True
        bool_logged_start = False
        bool_logged_end = False

        self.clock.reset()
        self.master_clock.reset()

        while self.clock.getTime() < self.experiment_delay:
            if self.clock.getTime() >= self.experiment_delay / 2:
                if not bool_logged_start:
                    self.exp_log.log_exp_start(self.master_clock.getTime())
                    bool_logged_start = True
                self.photodiode_square.draw()

            self.window.flip()

        self.absolute_total_time += self.experiment_delay

        for i in range(self.n_trials):
            index = self.experiment_stims[i]
            print(f'Image trial {i + 1} out of {self.n_trials}.')

            if index != -1:
                self.image_stim.image = self.images[index]
                image_name = self.image_paths[index].stem
            else:
                image_name = 'blank'

            stim_info = image_name

            total_time = 0
            self.clock.reset()
            self.photodiode_square.fillColor = self.photodiode_square.lineColor = self.square_color_on

            self.exp_log.log_stimulus(self.master_clock.getTime(), i, [index, image_name], stim_info)

            while self.clock.getTime() < self.image_on_period + total_time:
                if index != -1:
                    self.image_stim.draw()

                self.photodiode_square.draw()
                self.window.flip()

            total_time += self.image_on_period

            self.photodiode_square.fillColor = self.photodiode_square.lineColor = self.square_color_off
            while self.clock.getTime() < self.inter_trial_delay + total_time:
                self.photodiode_square.draw()
                self.window.flip()

        self.clock.reset()
        while self.clock.getTime() < self.experiment_delay:
            if self.clock.getTime() < self.experiment_delay / 2:
                self.photodiode_square.draw()
            if self.clock.getTime() >= self.experiment_delay / 2:
                if not bool_logged_end:
                    self.exp_log.log_exp_end(self.master_clock.getTime(), self.n_trials)
                    bool_logged_end = True
            self.window.flip()

        self.exp_log.save_log()
        self.experiment_running = False
