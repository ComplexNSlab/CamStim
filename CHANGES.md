# Changes Summary

## Camera Pipeline and Binning
- Added automatic stop flow after logic-analyzer termination (disable trigger, stop experiment, stop camera).
- GUI preview now applies the configured binning path so display output matches experiment settings.
- Save and display paths are separated; latest-frame snapshot is used for GUI tools (histogram, dFoF, background capture, range adjustment).
- Save queue enqueue from camera callback is now blocking under backpressure to prioritize not losing frames.
- Added `BIN_MODE` in `cam_config.yaml` with `"software"` or `"camera"`.
- Added camera capability probe and hardware-binning setup path (sum mode with MONO16 check) with software fallback.

## Saving, Naming, and Layout
- Unified file naming to `mouseID_experimentID` across saved outputs.
- Unified save layout to `SAVE_DIR/mouseID/experimentID`.
- On experiment start, save directory is prepared and `cam_config.yaml` is copied into the experiment folder.
- Save thread writes processed frames unconditionally; `SAVE_PROCESSED_FRAMES` option was removed.

## Config Simplification
- Replaced old save-settings file usage with `config.yaml`.
- Centralized `SAVE_DIR` and `SIGROK_EXE` in `config.yaml`.
- Moved all project YAML files into `config_files/` and updated runtime loaders to use that location.
- Added robust relative-path resolution for YAML config loading in `simple_cam_mx.py`, `BaseExperiment.py`, and `PCODAQ.py`.
- Updated `wf_main.py`, `gui_progress.py`, and `continuous_sigrok.py` to reference YAML files under `config_files/`.
- Removed unused camera config options: `EXPERIMENT`, `SAVE_PROCESSED_FRAMES`, `STRICT_NO_DROP_SAVE`, `PICO_SERIAL_PORT`, `PICO_PWM_FREQUENCY`, `PICO_PWM_DUTY`, `FORCE_FRAMERATE`, `SPECIAL_FRAMERATE`, and `UDP_TRIGGER_PORT`.
- Removed the UDP listener thread from `simple_cam_mx.py`.

## Experiment Launch and DAQ Cleanup
- `start_stim` now always uses subprocess launch; removed UDP stim mode (`u`) and method selection prompt from experiment dialogs.
- Removed unused DAQ modules and references for `NISDAQ`, `HTDAQ`, and `BlueDAQ`.
- Removed `run_BLUE_experiment.py`.
- Refactored `PCODAQ.py` to a minimal `ExperimentDAQ` class with only currently used DAQ interface (`sampling_rate`, `ni_log_filename`, `start_everything`, `stop_everything`), while keeping `Teensy` support in place.

## Experiment Type Organization and Dynamic Discovery
- Moved all experiment type modules into new `experiment_types/` package directory.
- Created shared `experiment_discovery.py` module for dynamic experiment discovery and loading.
- Replaced hardcoded `exp_types` dictionaries in `wf_main.py` and `simple_cam_mx.py` with dynamic discovery.
- Updated `simple_cam_mx.py` experiment selection dialog from numbered input to visual dropdown listbox.
- Experiment types are now auto-discovered from `experiment_types/` folder at runtime; adding new experiments requires only placing the module file in that folder.

## Script Cleanup and Validation
- Removed unused `sigrok_chunk.py`.
- Repeatedly validated touched runtime files with `python3 -m py_compile` after each major cleanup step.
