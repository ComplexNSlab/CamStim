# Changes Summary

## GUI Experiment Flow, Debug Mode, and Linux Compatibility
- Added Qt-native experiment controls in `camstim.py`: experiment dropdown, experiment ID, and mouse ID fields.
- Experiment dropdown now populates at GUI startup and remains available independently of camera start/stop.
- Routed GUI Preview/Start actions through selected Qt values instead of terminal/Tk prompts.
- Made right-side control panel scrollable so experiment controls remain visible on smaller windows.
- Fixed stop-camera thread joins to avoid `cannot join thread before it is started` runtime errors.

- Added `DEBUG_SKIP_TEENSY` support in `cam_config.yaml` to run experiments without Teensy hardware.
- `simple_cam_mx.py` now propagates debug no-Teensy mode to `wf_main.py` via `--skip-teensy`.
- `wf_main.py` now supports `--skip-teensy` and bypasses Teensy initialization/start/stop when enabled.

- Fixed subprocess launch robustness:
	- `wf_main.py` now adds repo root to `sys.path` so `core` imports work when launched as a script.
	- Subprocesses launched from `simple_cam_mx.py` now use `sys.executable` and repo-root `cwd`.
	- Added clearer status reporting when experiment subprocess exits with errors.

- Fixed config path handling regressions across runtime modules:
	- `simple_cam_mx.py`, `BaseExperiment.py`, and `TeensyController.py` now resolve relative YAML paths from repository-root `config_files/` first (with fallback).
	- `wf_main.py` and `continuous_sigrok.py` use repository-root `config_files/` paths.

- Updated Linux runtime defaults in `config_files/config.yaml`:
	- Preserved old Windows values as comments.
	- Active defaults now use Linux-safe values (`SAVE_DIR: /home/orlandi/data/camstim`, `SIGROK_EXE: sigrok-cli`).

- Made `continuous_sigrok.py` cross-platform by guarding Windows-only `msvcrt` import and adding stdin-based STOP handling for non-Windows systems.

- Added optional local third-party dependency integration for retinotopy support:
	- Clone `WarpedVisualStim` under `external/`.
	- `RetinotopyExperiment.py` now imports it from `external/WarpedVisualStim`.

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
- Added robust relative-path resolution for YAML config loading in `simple_cam_mx.py`, `BaseExperiment.py`, and `ExperimentDAQ.py`.
- Updated `wf_main.py`, `gui_progress.py`, and `continuous_sigrok.py` to reference YAML files under `config_files/`.
- Removed unused camera config options: `EXPERIMENT`, `SAVE_PROCESSED_FRAMES`, `STRICT_NO_DROP_SAVE`, `PICO_SERIAL_PORT`, `PICO_PWM_FREQUENCY`, `PICO_PWM_DUTY`, `FORCE_FRAMERATE`, `SPECIAL_FRAMERATE`, and `UDP_TRIGGER_PORT`.
- Removed the UDP listener thread from `simple_cam_mx.py`.

## Code Organization & Restructuring
- Created `core/` directory for core library modules: `BaseExperiment.py`, `ExperimentDAQ.py`, `ExperimentLogger.py`, `TeensyController.py`, `experiment_discovery.py`, `mvsdk.py`, `roi_module.py`, `LocallySparseNoise.py`
- Created `utils/` directory for utility scripts and hardware files: `simple_cam_mx.py` (camera helper), `continuous_sigrok.py` (logic analyzer), `wf_main.py` (experiment launcher), `teensyConnectTest.py` (Teensy testing), `WF_TeensyV3.ino` (Teensy firmware)
- Root now contains the main GUI entry point: `camstim.py` (renamed from `gui_progress.py`)
- Moved all non-entry-point Python files out of root for cleaner organization
- Updated all imports across files to reflect new module structure
- Removed `labjack_data.npy` (unused artifact from deprecated LabJack DAQ era)

## Experiment Launch and DAQ Cleanup
- `start_stim` now always uses subprocess launch; removed UDP stim mode (`u`) and method selection prompt from experiment dialogs.
- Removed unused DAQ modules and references for `NISDAQ`, `HTDAQ`, and `BlueDAQ`.
- Removed `run_BLUE_experiment.py`.
- Removed `FlashingLedExperiment.py` (unused, incompatible with current DAQ interface).
- Refactored `ExperimentDAQ.py` (formerly `PCODAQ.py`) to a minimal `ExperimentDAQ` class with only currently used DAQ interface (`sampling_rate`, `ni_log_filename`, `start_everything`, `stop_everything`).
- Extracted `TeensyController` class into its own dedicated module `TeensyController.py` for better separation of concerns.

## Experiment Type Organization and Dynamic Discovery
- Moved all experiment type modules into new `experiment_types/` package directory.
- Created shared `experiment_discovery.py` module for dynamic experiment discovery and loading.
- Replaced hardcoded `exp_types` dictionaries in `wf_main.py` and `simple_cam_mx.py` with dynamic discovery.
- Updated `simple_cam_mx.py` experiment selection dialog from numbered input to visual dropdown listbox.
- Experiment types are now auto-discovered from `experiment_types/` folder at runtime; adding new experiments requires only placing the module file in that folder.

## Script Cleanup and Validation
- Removed unused `sigrok_chunk.py`.
- Repeatedly validated touched runtime files with `python3 -m py_compile` after each major cleanup step.
