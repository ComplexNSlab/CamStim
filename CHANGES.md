
# Version [0.2.4]

# 2026-05-29

- Updated Retinotopy experiment runtime path (`experiment_types/RetinotopyExperiment.py`):
	- Fixed monitor/indicator setup to use `WarpedVisualStim.MonitorSetup.Monitor` and `Indicator` directly.
	- Added robust stop integration (`stop_experiment`) and status/progress updates compatible with current subprocess control.
	- Retinotopy stimulus is now displayed at full monitor pixel size instead of downsampled frame size.

- Removed unused legacy retinotopy config block (`config_files/RetinotopyExperiment_config.yaml`):
	- Deleted obsolete `DisplaySequence`/`ds_*` options that are not used by the current Retinotopy experiment implementation.

- Improved compatibility with modern NumPy for vendored `WarpedVisualStim`:
	- Added alias compatibility shim in `external/WarpedVisualStim/WarpedVisualStim/__init__.py`.
	- Replaced deprecated `np.bool` usage with `np.bool_` in `external/WarpedVisualStim/WarpedVisualStim/StimulusRoutines.py`.
	- Updated alias checks to avoid `FutureWarning`.

- Hardened teardown safety in `core/BaseExperiment.py`:
	- Guarded `__del__` window close to avoid cleanup errors when experiment initialization fails early.

# Version [0.2.3]

# 2026-05-29

- Reworked Fourier Ring Correlation execution to use a standalone worker script:
	- GUI FRC button now launches `utils/frc_worker.py` in a separate process instead of running FRC inside `camstim.py`.
	- This isolates matplotlib/FRC runtime from the Qt GUI path and preserves reliable interactive plotting behavior.

- Simplified GUI FRC subprocess launch logic in `camstim.py`:
	- Uses `sys.executable` directly for worker launch.
	- Removed the extra dependency preflight probe to reduce launch complexity.

- Expanded worker analysis in `utils/frc_worker.py`:
	- Computes and plots both `frc.two_frc(img1, img2)` and `frc.one_frc(img1)`.
	- Both plots include threshold and resolution marker when an intersection is found.

- Updated setup documentation in `README.md`:
	- Added explicit install step for `setproctitle` and `frc` in the Python dependency list.

- Added utility logger script `utils/mpulogger.py`:
	- CLI tool for serial MPU logging to CSV, with optional live matplotlib plotting mode.

- Polished GUI image-processing controls and histogram UX in `camstim.py`:
	- Reworked normalization controls to Auto/Disable buttons and moved histogram/FRC to explicit action buttons.
	- Updated histogram display behavior and interaction flow for the current standalone-window implementation.

- Added movement sensor lifecycle integration (`utils/wf_main.py`, `utils/mpulogger.py`):
	- New `USE_MOVEMENT_SENSOR` gate in `config_files/config.yaml` now controls optional movement logger startup during experiments.
	- Movement sensor serial port is read from `config_files/movementSensor.yaml` and output is saved as `{mouse_id}_{experiment_id}_movement.csv` in the experiment save directory.
	- Movement logger now supports graceful shutdown via `STOP` stdin command and signal handling.
	- Experiment shutdown order now stops the movement logger before Teensy shutdown.

- Improved movement logger plotting stability (`utils/mpulogger.py`):
	- Fixed plot-mode freeze on close by decoupling serial/CSV logging from the matplotlib UI loop.
	- Closing the plot window now cleanly exits plotting while headless CSV logging continues until STOP/signal.

- Hardened experiment-stop shutdown path for movement logging (`utils/simple_cam_mx.py`):
	- Stopping an experiment now sends `STOP` to `wf_main.py` and waits for graceful exit before any forced terminate/kill fallback.
	- Removed eager pre-termination of an existing experiment process during restart, preventing premature teardown of child cleanup.

- Added configurable movement plot window duration (`config_files/movementSensor.yaml`, `utils/wf_main.py`, `utils/mpulogger.py`):
	- New `WINDOW_SECONDS` setting in `config_files/movementSensor.yaml`.
	- `wf_main.py` forwards the setting to `mpulogger.py` via `--window-seconds`.

- Refined live movement plotting behavior (`utils/mpulogger.py`):
	- Plot now uses real-time x-axis in seconds over a sliding window.
	- X/Y/Z are plotted as online z-scores with vertical offsets; T remains raw with its own offset.

- Updated Arduino-side movement tick signaling (`utils/mpu6050plot/mpu6050plot.ino`):
	- Replaced blocking pulse timing with non-blocking `millis()` scheduling.
	- Pin 2 now emits a 100 ms HIGH pulse every 1 second without delaying sensor logging.

# Version [0.2.2]

# 2026-05-29

- Fixed camera data not being saved during real experiments (`utils/simple_cam_mx.py`):
	- `start_experiment` command handler was calling `get_exp_params(...)` without `save_outputs=True`.
	- `_prepare_save_directory` was never called, so `save_dir_ready` remained `False`.
	- The save worker loop waited indefinitely for a ready save directory while frames accumulated in the queue unwritten.
	- Fixed by passing `save_outputs=True` explicitly when handling the `start_experiment` command.

# 2026-05-28

- Added camera-less experiment preview mode (`camstim.py`, `utils/wf_main.py`, `utils/simple_cam_mx.py`):
	- Preview can be launched without starting the camera.
	- Mouse ID and experiment ID fields are optional during preview.
	- If both IDs are entered, preview behaves like a real run and saves files; otherwise no files are created.
	- Added `--preview` and `--save-preview` CLI flags to `wf_main.py`.
	- `save_outputs` flag threaded through GUI → worker → subprocess.

- Added `save_outputs` gating to `core/BaseExperiment.py` and `core/ExperimentLogger.py`:
	- `create_save_directories` is skipped when `save_outputs=False`.
	- `ExperimentLogger` skips config copy and log write when `save_outputs=False`.
	- Fixed `create_save_directories` to use `os.makedirs(exist_ok=True)`.
	- Applied same `save_outputs` pattern to `experiment_types/Continuous.py` and `experiment_types/SpontaneousActivity.py`.

- Added new experiment type `TextureExperimentFBSimple` (`experiment_types/TextureExperimentFBSimple.py`):
	- Loads all TIFF images from a folder automatically.
	- Single on-period per trial (no `image_repeat_times`, no `image_off_period`).
	- Single `np.random.permutation` randomization over all stimuli and blanks.
	- `stim_info` and `image_name` are identical (image filename stem, or `'blank'`).
	- Added `config_files/TextureExperimentFBSimple_config.yaml`.
	- Registered display name in `core/experiment_discovery.py`.

- Fixed `config_files/config.yaml`: switched `SAVE_DIR` to macOS local path `/Users/orlandi/data`.

- Fixed experiment start routing regression in `camstim.py`:
	- `CameraProcessClient.get_exp_params(...)` now routes by explicit `preview` flag instead of `hardware_trigger_enabled`.
	- Real experiment starts always send `start_experiment`, preventing accidental preview/no-save execution paths.
	- Preview calls now explicitly pass `preview=True`.

- Added spherical-warp compensation for the photodiode square in `core/BaseExperiment.py`:
	- When spherical warping is enabled, the photodiode square position and size are remapped to pre-warp coordinates so the final warped footprint stays close to the configured target area.
	- Added optional monitor setting `compensate_photodiode_for_warp` (default behavior is enabled when omitted).
	- Logs target vs compensated photodiode geometry at startup for validation.

# 2026-05-26
- Added native C-based save-path binning in `utils/cgrabcallback.c`:
	- New `bin_u8_batch_sum_pow2(src_addr, n, in_h, in_w, bin_size, dst_addr)` API for uint8 batch sum-binning into uint16 output.
	- Runs with the GIL released to reduce Python-thread contention during heavy binning work.

- Integrated selectable C binning backend into `utils/simple_cam_mx.py` save pipeline:
	- Save worker now uses the native C binning path for compatible cases (uint8 + power-of-two `BIN_SIZE`) and falls back to the existing NumPy path otherwise.
	- Added reusable C-output buffer handling to avoid repeated allocations.
	- Added startup/status logging of active binning backend (`C extension` vs `NumPy fallback`).

- Added save-path binning timing diagnostics in `utils/simple_cam_mx.py`:
	- New config flag `DEBUG_SAVE_BINNING_TIMING`.
	- Reports per-session aggregate binning timing for both C and NumPy paths (`batches`, `total`, `mean_ms`).

- Added explicit binning-backend config control:
	- New `USE_C_BINNING` option in `config_files/cam_config.yaml`.
	- Allows forcing NumPy binning even when the C extension is available.
	- Observed in-session benchmark: C-based save binning was approximately 4x faster than NumPy fallback on current workload.

- Simplified buffer-capacity tuning by making queue size derived from pool size:
	- Removed `SAVE_QUEUE_MAX_FRAMES` from `config_files/cam_config.yaml`.
	- `utils/simple_cam_mx.py` now auto-derives queue capacity as `ceil(1.5 * FRAME_POOL_FRAMES)` (with a minimum floor), making `FRAME_POOL_FRAMES` the primary buffering knob.

- Updated high-throughput defaults in `config_files/cam_config.yaml` for current testing:
	- `FRAME_POOL_FRAMES: 1536`
	- `USE_C_BINNING: true`
	- `DEBUG_SAVE_BINNING_TIMING: true`

# 2026-05-25
- Added a GUI-only `Live Display Updates` toggle in Image Processing (`camstim.py`):
	- When disabled, the camera worker skips gathering `display_frame_data`, so no new preview frames are produced for the GUI.
	- When enabled, live display resumes with the existing display-copy backend behavior.

- Removed config wiring for the live-display toggle:
	- Deleted `DISPLAY_OUTPUT_ENABLED` from `config_files/cam_config.yaml`.
	- Client and worker now default display output to enabled at startup and rely on runtime GUI commands for changes.

# 2026-05-23
- Added sensor-temperature support through MindVision `CameraSpecialControl`:
	- Added `CameraGetSensorTemperature(hCamera)` in `core/mvsdk2024.py`.
	- Uses empirically identified control code `0x0014` and decodes a `float32` temperature in °C.

- Added a temperature probe utility:
	- New `utils/probe_special_control.py` sweeps undocumented control codes and reports plausible temperature hits.

- Added live temperature to GUI status text:
	- `camstim.py` now shows `Temp: xx.x°C` in `self.stats_label` when available.

- Added per-frame temperature persistence in save metadata:
	- `utils/simple_cam_mx.py` now attaches temperature samples to queued save frames.
	- Saved metadata now includes `sensor_temperatures_c` aligned with saved frames/timestamps.
	- Added `sensor_temperature_stats_c` summary (`count`, `min`, `max`, `mean`) when valid samples exist.

- Improved camera worker shutdown robustness:
	- `CameraProcessClient.stop()` now force-terminates the worker process if graceful join times out, preventing stale camera-handle leaks across restarts.

# 2026-05-22
- Simplified display-frame copy handling in `utils/simple_cam_mx.py`:
	- Added optional mutable display mode controlled by `DISPLAY_MUTABLE_BUFFERS`.
	- When enabled, the callback preallocates a single full-frame `display_frame_data` buffer once and reuses it for every frame.
	- The callback now copies directly into that preallocated buffer via `cgrabcallback.fast_memcpy` (or `ctypes.memmove` fallback), removing per-frame bytes allocation overhead in display mode.
	- Default behavior remains immutable display handoff when mutable mode is disabled.

- Added `DISPLAY_MUTABLE_BUFFERS` to `config_files/cam_config.yaml` (default `false`) to keep safe defaults while allowing display-only low-overhead mode.

# 2026-05-21
- Improved ROI and preview handling in the GUI and worker:
	- ROI selection now stays anchored to the displayed image rect during drag, which fixes inconsistent ROI draw behavior on scaled previews.
	- ROI apply in the GUI now trims requested ROI width/height to the nearest valid multiple of `BIN_SIZE` when `BIN_EXP_LIVE` is enabled and reports the adjustment in status.
	- GUI frame decoding now tolerates oversized/padded frame buffers after ROI changes by slicing to the active resolution before reshape, preventing preview stalls after ROI apply.
	- Histogram rendering now uses a Qt window instead of OpenCV HighGUI calls from a background thread, avoiding the OpenCV GUI crash seen on macOS and keeping behavior consistent in the existing Qt app.

- Improved configuration display behavior in the GUI (`camstim.py`):
	- Camera and Teensy configuration panes now populate on application load instead of waiting for camera start.
	- Experiment configuration now refreshes immediately when the experiment dropdown selection changes.
	- Experiment config file lookup now accepts both legacy and snake-case filenames so `SpontaneousActivity` config files still load correctly.

- Improved saved frame provenance in `utils/simple_cam_mx.py`:
	- Metadata now includes full ROI coordinates and ROI dimensions alongside saved frame dimensions.
	- Initial preview TIFF export now also writes binned companion TIFFs when experiment binning is enabled.

- Added richer camera startup diagnostics in `utils/simple_cam_mx.py`:
	- Startup report now enumerates available resolution presets (`resolution_modes`) including output size, FOV, and per-mode bin/skip settings.
	- Startup report now includes decoded binning capability masks (`binning_support`) and raw mask values (`binning_masks_raw`) for sum/average/skip.
	- Added explicit startup status message indicating active frame-grab implementation:
		- C extension path: `cgrabcallback.fast_memcpy`
		- Python fallback path: `ctypes.memmove`

- Fixed GUI preview latency in the camera worker:
	- Removed the redundant display-frame queue-depth gate so the worker always publishes the newest frame instead of holding back updates when the queue is already configured to keep only the latest item.

- Added interactive ROI controls to the GUI (`camstim.py`):
	- New `Image -> Select ROI` action (enabled only while camera is running).
	- ROI drawing on the live preview via drag-rectangle interaction.
	- GUI sends ROI requests to the camera worker and reports applied ROI dimensions/offsets in status.

- Added ROI control commands to the camera worker (`utils/simple_cam_mx.py`):
	- New `set_roi` command applies custom ROI using `CameraSetImageResolution` with pause/replay handling.
	- New `reset_roi` command restores full-frame capture using camera capability max width/height.
	- ROI apply path now refreshes frame pool configuration and republishes ready state after resolution changes.

- Added GUI robustness for dynamic ROI resolution changes (`camstim.py`):
	- Clears cached latest frame on ROI apply events.
	- Drops stale frames whose byte/pixel count does not match current camera dimensions, preventing reshape errors during ROI transitions.

- Added `Image -> Reset ROI` action (enabled only while camera is running) to return to full-frame acquisition from the GUI.

# 2026-05-20 (3)
- Renamed `baseExperiment` experiment type to `spontaneousActivity` for clarity:
	- `experiment_types/baseExperiment.py` → `experiment_types/SpontaneousActivity.py`; class renamed `SpontaneousActivity`.
	- `config_files/baseExperiment.yaml` → `config_files/SpontaneousActivity_config.yaml`; `EXPERIMENT_NAME` updated to match.
  - `experiment_types/__init__.py` removed (no longer needed; experiment discovery is fully dynamic).
  - `core/experiment_discovery.py` patched to handle the new class/module naming so `SpontaneousActivity` is correctly discovered and listed in the GUI dropdown.

# 2026-05-20 (2)
- Added `utils/cgrabcallback.c` — a C extension that replaces `ctypes.memmove` in the camera `GrabCallback` hot path with a direct address-to-address `memcpy`, eliminating Python object creation overhead per frame.
  - Exposes `fast_memcpy(dst_addr, src_addr, nbytes)` (raw integer pointer addresses) and `fast_memcpy_from_buf(dst_addr, src_buffer, nbytes)` (Python buffer as source).
  - `simple_cam_mx.py` imports the extension with a graceful `ImportError` fallback to `ctypes.memmove` when the `.so`/`.pyd` is not present.
  - `pRawData` is now cast through `ctypes.cast(..., c_void_p).value` before the call, ensuring compatibility on Windows where the mvsdk may pass a ctypes pointer object instead of a plain integer.
  - Frame pool slot addresses are cached once in `_configure_frame_pool` (`self._frame_pool_addrs`) so no per-frame address lookup is needed in the callback.
- Added `utils/setup.py` build script for the C extension (setuptools, Python 3.10).
- Updated `README.md` Installation section with step-by-step build instructions for both macOS/Linux and Windows (including required MSVC Build Tools prerequisite).

# 2026-05-20
- Major camera framegrab callback and save pipeline improvements:
	- Refactored threading and queueing for camera acquisition and save path to eliminate queue corruption and runaway save queue issues.
	- Implemented bounded queues, latest-frame snapshot for display, and non-blocking enqueue for high-throughput, no-drop saving.
	- Added YAML-configurable queue and batch sizes for the save pipeline.
	- Strict overflow handling: no silent frame drops, with warnings if the pipeline is overloaded.
	- Added preallocated frame buffer pool for the camera callback to avoid per-frame allocation stalls.
	- Optionally disables Python garbage collection during acquisition for lower latency.
	- Added callback timing instrumentation (perf_counter) and session summary/metadata export for performance debugging.
	- Save file path and collision handling now robust and user-informative.
	- All changes validated for 2048x2048@50fps, 16GB RAM, SSD, and strict no-frame-loss requirements.
- Refactored `BaseExperiment` to be fully config-driven and minimal:
	- Wait time and experiment name are now loaded from `config_files/baseExperiment.yaml`.
	- Experiment name is logged as `exp_protocol` in `exp_log.log['exp_parameters']` (matching VisualFieldMapping convention).
	- All dependencies on psychopy, monitor, and photodiode logic have been removed.
	- The experiment simply logs start/end, waits for the configured time, and saves the log.
	- File is now clean, minimal, and ready for extension or use as a generic experiment template.
- Created `config_files/baseExperiment.yaml` with default wait time and name fields.
- Updated changelog to document all changes.
# Changelog

## 2026-05-09 (2)
- Added root `VERSION` file containing the application version string (currently `0.1.0`).
- `camstim.py` now reads `VERSION` at startup and displays the version in the window title (`camstim 0.1.0`), startup/stop display text, and the stats label.
- `utils/simple_cam_mx.py` injects a `VERSION:` line into each copied YAML config file (`cam_config.yaml`, `teensyParams.yaml`) during experiment save-directory setup, preserving any pre-existing `VERSION` key unchanged.
- Removed `.DS_Store`, `__pycache__` directories, and `.pyc` bytecode files from the working tree; these are already covered by `.gitignore`.
- Removed UDP command-server functionality from `utils/wf_main.py`; experiment launch is now CLI-only via script arguments (with optional `--skip-teensy`).

## 2026-05-09
- Gain controls now use camera capability-reported multiplier range/step in the GUI, including step-locked spinner increments and value snapping to valid hardware steps.
- Clarified `ANALOG_GAIN` semantics in `config_files/cam_config.yaml` as real multiplier units (float values like `2.5`) rather than raw SDK integer units.
- Removed legacy root-level `cam_config.yaml`; runtime now uses `config_files/cam_config.yaml` as the config source.
- Restored GUI-side normalization behavior in multiprocessing mode by computing display dynamic range from live GUI frames.
- Reworked GUI `Remove Background` and `Enable dFoF` toggles to run fully in the GUI process (display-only) with no worker-side acquisition changes.
- Added shape-alignment safeguards for GUI background/dFoF references so display processing no longer fails on mixed-size frames (for example binned display vs full-resolution references).
- Updated histogram rendering to always use the full 8-bit domain (`0..255`) with adaptive bin count based on observed intensity distribution.
- Reduced histogram GUI overhead with throttled redraw cadence and skip-when-frame-unchanged logic.
- Added GUI-only special-pixel highlighting (0 shown in blue, 255 shown in red) and exposed it as a new toggle in Image Processing controls.
- Rearranged Image Processing controls into a 2-column layout (3 options per column).

## 2026-05-08
- Moved the camera acquisition path into a dedicated subprocess so the GUI no longer shares the same Python execution path as frame grabbing.
- Added a GUI proxy layer in `camstim.py` to control camera, trigger, experiment, and status flow through IPC queues.
- Kept live display working by forwarding the latest frame snapshot from the worker back to the GUI.
- Restored experiment-complete auto-stop behavior so the GUI returns to the same stop-experiment / stop-camera state after a run finishes.
- Fixed the histogram normalize crash by routing the checkbox to the GUI proxy instead of the old in-process camera object.
- Hardened save-thread shutdown so the save file handle is closed idempotently and no longer raises `OSError: handle is closed` at experiment end.
- Added Windows-only priority boosts for the camera worker and callback thread to reduce scheduling interference during acquisition.

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

## Visual Field Mapping Updates
- `VisualFieldMapping` now reads temporal frequency from config and includes it in the stimulus combinations.
- Added `grating_phase_temporal_frequencies` list in `visual_field_mapping_config.yaml` for per-trial temporal-frequency sweeps.
- Updated trial stimulus logging to include temporal frequency alongside position/orientation/sf/size.
- Refined blank-trial behavior: blanks are now appended once per repeat (total blanks = `n_repeats`) after each repeat's full parameter sweep.
- Added explanatory comments in `visual_field_mapping_config.yaml` for orientation units, size/position units, and supported `grating_mask` options.

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
