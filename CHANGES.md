# Changes Summary

## Camera and GUI
- Added automatic stop after logic analyzer shutdown, including hardware-trigger disable, experiment stop, and camera stop.
- Moved display binning into the GUI path so the preview matches the configured bin size, with intensity normalized by bin area.
- Reset live averaging and FPS on the first frame after experiment start, and average only over filled buffer entries.
- Split saving and display into separate queues so GUI updates do not block frame saving.
- Kept the display queue bounded to only recent frames and added latest-frame snapshot access for histogram, dFoF, background capture, and range-adjust tools.
- Replaced unsafe direct queue clearing with a safe drain helper.
- Switched callback queue writes to non-blocking enqueue behavior.
- Added save-queue backpressure so display updates pause when the save queue is close to full.
- Added strict save-queue overflow handling, optional drop mode, and dropped-frame counters to avoid silent data loss during acquisition.

## Saving and Naming
- Standardized saved outputs to `mouseID_experimentID` for `.sr`, `.bin`, metadata `.npy`, experiment `.pkl`, and `.txt` files.
- Changed the save layout to `SAVE_DIR/mouseID/experimentID` without extra recording subfolders.
- Create the experiment save directory at start and copy `cam_config.yml` into it as `cam_config.yaml`.
- Improved save throughput with buffered writes, batched frame writes, and configurable queue settings including `SAVE_QUEUE_MAX_FRAMES`, `SAVE_BATCH_FRAMES`, and `STRICT_NO_DROP_SAVE`.

## Configuration and Paths
- Replaced `save_settings_config.yaml` with `config.yaml` and updated active code to use it.
- Centralized `SAVE_DIR` and `SIGROK_EXE` in `config.yaml` and updated sigrok helpers to read from there.
- Updated subprocess launches to resolve local repo scripts instead of relying on stale hardcoded external paths.
- Removed redundant `SAVE_DIR` from `cam_config.yml` and kept save-related runtime settings in the central config.

## Cleanup and Validation
- Removed legacy `simple_cam.py` and kept `simple_cam_mx.py` as the active camera path.
- BaseExperiment now saves logs directly in `SAVE_DIR/mouseID/experimentID` using the same `mouseID_experimentID` base name.
- Added this `CHANGES.md` and validated modified runtime files with repeated `python3 -m py_compile` checks and path/reference audits.
