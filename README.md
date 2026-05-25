# CAMStim — Camera-Based Visual Stimulus Presentation

A PyQt6-based GUI application for conducting visual neuroscience experiments with dual wavelength real-time camera acquisition and stimulus control via Teensy microcontroller.

## Project Structure

```
camstim/
├── camstim.py                    # Main GUI entry point
├── core/                         # Core library modules
│   ├── BaseExperiment.py        # Abstract base class for experiments
│   ├── ExperimentDAQ.py         # Minimal DAQ interface
│   ├── TeensyController.py      # Teensy microcontroller interface
│   ├── ExperimentLogger.py      # Experiment logging
│   ├── experiment_discovery.py  # Dynamic experiment discovery
│   ├── LocallySparseNoise.py    # Stimulus generation utilities
│   └── ...                      # Other utilities
├── utils/                        # Utility scripts and tools
│   ├── wf_main.py              # Experiment launcher subprocess
│   ├── simple_cam_mx.py         # Camera acquisition and control
│   ├── continuous_sigrok.py     # Logic analyzer integration
│   ├── teensyConnectTest.py     # Teensy connectivity testing
│   └── WF_TeensyV3.ino         # Teensy firmware
├── experiment_types/            # Experiment implementations
│   ├── SimpleOrientationExperiment.py
│   ├── TextureExperimentFB.py
│   └── ...                      # Other experiment types
├── config_files/                # YAML configuration files
│   ├── config.yaml
│   ├── monitor_config.yaml
│   ├── teensyParams.yaml
│   └── ...                      # Experiment-specific configs
└── notebooks/                   # Jupyter analysis notebooks
```

## Installation

### Prerequisites
- Python 3.8+
- Conda (Anaconda or Miniconda)

### Setup

1. **Clone or navigate to the repository:**
   ```bash
   cd /path/to/camstim
   ```

2. **Create a conda environment:**
   ```bash
   conda create -n camstim python=3.10
   conda activate camstim
   ```

3. **Install dependencies:**
   ```bash
   conda install -c conda-forge opencv gtk3 gstreamer pyqt qt-main pyqt6
   pip install 'setuptools<70'
   pip install --upgrade psychopy tifffile h5py scikit-image
   ```
   
   **Notes on compatibility (only for windows):**
   - The setuptools downgrade to <70 is required for the `pkg_resources` module that psychopy needs
   - Upgrading psychopy ensures Python 3.10 compatibility and installs all required dependencies
   
   Optional (for data analysis):
   ```bash
   pip install jupyter
   ```

   Optional (for `RetinotopyExperiment`):
   ```
   pip install -r external/WarpedVisualStim/requirements.txt
   ```
   `RetinotopyExperiment` imports `WarpedVisualStim` from `external/WarpedVisualStim`.

4. **Build the camera acquisition C extension (`cgrabcallback`):**

   The C extension provides a fast `memcpy` path in the camera callback, avoiding Python object
   creation overhead in the hot path. It is optional — `simple_cam_mx.py` falls back to
   `ctypes.memmove` automatically if the extension is not present.

   **macOS / Linux:**
   ```bash
   conda activate camstim
   cd utils
   python setup.py build_ext --inplace
   ```

   **Windows:**

   Prerequisites (one-time setup):
   - Install [Build Tools for Visual Studio 2022](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
     and select the **"Desktop development with C++"** workload.

   Then build from a **conda-enabled command prompt** (e.g. Anaconda Prompt):
   ```bat
   conda activate camstim
   cd utils
   python setup.py build_ext --inplace
   ```
   This produces `cgrabcallback.cpython-310-win_amd64.pyd` in `utils/`.

   > **Note:** The `.so` / `.pyd` file is platform- and Python-version-specific and is not
   > committed to the repository. It must be rebuilt on each machine.

5. **Verify installation:**
   ```bash
   python -c "from core.BaseExperiment import BaseExperiment; print('✓ Installation successful')"
   ```

## Quick Start

1. **Activate the conda environment:**
   ```bash
   conda activate camstim
   ```

2. **Launch the main GUI:**
   ```bash
   python camstim.py
   ```

3. **Select an experiment type** from the dropdown
4. **Configure parameters** via the GUI
5. **Start acquisition** — camera feed and Teensy controls are synchronized

## Entry Points

- **`camstim.py`** — Main GUI application (camera control, experiment selection, live preview)
- **`utils/wf_main.py`** — CLI experiment launcher subprocess (`python utils/wf_main.py <exp_name> <experiment_id> <mouse_id> [--skip-teensy]`)
- **`utils/teensyConnectTest.py`** — Test Teensy connectivity

## Adding New Experiments

1. Create a new experiment class in `experiment_types/` that inherits from `BaseExperiment`
2. Implement `load_experiment_config()` and `draw_stim()` methods
3. Create a corresponding YAML config file in `config_files/`
4. The experiment will be automatically discovered and listed in the GUI

See `experiment_types/SimpleOrientationExperiment.py` for a complete example.

## Configuration

All YAML configuration files are in `config_files/`:
- **`config.yaml`** — General system configuration
- **`monitor_config.yaml`** — Monitor/display parameters
- **`teensyParams.yaml`** — Teensy hardware configuration (port, baud rate, etc.)
- **`*_config.yaml`** — Experiment-specific parameters

## Documentation

See `CHANGES.md` for recent refactoring and architectural decisions.

## Contributors

- Federico Bolaños: original code
- Marina Xu: major reimplementation using a teensy microcontroller and sigrok analyzer as DAQ. New Pyqt6 GUI from scratch
- Jamie Sanson: teensy implementation of the dual wavelength imaging mode and camera triggering. Additional modules for compatibility with RenStimPi system
- Javier Orlandi: code optimization and maintenance

## License

See `LICENSE` file.
