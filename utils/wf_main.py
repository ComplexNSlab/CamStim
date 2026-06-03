import sys
import os
import threading
import subprocess
import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path(__file__).resolve().parent.parent / 'config_files'
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.stdout.reconfigure(line_buffering=True)
sys.stdin.reconfigure(line_buffering=True)

from time import sleep, time

from core.ExperimentDAQ import ExperimentDAQ
from core.TeensyController import TeensyController
from core.experiment_discovery import discover_experiment_types, resolve_experiment_config_file

current_exp = None
teensy_board = None
movement_sensor_process = None
stop_flag = False
bool_DEBUG = True

exp_types = discover_experiment_types()


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(value)


def _load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def _start_movement_sensor(experiment_id, mouse_id, exp, preview=False):
    if preview:
        return None

    config = _load_yaml(CONFIG_DIR / 'config.yaml')
    if not _as_bool(config.get('USE_MOVEMENT_SENSOR', False)):
        return None

    movement_cfg = _load_yaml(CONFIG_DIR / 'movementSensor.yaml')
    sensor_port = movement_cfg.get('MOVEMENT_SENSOR_PORT', movement_cfg.get('port'))
    plot_flag = '1' if _as_bool(movement_cfg.get('PLOT', False)) else '0'
    window_seconds = movement_cfg.get('WINDOW_SECONDS', 30)
    if not sensor_port:
        print('\nMovement sensor enabled but no port found in config_files/movementSensor.yaml.')
        return None

    output_dir = Path(getattr(exp, 'save_dir', '') or REPO_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f'{mouse_id}_{experiment_id}_movement.csv'

    cmd = [
        sys.executable,
        str((REPO_ROOT / 'utils' / 'mpulogger.py').resolve()),
        '--plot', plot_flag,
        '--window-seconds', str(window_seconds),
        '--file', str(output_csv),
        '--port', str(sensor_port),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdin=subprocess.PIPE,
        text=True,
    )
    print(f'\nMovement sensor started on {sensor_port}, saving to {output_csv}.')
    return proc


def _stop_movement_sensor(proc):
    if proc is None:
        return

    try:
        if proc.poll() is None and proc.stdin is not None:
            proc.stdin.write('STOP\n')
            proc.stdin.flush()

        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)
        print('\nMovement sensor stopped.')
    except Exception as e:
        print(f'\nError stopping movement sensor: {e}')
    finally:
        try:
            if proc.stdin is not None and not proc.stdin.closed:
                proc.stdin.close()
        except Exception:
            pass


def _force_exit_if_still_running(delay_s=8.0):
    """Fail-safe: if STOP cleanup hangs, exit the worker process anyway."""
    try:
        sleep(float(delay_s))
    except Exception:
        return
    if stop_flag:
        try:
            sys.stdout.flush()
        except Exception:
            pass
        os._exit(0)


def execute_exp(exp_name, experiment_id, mouse_id, skip_teensy=False, preview=False, save_outputs=None):
    data_aq = ExperimentDAQ(experiment_id, bool_DEBUG)
    teensy_board = None
    movement_proc = None
    if not skip_teensy:
        teensy_board = TeensyController(experiment_id, bool_DEBUG, str(CONFIG_DIR / "teensyParams.yaml"))
        print("\nTeensy started.")
    else:
        print("\nDEBUG: skipping Teensy startup.")

    exp_type = exp_types.get(exp_name)
    if exp_type is None:
        print("\nExperiment name error.")
        return

    config_file = str(resolve_experiment_config_file(exp_name, exp_type))

    exp = exp_type(
        experiment_id,
        mouse_id,
        data_aq,
        str(CONFIG_DIR / "monitor_config.yaml"),
        str(CONFIG_DIR / "config.yaml"),
        config_file,
        debug=bool_DEBUG,
        preview=preview,
        save_outputs=save_outputs,
    )

    exp.load_experiment_config()
    exp.start_data_acquisition()
    if teensy_board:
        teensy_board.start_teensy()

    # Movement sensor should run independently from Teensy debug mode.
    movement_proc = _start_movement_sensor(experiment_id, mouse_id, exp, preview=preview)

    print("\nSTARTING experiment...")

    threading.Thread(target=listen_for_stop, args=(teensy_board, exp, movement_proc), daemon=True).start()

    exp.run_experiment()
    global stop_flag
    stop_flag = True

    return teensy_board, exp, movement_proc


def listen_for_stop(teensy_board, exp, movement_proc):
    global stop_flag
    for line in sys.stdin:
        if line.strip().upper() == "STOP":
            print("\nReceived STOP command via subprocess.", flush=True)
            stop_flag = True

            # If experiment teardown blocks, force process exit after a grace period
            # so the parent does not need to terminate this worker.
            threading.Thread(target=_force_exit_if_still_running, args=(8.0,), daemon=True).start()

            # Stop movement sensor first so it exits before Teensy is stopped.
            _stop_movement_sensor(movement_proc)

            # Stop Teensy if available
            if teensy_board:
                try:
                    teensy_board.stop_teensy()
                    print("\nTeensy stopped.", flush=True)
                except Exception as e:
                    print(f"\nError stopping Teensy: {e}")

            # Stop experiment if available
            if exp:
                try:
                    if hasattr(exp, "stop_experiment"):
                        exp.stop_experiment()
                    if hasattr(exp, "stop_data_acquisition"):
                        exp.stop_data_acquisition()
                    exp.experiment_running = False
                    print("\nExperiment stopped.", flush=True)
                except Exception as e:
                    print(f"\nError stopping experiment: {e}")
            break


if __name__ == "__main__":
    current_exp = None
    teensy_board = None
    movement_sensor_process = None
    try:
        raw_args = sys.argv[1:]
        preview = ('--preview' in raw_args)
        save_preview = ('--save-preview' in raw_args)
        skip_teensy = ('--skip-teensy' in raw_args)
        args = [arg for arg in raw_args if arg not in ('--skip-teensy', '--preview', '--save-preview')]

        if len(args) < 3:
            print("\nError in receiving experiment inputs.")
            print("Usage: python utils/wf_main.py <exp_name> <experiment_id> <mouse_id> [--skip-teensy]")
            sys.exit(1)

        exp_name = args[0]
        experiment_id = args[1]
        mouse_id = args[2]
        if preview:
            if save_preview:
                print("\nPreview mode active: experiment files will be saved.")
            else:
                print("\nPreview mode active: file and data outputs are disabled.")

        teensy_board, current_exp, movement_sensor_process = execute_exp(
            exp_name,
            experiment_id,
            mouse_id,
            skip_teensy=skip_teensy,
            preview=preview,
            save_outputs=(not preview) or save_preview,
        )

        while not stop_flag:
            sleep(0.1)

        _stop_movement_sensor(movement_sensor_process)
        if teensy_board:
            teensy_board.stop_teensy()
            print("\nTeensy stopped.")
        if current_exp:
            current_exp.stop_data_acquisition()
            print("\nData acquisition stopped.")
            # exp.stop_experiment()

        print("\nALL DONE with experiment {}! ".format(experiment_id))
    
    except KeyboardInterrupt:
        print("\nReceived CTRL-C event")
        _stop_movement_sensor(movement_sensor_process)
        if current_exp and current_exp.experiment_running:
            # We need to run the stopping functions.
            pass
        if current_exp and current_exp.acquisition_running:
            current_exp.stop_data_acquisition()
        if teensy_board:
            try:
                teensy_board.stop_teensy()
            finally:
                print("\nTeensy stopped.")

    finally:
        print("\nExperiment routine completed. Thanks!")