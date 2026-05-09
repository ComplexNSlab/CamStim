import sys
import threading
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
from core.experiment_discovery import discover_experiment_types

current_exp = None
teensy_board = None
stop_flag = False
bool_DEBUG = True

exp_types = discover_experiment_types()


def execute_exp(exp_name, experiment_id, mouse_id, skip_teensy=False):
    data_aq = ExperimentDAQ(experiment_id, bool_DEBUG)
    teensy_board = None
    if not skip_teensy:
        teensy_board = TeensyController(experiment_id, bool_DEBUG, str(CONFIG_DIR / "teensyParams.yaml"))
        print("\nTeensy started.")
    else:
        print("\nDEBUG: skipping Teensy startup.")

    config_file = str(CONFIG_DIR / f"{exp_name.replace(' ', '_').lower()}_config.yaml")

    exp_type = exp_types.get(exp_name)
    if exp_type is None:
        print("\nExperiment name error.")
        return

    exp = exp_type(experiment_id, mouse_id, data_aq, str(CONFIG_DIR / "monitor_config.yaml"), str(CONFIG_DIR / "config.yaml"), config_file, debug=bool_DEBUG)

    exp.load_experiment_config()
    exp.start_data_acquisition()
    if teensy_board:
        teensy_board.start_teensy()

    print("\nSTARTING experiment...")

    threading.Thread(target=listen_for_stop, args=(teensy_board, exp), daemon=True).start()

    exp.run_experiment()
    global stop_flag
    stop_flag = True

    return teensy_board, exp


def listen_for_stop(teensy_board, exp):
    global stop_flag
    for line in sys.stdin:
        if line.strip().upper() == "STOP":
            print("\nReceived STOP command via subprocess.", flush=True)
            stop_flag = True

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
    try:
        args = [arg for arg in sys.argv[1:] if arg != "--skip-teensy"]
        skip_teensy = (len(args) != len(sys.argv[1:]))

        if len(args) < 3:
            print("\nError in receiving experiment inputs.")
            print("Usage: python utils/wf_main.py <exp_name> <experiment_id> <mouse_id> [--skip-teensy]")
            sys.exit(1)

        exp_name = args[0]
        experiment_id = args[1]
        mouse_id = args[2]
        teensy_board, current_exp = execute_exp(exp_name, experiment_id, mouse_id, skip_teensy=skip_teensy)

        while not stop_flag:
            sleep(0.1)

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
        if current_exp.experiment_running:
            # We need to run the stopping functions.
            pass
        if current_exp.acquisition_running:
            current_exp.stop_data_acquisition()
        try:
            teensy_board.stop_teensy()
        finally:
            print("\nTeensy stopped.")

    finally:
        print("\nExperiment routine completed. Thanks!")