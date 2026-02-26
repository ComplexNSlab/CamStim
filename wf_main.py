import sys
import numpy as np
import socket
import json
import threading

sys.stdout.reconfigure(line_buffering=True)
sys.stdin.reconfigure(line_buffering=True)

from time import sleep, time

# the imports below are no longer needed, since for now there is no need to support NIDAQMX anymore
#from NISDAQ import NISDAQ
from PCODAQ import PCODAQ
#from HTDAQ import HTDAQ
from PCODAQ import Teensy

from SimpleOrientationExperiment import SimpleOrientationExperiment
from TextureExperimentFB import TextureExperimentFB
from TextureExperimentFBVGG import TextureExperimentFBVGG
from TextureExperimentFBVGGMultiTime import TextureExperimentFBVGGMultiTime
from DynamicBatteryExperiment import  DynamicBatteryExperiment
from SquareExperiment import SquareExperiment
from LocallySparseNoiseExperiment import LocallySparseNoiseExperiment
from ElevationMapperExperiment import ElevationMapperExperiment
from RetinotopyExperiment import RetinotopyExperiment
from VisualFieldMapping import VisualFieldMapping

current_exp = None
teensy_board = None
current_exp_thread = None
stop_flag = False
experiment_running = False
bool_DEBUG = True


exp_types = {"Locally Sparse Noise": LocallySparseNoiseExperiment,
            "Dynamic Battery": DynamicBatteryExperiment,
            "Simple Orientation": SimpleOrientationExperiment,
            "Elevation Mapper": ElevationMapperExperiment,
            "Retinotopy": RetinotopyExperiment,
            "Texture FB": TextureExperimentFB,
            "Texture FB-VGG": TextureExperimentFBVGG,
            "Texture FB-VGGMultiTime": TextureExperimentFBVGGMultiTime,
            "Square": SquareExperiment,
            "Visual Field Mapping": VisualFieldMapping}


def execute_exp_in_thread(exp_name, experiment_id, mouse_id):
    global current_exp, teensy_board, experiment_running, current_exp_thread

    try:
        config_file = f"{exp_name.replace(' ', '_').lower()}_config.yaml"

        exp_type = exp_types.get(exp_name)
        if exp_type is None:
            print("\nExperiment name error.")
            return


        data_aq = PCODAQ(experiment_id, bool_DEBUG)
        teensy_board = Teensy(experiment_id, bool_DEBUG, "teensyParams.yaml")
        print("\nTeensy started.")

        current_exp = exp_type(experiment_id, mouse_id, data_aq, "monitor_config.yaml", "save_settings_config.yaml", config_file, debug=bool_DEBUG)

        current_exp.load_experiment_config()
        current_exp.start_data_acquisition()
        teensy_board.start_teensy()

        experiment_running = True
        print("\nSTARTING experiment...")

        current_exp.run_experiment()
        experiment_running = False

        # threading.Thread(target=listen_for_stop, args=(teensy_board, exp), daemon=True).start()

        print("\nALL DONE with experiment {}! ".format(experiment_id))

    except Exception as e:
        print(f"\nError in experiment: {e}")
        experiment_running = False

    # return teensy_board, exp


def execute_exp(exp_name, experiment_id, mouse_id):
    data_aq = PCODAQ(experiment_id, bool_DEBUG)
    teensy_board = Teensy(experiment_id, bool_DEBUG, "teensyParams.yaml")
    print("\nTeensy started.")

    config_file = f"{exp_name.replace(' ', '_').lower()}_config.yaml"

    exp_type = exp_types.get(exp_name)
    if exp_type is None:
        print("\nExperiment name error.")
        return

    exp = exp_type(experiment_id, mouse_id, data_aq, "monitor_config.yaml", "save_settings_config.yaml", config_file, debug=bool_DEBUG)

    exp.load_experiment_config()
    exp.start_data_acquisition()
    teensy_board.start_teensy()

    experiment_running = True
    print("\nSTARTING experiment...")

    threading.Thread(target=listen_for_stop, args=(teensy_board, exp), daemon=True).start()

    exp.run_experiment()
    experiment_running = False

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
        if len(sys.argv) > 1:
            exp_name = sys.argv[1]
            experiment_id = sys.argv[2]
            mouse_id = sys.argv[3]
            teensy_board, current_exp = execute_exp(exp_name, experiment_id, mouse_id)

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

        else:
            print("\nError in receiving experiment inputs. Starting UDP trigger mode.")
            UDP_IP = "127.0.0.1" 
            UDP_PORT = 5005
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind((UDP_IP, UDP_PORT))
            print(f"\nWaiting for commands on {UDP_IP}:{UDP_PORT}")

            while True:
                data, address = sock.recvfrom(1024)
                msg = data.decode()
                print(f"\nReceived message: {msg}")

                try:
                    payload = json.loads(msg)
                except json.JSONDecodeError:
                    print("\nInvalid message. Requires JSON.")
                    continue

                cmd = payload.get("cmd")

                if cmd == "START":
                    if experiment_running:
                        print("\nExperiment already running. Ignoring START.")
                        continue

                    print("\nReceived START command.")
                    exp_name = payload["exp_name"]
                    experiment_id = payload["experiment_id"]
                    mouse_id = payload["mouse_id"]

                    current_exp_thread = threading.Thread(target=execute_exp_in_thread, args=(exp_name, experiment_id, mouse_id), daemon=True).start()

                    # teensy_board, current_exp = execute_exp(exp_name, experiment_id, mouse_id)

                elif cmd == "STOP":
                    print("\nReceived STOP command.")
                    if experiment_running and current_exp:
                        try:
                            current_exp_thread.stop_data_acquisition()
                            if teensy_board:
                                teensy_board.stop_teensy()
                            experiment_running = False
                            print("\nExperiment stopped.")
                        except Exception as e:
                            print(f"\nError: {e}")
                    else:
                        print("\nNo active experiment.")

                else:
                    print(f"\nUnknown command: {cmd}.")
    

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