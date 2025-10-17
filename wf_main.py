""" Simple example of digital output

    This example outputs the values of data on line 0 to 7
"""
import sys
import numpy as np
import socket
import json

# sys.stdout.reconfigure(line_buffering=True)

from time import sleep, time
from datetime import datetime

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
from visual_field_mapping import protocolTest


bool_DEBUG = True

# def create_experiment_name():
#     #this if doesn't work 
#     if not bool_DEBUG:
#         experiment_id = input("Enter Experiment name: (MXXXX_XXXXX_XX): ")
#         mouse_id = input("Enter mouse ID (XXXXX): ")
#     else:
#         # when debugging we make a fake tag 
#         # now = datetime.now()
#         # date = str(now).split(" ")[0].replace("-", "")[2:]
#         # experiment_id = "M{}_123411_FB".format(date)
#         # mouse_id = "12345"

#         experiment_id = input("Enter Experiment name: (MXXXX_XXXXX_XX): ")
#         mouse_id = input("Enter mouse ID (XXXXX): ")

#     return experiment_id, mouse_id



def execute_exp(exp_name, experiment_id, mouse_id):
    exp_types = {"Locally Sparse Noise": LocallySparseNoiseExperiment,
            "Dynamic Battery": DynamicBatteryExperiment,
            "Simple Orientation": SimpleOrientationExperiment,
            "Elevation Mapper": ElevationMapperExperiment,
            "Retinotopy": RetinotopyExperiment,
            "Texture FB": TextureExperimentFB,
            "Texture FB-VGG": TextureExperimentFBVGG,
            "Texture FB-VGGMultiTime": TextureExperimentFBVGGMultiTime,
            "Square": SquareExperiment,
            "Visual Field Mapping": protocolTest}

    config_file = f"{exp_name.replace(' ', '_').lower()}_config.yaml"

    exp_type = exp_types.get(exp_name)
    if exp_type is None:
        print("Experiment name error.")
        return

    data_aq = PCODAQ(experiment_id, bool_DEBUG)
    teensy_board = Teensy(experiment_id, bool_DEBUG, "teensyParams.yaml")
    print("Teensy started.")

    exp = exp_type(experiment_id, mouse_id, data_aq, "monitor_config.yaml", "save_settings_config.yaml", config_file, debug=bool_DEBUG)

    exp.load_experiment_config()
    exp.start_data_acquisition()
    teensy_board.start_teensy()
    print("STARTING expriment...")

    exp.run_experiment()
    teensy_board.stop_teensy()
    exp.stop_data_acquisition()
    print("ALL DONE with experiment {}! ".format(experiment_id))


if __name__ == "__main__":
    try:

        if len(sys.argv) > 1:
            exp_name = sys.argv[1]
            experiment_id = sys.argv[2]
            mouse_id = sys.argv[3]
            execute_exp(exp_name, experiment_id, mouse_id)

        else:
            print("Error in receiving experiment inputs. Starting UDP trigger mode.")
            UDP_IP = "127.0.0.1" 
            UDP_PORT = 5005
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind((UDP_IP, UDP_PORT))
            print(f"Waiting for commands on {UDP_IP}:{UDP_PORT}")

            current_exp = None
            teensy_board = None

            while True:
                data, address = sock.recvfrom(1024)
                msg = data.decode()
                print(f"Received message: {msg}")

                try:
                    payload = json.loads(msg)
                except json.JSONDecodeError:
                    print("Invalid message. Requires JSON.")
                    continue

                cmd = payload.get("cmd")

                if cmd == "START":
                    print("Received START command.")
                    exp_name = payload["exp_name"]
                    experiment_id = payload["experiment_id"]
                    mouse_id = payload["mouse_id"]
                    execute_exp(exp_name, experiment_id, mouse_id)

                elif cmd == "STOP":
                    print("Received STOP command.")
                    try:
                        if current_exp and current_exp.experiment_running:
                            current_exp.stop_data_acquisition()
                            teensy_board.stop_teensy()
                            print("Experiment stopped.")
                        else:
                            print("No active experiment.")
                    except Exception as e:
                        print(f"Error: {e}")


        # experiment_id, mouse_id = create_experiment_name()
        # change the line below to use PCO or NIS (2p)
        


    except KeyboardInterrupt:
        print("Received CTRL-C event")
        if exp.experiment_running:
            # We need to run the stopping functions.
            pass
        if exp.acquisition_running:
            exp.stop_data_acquisition()
        try:
            teensy_board.stop_teensy()
        finally:
            print("Teensy stopped.")

    finally:
        print("Experiment routine completed. Thanks!")