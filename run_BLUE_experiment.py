from BlueDAQ import BlueDAQ
from FlashingLedExperiment import FlashingLedExperiment
from datetime import datetime

def create_experiment_name():
    experiment_id = input("Enter Experiment name: (MXXXX_XXXXX_XX): ")
    mouse_id = input("Enter mouse ID (XXXXX): ")
    return experiment_id, mouse_id

if __name__ == "__main__":
    try:
        # Get experiment and mouse IDs
        experiment_id, mouse_id = create_experiment_name()

        # Initialize DAQ
        daq = BlueDAQ(experiment_id=experiment_id, DEBUG=False)

        # Create and run experiment
        experiment = FlashingLedExperiment(
            experiment_id=experiment_id,
            mouse_id=mouse_id,
            daq=daq,
            monitor_config_filename='monitor_config.yaml',
            save_settings_config_filename='save_settings_config.yaml',
            exp_config_filename='retinotopy_config.yaml',  # Using any config since we don't need specific settings
            debug=False
        )

        print(f"\nStarting LED Flash Experiment for mouse {mouse_id}")
        print("Waiting for IR LED signal to start experiment...")

        # Start acquisition and run experiment
        daq.start_acquisition()
        experiment.run_experiment()
        daq.stop_acquisition()

        print("\nExperiment completed successfully!")

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
        if 'daq' in locals():
            daq.stop_acquisition()
            daq.stop_cameras()
    except Exception as e:
        print(f"\nError during experiment: {str(e)}")
        if 'daq' in locals():
            daq.stop_acquisition()
            daq.stop_cameras()
        raise 