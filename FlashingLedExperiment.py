from BaseExperiment import BaseExperiment
import threading
import time
import numpy as np

class FlashingLedExperiment(BaseExperiment):
    def __init__(self, experiment_id, mouse_id, daq, monitor_config_filename, 
                 save_settings_config_filename, exp_config_filename, debug):
        super().__init__(experiment_id, mouse_id, daq, monitor_config_filename,
                        save_settings_config_filename, exp_config_filename, debug)
        
        # Threading events for LED state management
        self.ir_led_active = threading.Event()
        self.experiment_running = threading.Event()
        
        # Pin definitions (USB6001)
        self.IR_ANALOG_PIN = 3  # AI3
        self.BLUE_ANALOG_PIN = 4  # AI4
        self.IR_DIGITAL_PIN = 0  # P0.0
        self.BLUE_DIGITAL_PIN = 1  # P0.1

        # Counter for blue flashes
        self.blue_flash_count = 0

    def load_experiment_config(self):
        # Minimal config needed since parameters are controlled by RPi
        pass

    def ir_led_monitor(self):
        """Thread function to monitor IR LED state"""
        while self.experiment_running.is_set():
            # Check digital input P0.0 for IR LED state
            if self.daq.read_digital_input(self.IR_DIGITAL_PIN):
                if not self.ir_led_active.is_set():
                    self.ir_led_active.set()
                    self.exp_log.log_exp_start(time.time())
                    print("\nIR LED ON - Experiment started")
                    print("Monitoring for blue LED flashes...")
            else:
                if self.ir_led_active.is_set():
                    self.ir_led_active.clear()
                    self.exp_log.log_exp_end(time.time(), self.blue_flash_count)
                    self.experiment_running.clear()
                    print(f"\nIR LED OFF - Experiment ended")
                    print(f"Total blue LED flashes detected: {self.blue_flash_count}")
            time.sleep(0.001)  # Small sleep to prevent CPU hogging

    def blue_led_monitor(self):
        """Thread function to monitor Blue LED flashes"""
        while self.experiment_running.is_set() and self.ir_led_active.is_set():
            if self.daq.read_digital_input(self.BLUE_DIGITAL_PIN):
                self.blue_flash_count += 1
                self.exp_log.log_stimulus( time.time(), self.blue_flash_count, ["led"], "blue_led")
                print(f"Blue LED flash detected ({self.blue_flash_count})", end='\r')
            time.sleep(0.001)  # Small sleep to prevent CPU hogging

    def run_experiment(self):
        """Main experiment execution"""
        # Start DAQ acquisition
        self.acquisition_running = True
        self.experiment_running.set()

        # Start camera triggers
        self.daq.start_cameras()

        # Start LED monitoring threads
        ir_thread = threading.Thread(target=self.ir_led_monitor)
        blue_thread = threading.Thread(target=self.blue_led_monitor)
        
        ir_thread.start()
        blue_thread.start()

        # Wait for experiment to complete (IR LED to go low)
        while self.experiment_running.is_set():
            time.sleep(0.1)

        # Stop cameras
        self.daq.stop_cameras()

        # Wait additional 3 seconds for any remaining signals
        print("\nWaiting 3 seconds for final signals...")
        time.sleep(3)

        # Clean up
        self.acquisition_running = False
        ir_thread.join()
        blue_thread.join()

        # Save the NI data
        if self.daq.ni_log_filename:
            self.daq.save_data(self.daq.ni_log_filename)

        return True  # Experiment completed successfully