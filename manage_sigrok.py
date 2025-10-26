import subprocess
import sys
from datetime import datetime
import threading
from pathlib import Path

stop_flag = None
LA = None

sigrok_exe = 'C:/Program Files/sigrok/sigrok-cli/sigrok-cli.exe' 
save_dir = Path('C:/Data/logicAnalyzer_Recordings').absolute()
save_dir.mkdir(parents=True, exist_ok=True)  

def check_sigrok_environment(sigrok_exe):
    """Check what drivers and devices are available"""
    
    commands = [
        f'"{sigrok_exe}" --list-supported',
        f'"{sigrok_exe}" --scan'
    ]
    
    for cmd in commands:
        print(f"\n=== Running: {cmd} ===")
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
        except Exception as e:
            print(f"Error: {e}")

# check_sigrok_environment()

def capture_sigrok_data(sigrok_exe, output_file, samplerate="1MHz", duration=30):

    cmd = [
        sigrok_exe,
        '--driver', 'fx2lafw',
        '--config', f'samplerate={samplerate}',
        '--time', f'{duration}s',
        f'--output-file={str(output_file)}',
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return process
            
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def listen_for_stop(process):
    global stop_flag
    for line in sys.stdin:
        if line.strip().upper() == "STOP":
            print("Logic analyzer received STOP command.", flush=True)
            stop_flag = True

            if process and process.poll() is None:  # Only if still running
                try:
                    process.terminate()
                    print("Logic analyzer stopped early by user.", flush=True)
                except Exception as e:
                    print(f"Error stopping logic analyzer: {e}")
            else:
                print("Logic analyzer was already finished.", flush=True)
            break


year, month, day = datetime.now().year, datetime.now().month, datetime.now().day
date = f"{year}{month:02d}{day:02d}"
experiment_id, mouse_id = sys.argv[1], sys.argv[2]
output_file = save_dir / f"{date}_{experiment_id}_{mouse_id}.sr"
output_file = output_file.absolute()
samplerate = "1MHz"
duration = 20

process = capture_sigrok_data(sigrok_exe, output_file=output_file, duration=duration, samplerate=samplerate)

stop_thread = threading.Thread(target=listen_for_stop, args=(process,), daemon=True).start()

stdout, stderr = process.communicate()

if stop_flag:
    print(f"Logic recording stopped early by user. Partial data saved to: {output_file.resolve()}")
else:
    print(f"Logic recording completed normally. File saved to: {output_file.resolve()}")

if stdout:
    print("STDOUT:", stdout)
if stderr:
    print("STDERR:", stderr)

print('Sigrok process complete.')