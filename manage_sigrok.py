import subprocess
import sys
from datetime import datetime
import threading
from pathlib import Path

stop_flag = None
LA = None

sigrok_exe = 'C:/Program Files/sigrok/sigrok-cli/sigrok-cli.exe'  

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
    
    print(f"\nRunning command: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return process
            
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return None


def listen_for_stop(process):
    global stop_flag
    for line in sys.stdin:
        if line.strip().upper() == "STOP":
            print("\nLogic analyzer received STOP command.", flush=True)
            stop_flag = True

            if process and process.poll() is None:  # Only if still running
                try:
                    process.terminate()
                    print("\nLogic analyzer stopped early by user.", flush=True)
                except Exception as e:
                    print(f"\nError stopping logic analyzer: {e}")
            else:
                print("\nLogic analyzer was already finished.", flush=True)
            break


year, month, day = datetime.now().year, datetime.now().month, datetime.now().day
date = f"{year}{month:02d}{day:02d}"
experiment_id, mouse_id = sys.argv[1], sys.argv[2]
save_dir = Path(f"C:\\\\Data\\{experiment_id}\\logicAnalyzer_Recordings").absolute()
save_dir.mkdir(parents=True, exist_ok=True) 
output_file = save_dir / f"{date}_{experiment_id}_{mouse_id}.sr"
output_file = output_file.absolute()
samplerate = "1MHz"
duration = 20

process = capture_sigrok_data(sigrok_exe, output_file=output_file, duration=duration, samplerate=samplerate)

stop_thread = threading.Thread(target=listen_for_stop, args=(process,), daemon=True).start()

stdout, stderr = process.communicate()

if stop_flag:
    print(f"\nLogic recording stopped early by user. Partial data saved to: {output_file.resolve()}")
else:
    print(f"\nLogic recording completed normally. File saved to: {output_file.resolve()}")

if stdout:
    print("\nSTDOUT:", stdout)
if stderr:
    print("\nSTDERR:", stderr)

print('\nSigrok process complete.')