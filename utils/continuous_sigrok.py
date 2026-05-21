import subprocess
import sys
from datetime import datetime
import threading
from pathlib import Path
import time
import os
import signal
import yaml

try:
    import msvcrt
except ImportError:
    msvcrt = None

stop_flag = False


def load_runtime_config():
    config_path = Path(__file__).resolve().parent.parent / 'config_files' / 'config.yaml'
    if not config_path.is_file():
        raise Exception(f"Config file not found: {config_path}")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file) or {}

    save_root = config.get('SAVE_DIR')
    if not save_root:
        raise Exception(f"SAVE_DIR missing in {config_path}")

    sigrok_exe = config.get('SIGROK_EXE')
    if not sigrok_exe:
        raise Exception(f"SIGROK_EXE missing in {config_path}")

    # Resolve sigrok_exe to absolute path if possible
    exe_path = Path(sigrok_exe)
    if not exe_path.is_absolute():
        # Try to resolve relative to repo root or system PATH
        repo_candidate = Path(__file__).resolve().parent.parent / sigrok_exe
        if repo_candidate.exists():
            sigrok_exe = str(repo_candidate)
        else:
            # Fallback: use as-is (system PATH)
            sigrok_exe = str(sigrok_exe)
    else:
        sigrok_exe = str(exe_path)

    return save_root, sigrok_exe


def load_save_root():
    save_root, _ = load_runtime_config()
    return save_root

def stream_reader(pipe, label):
    """Continuously read a text stream line by line and forward to stdout."""
    try:
        for line in pipe:
            sys.stdout.write(f"[{label}] {line}")
            sys.stdout.flush()
    except Exception as e:
        sys.stdout.write(f"[{label}] Reader error: {e}\n")
        sys.stdout.flush()
    finally:
        if hasattr(pipe, 'close'):
            pipe.close()

def listen_for_stop_thread():
    """Thread that continuously checks for STOP command."""
    global stop_flag
    print("\nListening for STOP command (type 'STOP' and press Enter)...")

    # Always listen for STOP on stdin, regardless of platform
    print("(Type STOP and press Enter in this terminal, or send STOP on stdin)")
    buffer = ''
    while not stop_flag:
        # Check msvcrt for keypresses (Windows), but always check stdin non-blocking
        try:
            if msvcrt is not None and msvcrt.kbhit():
                char = msvcrt.getch().decode('utf-8', errors='ignore')
                if char == '\r':  # Enter key
                    if buffer.upper() == 'STOP':
                        print("\n\n✓ STOP command received!", flush=True)
                        stop_flag = True
                        break
                    print(f"\n[Received: {buffer}] - not STOP", flush=True)
                    buffer = ''
                elif char == '\b':  # Backspace
                    buffer = buffer[:-1]
                else:
                    buffer += char
        except Exception:
            pass
        # Non-blocking check for stdin (works on Unix, and on Windows if piped)
        import select
        try:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                line = sys.stdin.readline()
                if line.strip().upper() == 'STOP':
                    print("\n\n✓ STOP command received!", flush=True)
                    stop_flag = True
                    break
        except Exception:
            time.sleep(0.1)
            continue
        time.sleep(0.05)

def capture_sigrok_continuous(sigrok_exe, output_file, samplerate="20kHz"):
    """Start sigrok-cli with proper process handling and streaming output."""
    output_path = str(output_file.resolve())

    cmd = [
        sigrok_exe,
        '--driver', 'fx2lafw',
        '--config', f'samplerate={samplerate}',
        '--continuous',
        '-o', output_path,
        '-O', 'srzip'
    ]
    
    print(f"\nStarting sigrok-cli with command: {' '.join(cmd)}")
    print("="*60)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        universal_newlines=True,
        encoding='utf-8'
    )

    threading.Thread(target=stream_reader, args=(process.stdout, "OUT"), daemon=True).start()
    threading.Thread(target=stream_reader, args=(process.stderr, "ERR"), daemon=True).start()

    return process

def stop_sigrok(process):
    """Stop exactly like the test - just terminate and wait."""
    if process.poll() is not None:
        return

    print(f"\nStopping sigrok-cli...")

    try:
        process.terminate()
        process.wait(timeout=5)
        print("Process terminated.")
    except subprocess.TimeoutExpired:
        print("Terminate timeout, killing...")
        process.kill()
        process.wait()
    except Exception as e:
        print(f"Error stopping process: {e}")

def signal_handler(sig, frame):
    """Handle interrupt signals."""
    global stop_flag
    print(f"\nReceived signal {sig} - stopping capture.")
    stop_flag = True

def main():
    global stop_flag

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if len(sys.argv) < 3:
        print("Usage: python continuous_sigrok.py <experiment_id> <mouse_id> [base_filename]")
        sys.exit(1)
    
    experiment_id, mouse_id = sys.argv[1], sys.argv[2]
    forced_base_filename = sys.argv[3] if len(sys.argv) >= 4 else None
    
    save_root, sigrok_exe = load_runtime_config()
    save_dir = Path(save_root) / mouse_id / experiment_id
    save_dir.mkdir(parents=True, exist_ok=True)

    if forced_base_filename:
        base_filename = forced_base_filename
    else:
        base_filename = f"{mouse_id}_{experiment_id}"

    # --- Output file collision handling ---
    candidate = base_filename
    suffix = 1
    while True:
        output_file = save_dir / f"{candidate}.sr"
        if not output_file.exists():
            break
        candidate = f"{base_filename}_{suffix}"
        suffix += 1
    if candidate != base_filename:
        print(f"Warning: .sr file exists for '{base_filename}'. Using '{candidate}' instead.")
    # --- End collision handling ---

    samplerate = "20kHz"

    print(f"\n{'-'*60}")
    print(f"LOGIC ANALYZER RECORDING")
    print(f"{'-'*60}")
    print(f"Experiment: {experiment_id}")
    print(f"Mouse: {mouse_id}")
    print(f"Samplerate: {samplerate}")
    print(f"Output file: {output_file}")
    print(f"\n{'-'*60}")
    print("To stop recording:")
    print("  - Type STOP and press Enter")
    print("  - OR press Ctrl+C")
    print(f"{'-'*60}\n")

    stop_flag = False

    # Start listener thread
    stop_thread = threading.Thread(target=listen_for_stop_thread, daemon=True)
    stop_thread.start()

    # Start recording
    start_time = time.time()
    process = capture_sigrok_continuous(sigrok_exe, output_file, samplerate)

    # Wait a moment to ensure capture starts
    time.sleep(2)
    
    # Check initial file status
    if output_file.exists():
        initial_size = output_file.stat().st_size
        print(f"Initial file size: {initial_size} bytes")
    else:
        print("Waiting for file to be created...")

    # Monitor the process
    try:
        while not stop_flag:
            # Check if process is still running
            if process.poll() is not None:
                print(f"\n  Process ended unexpectedly (exit code: {process.returncode})")
                break
            
            time.sleep(0.5)
            
            # Show file size periodically
            if output_file.exists():
                current_size = output_file.stat().st_size
                if current_size > 0 and not hasattr(main, 'shown_size'):
                    print(f"File is growing: {current_size} bytes")
                    main.shown_size = True

        # If we exited due to stop_flag, stop gracefully
        if stop_flag and process.poll() is None:
            print("\nSTOP command detected, stopping capture...")
            stop_sigrok(process)
            
    except Exception as e:
        print(f"\nError: {e}")
        if process.poll() is None:
            stop_sigrok(process)

    elapsed = time.time() - start_time
    
    # Give a moment for final write
    time.sleep(1)

    # Final report
    if output_file.exists():
        size_bytes = output_file.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        
        print(f"\n\n{'-'*60}")
        print(f"RECORDING COMPLETE")
        print(f"{'-'*60}")
        print(f"Duration: {elapsed:.1f} seconds")
        print(f"File saved: {output_file}")
        print(f"File size: {size_mb:.3f} MB ({size_bytes} bytes)")
        
        if size_bytes > 0:
            print("SUCCESS! Data was captured.")
        else:
            print("\n ERROR: File size is 0 bytes!")
            print("No data was captured.")
    else:
        print(f"\n No output file created!")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nReceived Ctrl+C - stopping capture.")
        stop_flag = True
        time.sleep(2)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()