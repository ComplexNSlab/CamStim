import subprocess

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

def capture_sigrok_data(sigrok_exe, output_file, duration_seconds, samplerate="1MHz"):

    cmd = [
        sigrok_exe,
        '--driver', 'fx2lafw',
        '--config', f'samplerate={samplerate}',
        '--time', f'{duration_seconds}s',  # Duration in seconds
        '--output-file', f'file={output_file}'
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=duration_seconds + 10  # Add buffer for startup/cleanup
        )
        
        if result.returncode == 0:
            print(f"Capture successful! Data saved to: {output_file}")
            return True
        else:
            print(f"Capture failed with return code: {result.returncode}")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("Capture timed out")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


capture_sigrok_data(sigrok_exe, output_file="x.sr", duration_seconds=10, samplerate="1MHz")