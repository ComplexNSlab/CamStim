import subprocess
import sys
from datetime import datetime
import threading
from pathlib import Path
import time
import os
import zipfile
import yaml

stop_flag = False


def load_sigrok_exe():
    config_path = Path(__file__).resolve().parent / 'config.yaml'
    if not config_path.is_file():
        raise Exception(f"Config file not found: {config_path}")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file) or {}

    sigrok_exe = config.get('SIGROK_EXE')
    if not sigrok_exe:
        raise Exception(f"SIGROK_EXE missing in {config_path}")

    return sigrok_exe


def load_save_root():
    config_path = Path(__file__).resolve().parent / 'config.yaml'
    if not config_path.is_file():
        raise Exception(f"Config file not found: {config_path}")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file) or {}

    save_root = config.get('SAVE_DIR')
    if not save_root:
        raise Exception(f"SAVE_DIR missing in {config_path}")

    return save_root

def capture_sigrok_data_chunked(sigrok_exe, output_file, samplerate="1MHz", chunk_samples=1000000):
    """
    Capture data in chunks and append to the .sr file.
    This bypasses the internal buffer limitation.
    """
    chunk_num = 1
    all_chunks = []
    
    while not stop_flag:
        chunk_file = output_file.parent / f"{output_file.stem}_chunk{chunk_num:001d}.sr"
        all_chunks.append(chunk_file)
        
        cmd = [
            sigrok_exe,
            '--driver', 'fx2lafw',
            '--config', f'samplerate={samplerate}',
            '--samples', str(chunk_samples),
            f'--output-file={chunk_file}',
        ]
        
        print(f"\nCapturing chunk {chunk_num}...")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Monitor the process while it's running, but don't block indefinitely
        while process.poll() is None and not stop_flag:
            time.sleep(0.1)  # Small sleep to prevent CPU spinning
        
        # If stop_flag was set, terminate the current chunk
        if stop_flag and process.poll() is None:
            print(f"\nStopping chunk {chunk_num} early...")
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        
        # If the chunk completed normally (not due to stop_flag), increment chunk counter
        if not stop_flag:
            chunk_num += 1
    
    print(f"\nStop flag detected, combining {len(all_chunks)} chunks...")
    
    if len(all_chunks) > 1:
        if combine_sr_files(all_chunks, output_file):
            # Clean up chunks only if combination was successful
            print("Cleaning up chunk files...")
            for chunk in all_chunks:
                if chunk.exists():
                    chunk.unlink()
                    print(f"  Removed: {chunk.name}")
        else:
            print("\nCombination failed, keeping individual chunk files.")
        
    elif len(all_chunks) == 1:
        print(f"Single chunk captured, moving to {output_file}.")
        all_chunks[0].rename(output_file)

    return all_chunks


def listen_for_stop(process_holder):
    """Listen for STOP command and set flag"""
    global stop_flag
    for line in sys.stdin:
        if line.strip().upper() == "STOP":
            print("\nLogic analyzer received STOP command.", flush=True)
            stop_flag = True
            break

from natsort import natsorted

def combine_sr_files(chunk_files, output_file):
    """
    Manually combine .sr files by extracting and concatenating their contents.
    Handles logic-1-1, logic-1-2, etc. as chronological segments.
    """
    print(f"\nManually combining {len(chunk_files)} chunks...")
    
    try:
        # First, extract and sort all segment files chronologically
        all_segments = []  # Will hold (chunk_index, segment_filename, data)
        
        for chunk_idx, chunk_file in enumerate(chunk_files):
            print(f"  Processing chunk {chunk_idx+1}/{len(chunk_files)}: {chunk_file.name}")
            
            with zipfile.ZipFile(chunk_file, 'r') as zf:
                # Get all logic files and sort them for this chunk
                logic_files = [f for f in zf.namelist() 
                             if f.startswith('logic-') and (f.endswith('.bin') or '.' not in f)]
                
                # Use natsort to ensure correct segment order within this chunk
                for filename in natsorted(logic_files):
                    with zf.open(filename) as f:
                        data = f.read()
                        all_segments.append((chunk_idx, filename, data))
                        print(f"    Found segment: {filename} ({len(data)} bytes)")
        
        if not all_segments:
            print("No segment files found in chunks")
            return False
        
        # Now sort all segments by chunk index (keeping the natural order within each chunk)
        all_segments.sort(key=lambda x: x[0])
        
        # Create the combined .sr file
        print(f"\n  Creating combined file: {output_file.name}")
        with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add metadata from first chunk
            with zipfile.ZipFile(chunk_files[0], 'r') as first_chunk:
                for filename in first_chunk.namelist():
                    if not filename.startswith('logic-'):
                        with first_chunk.open(filename) as f:
                            zf.writestr(filename, f.read())
                            print(f"    Added metadata: {filename}")
            
            # Group segments by their base name (e.g., "logic-1" for "logic-1-1", "logic-1-2")
            segments_by_base = {}
            for chunk_idx, filename, data in all_segments:
                base_name = '-'.join(filename.split('-')[:-1])  # "logic-1" from "logic-1-1"
                if base_name not in segments_by_base:
                    segments_by_base[base_name] = []
                segments_by_base[base_name].append(data)
            
            # For each base name, concatenate segments in the order they appear (already correct)
            for base_name, segment_data_list in segments_by_base.items():
                combined_data = b''.join(segment_data_list)
                # Use the base name as the output filename (e.g., "logic-1")
                zf.writestr(base_name, combined_data)
                total_mb = len(combined_data) / (1024*1024)
                print(f"    Added combined data: {base_name} ({len(segment_data_list)} segments, {total_mb:.2f} MB)")
        
        # Verify
        if output_file.exists():
            size_mb = output_file.stat().st_size / (1024*1024)
            print(f"\nSuccessfully created {output_file.name} ({size_mb:.2f} MB)")
            return True
        else:
            print("\nFailed to create output file")
            return False
            
    except Exception as e:
        print(f"\nError during manual combination: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    global stop_flag
    
    # Get experiment parameters
    experiment_id, mouse_id = sys.argv[1], sys.argv[2]
    
    # Set up save directory from config.yaml
    save_root = load_save_root()
    save_dir = Path(save_root) / mouse_id / experiment_id
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Output file
    base_filename = f"{mouse_id}_{experiment_id}"
    output_file = save_dir / f"{base_filename}.sr"
    
    # Handle existing files
    counter = 1
    while output_file.exists():
        output_file = save_dir / f"{base_filename}_{counter}.sr"
        counter += 1
    
    samplerate = "20kHz"  
    chunk_samples = 1000000  # 1M samples per chunk (under 2.4M limit)
    
    print(f"\n{'='*60}")
    print(f"LOGIC ANALYZER RECORDING")
    print(f"{'='*60}")
    print(f"Experiment: {experiment_id}")
    print(f"Mouse: {mouse_id}")
    print(f"Samplerate: {samplerate}")
    print(f"Chunk size: {chunk_samples} samples")
    print(f"Output file: {output_file}")
    print(f"\n{'='*60}")
    
    # Start stop listener
    stop_thread = threading.Thread(target=listen_for_stop, args=(None,), daemon=True)
    stop_thread.start()
    
    # Capture in chunks
    sigrok_exe = load_sigrok_exe()
    chunks = capture_sigrok_data_chunked(sigrok_exe, output_file, samplerate, chunk_samples)
    
    print(f"\n\n{'='*60}")
    print(f"Recording complete!")
    print(f"Captured {len(chunks)} chunks.")
    print(f"File saved: {output_file}.")
    print(f"\n{'='*60}")


if __name__ == '__main__':
    main()