#!/usr/bin/env python3
"""MPU serial logger with optional plotting and graceful shutdown support.

Graceful stop methods:
- write "STOP" to stdin
- send SIGINT/SIGTERM
- close plot window (when plotting)
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
import re
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import serial


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', type=int, default=0, help='Enable plotting (1=yes, 0=no)')
    parser.add_argument('--file', type=str, default=None, help='Output csv file path (default timestamped name)')
    parser.add_argument('--port', type=str, default='/dev/tty.usbserial-A50285BI', help='Sensor serial port')
    parser.add_argument('--baud', type=int, default=115200, help='Serial baud rate')
    return parser.parse_args()


def _install_signal_handlers(stop_event: threading.Event) -> None:
    def _handler(signum, _frame):
        stop_event.set()
        print(f"Stopping on signal {signum}...", flush=True)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def _start_stdin_listener(stop_event: threading.Event) -> threading.Thread:
    def _listen() -> None:
        try:
            for line in sys.stdin:
                if line.strip().upper() == 'STOP':
                    print('Received STOP command.', flush=True)
                    stop_event.set()
                    break
        except Exception:
            # Some launch modes may not provide a readable stdin stream.
            pass

    t = threading.Thread(target=_listen, daemon=True)
    t.start()
    return t


def _parse_sensor_line(line: str):
    # Accept whitespace/comma/mixed formatting by extracting signed integers.
    nums = re.findall(r'-?\d+', line.replace('\x00', ''))
    if len(nums) < 7:
        return None
    try:
        x, y, z, gx, gy, gz, t_sensor = map(int, nums[:7])
    except ValueError:
        return None
    return x, y, z, gx, gy, gz, t_sensor


def _logger_loop(
    ser: serial.Serial,
    writer: csv.writer,
    csv_file,
    stop_event: threading.Event,
    sample_buffer: deque,
    buffer_lock: threading.Lock,
) -> None:
    while not stop_event.is_set():
        raw = ser.readline()
        if not raw:
            time.sleep(0.001)
            continue

        parsed = _parse_sensor_line(raw.decode(errors='ignore').strip())
        if parsed is None:
            continue

        x, y, z, gx, gy, gz, t_sensor = parsed
        t = datetime.now().timestamp()
        writer.writerow([t, x, y, z, gx, gy, gz, t_sensor])
        csv_file.flush()
        with buffer_lock:
            sample_buffer.append((x, y, z, t_sensor))

def run_plot(stop_event: threading.Event, sample_buffer: deque, buffer_lock: threading.Lock) -> None:
    import matplotlib

    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    x_data, y_data, z_data, t_data = [], [], [], []
    max_points = 100
    plot_closed_event = threading.Event()

    fig, ax = plt.subplots()

    line_x, = ax.plot([], [], label='X')
    line_y, = ax.plot([], [], label='Y')
    line_z, = ax.plot([], [], label='Z')
    line_t, = ax.plot([], [], label='T')

    ax.set_ylim(-20000, 20000)
    ax.set_xlim(0, max_points)
    ax.legend()
    ax.grid()

    def on_close(_event):
        # Closing the plot should only stop plotting, not logging.
        plot_closed_event.set()

    fig.canvas.mpl_connect('close_event', on_close)
    plt.show(block=False)

    # Manual redraw loop avoids backend-specific animation shutdown freezes.
    while not stop_event.is_set() and not plot_closed_event.is_set() and plt.fignum_exists(fig.number):
        with buffer_lock:
            while sample_buffer:
                x, y, z, t_sensor = sample_buffer.popleft()
                x_data.append(x)
                y_data.append(y)
                z_data.append(z)
                t_data.append(t_sensor * 10000)

        x_data[:] = x_data[-max_points:]
        y_data[:] = y_data[-max_points:]
        z_data[:] = z_data[-max_points:]
        t_data[:] = t_data[-max_points:]

        line_x.set_data(range(len(x_data)), x_data)
        line_y.set_data(range(len(y_data)), y_data)
        line_z.set_data(range(len(z_data)), z_data)
        line_t.set_data(range(len(t_data)), t_data)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.05)

    if plt.fignum_exists(fig.number):
        plt.close(fig)


def main() -> int:
    args = parse_args()
    stop_event = threading.Event()
    _install_signal_handlers(stop_event)
    _start_stdin_listener(stop_event)

    output_file = Path(args.file) if args.file else Path(datetime.now().strftime('mpu6050_%Y%m%d_%H%M%S.csv'))
    output_file.parent.mkdir(parents=True, exist_ok=True)

    ser = serial.Serial(args.port, args.baud, timeout=0)
    ser.reset_input_buffer()

    csv_file = output_file.open('w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['time', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 't'])
    csv_file.flush()

    sample_buffer = deque(maxlen=4096)
    buffer_lock = threading.Lock()

    print(f"Saving to {output_file}", flush=True)
    print(f"Plotting: {'ON' if args.plot else 'OFF'}", flush=True)

    logger_thread = threading.Thread(
        target=_logger_loop,
        args=(ser, writer, csv_file, stop_event, sample_buffer, buffer_lock),
        daemon=True,
    )
    logger_thread.start()

    try:
        if args.plot:
            run_plot(stop_event, sample_buffer, buffer_lock)
            # If the plot window was closed manually, continue logging without UI
            # until STOP/signal is received.
            if not stop_event.is_set():
                print('Plot window closed; continuing headless logging.', flush=True)
                while not stop_event.is_set():
                    time.sleep(0.05)
        else:
            while not stop_event.is_set():
                time.sleep(0.05)
    finally:
        stop_event.set()
        logger_thread.join(timeout=2.0)
        csv_file.close()
        ser.close()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
