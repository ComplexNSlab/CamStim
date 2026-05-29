import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import serial
import csv
import argparse
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument('--plot', type=int, default=0, help='Enable plotting (1=yes, 0=no)')
parser.add_argument('--file', type=str, default=None, help='Set the file name (default datetime)')
parser.add_argument('--port', type=str, default='/dev/tty.usbserial-A50285BI', help='Port the mpu6050 is connected to')
args = parser.parse_args()

PLOT = args.plot
filename = args.file
port = args.port

ser = serial.Serial(port, 115200, timeout=0)  # ✅ non-blocking
ser.reset_input_buffer()


# --- CREATE NEW CSV FILE ---
if not filename:
    filename = datetime.now().strftime("mpu6050_%Y%m%d_%H%M%S.csv")
csv_file = open(filename, "w", newline="")
csv_writer = csv.writer(csv_file)

# header
csv_writer.writerow(["time", "ax", "ay", "az", "gx", "gy", "gz"])

print(f"Saving to {filename}")
print(f"Plotting: {'ON' if PLOT else 'OFF'}")

if not PLOT:
    try:
        while True:
            line = ser.readline().decode(errors='ignore').strip()
            line = line.replace('\x00', '')

            vals = line.split()

            if len(vals) >= 7:
                try:
                    x, y, z, gx, gy, gz, T = map(int, vals[:7])
                except:
                    continue

                t = datetime.now().timestamp()
                # --- SAVE TO CSV ---
                csv_writer.writerow([t, x, y, z, gx, gy, gz, T])

    except KeyboardInterrupt:
        print("Stopping...")
        csv_file.close()
        ser.close()
else:
    # data buffers
    x_data, y_data, z_data, t_data = [], [], [], []

    MAX_POINTS = 100   # ✅ keep small for speed

    fig, ax = plt.subplots()

    line_x, = ax.plot([], [], label='X')
    line_y, = ax.plot([], [], label='Y')
    line_z, = ax.plot([], [], label='Z')
    line_t, = ax.plot([], [], label='T')

    ax.set_ylim(-20000, 20000)
    ax.set_xlim(0, MAX_POINTS)
    ax.legend()
    ax.grid()

    def update(frame):
        # ✅ read ALL available lines (prevents lag buildup)
        while ser.in_waiting:
            line = ser.readline().decode(errors='ignore').strip()
            line = line.replace('\x00', '')
            vals = line.split()

            if len(vals) == 7:
                try:
                    x, y, z, gx, gy, gz, T = map(int, vals[:7])   # accel only for speed
                except:
                    continue

                # time (Python time in seconds)
                t = datetime.now().timestamp()

                # --- SAVE TO CSV ---
                csv_writer.writerow([t, x, y, z, gx, gy, gz, T])

                x_data.append(x)
                y_data.append(y)
                z_data.append(z)
                t_data.append(T*10000)

        # ✅ trim buffer
        x_data[:] = x_data[-MAX_POINTS:]
        y_data[:] = y_data[-MAX_POINTS:]
        z_data[:] = z_data[-MAX_POINTS:]
        t_data[:] = t_data[-MAX_POINTS:]

        # ✅ update lines WITHOUT clearing plot
        line_x.set_data(range(len(x_data)), x_data)
        line_y.set_data(range(len(y_data)), y_data)
        line_z.set_data(range(len(z_data)), z_data)
        line_t.set_data(range(len(t_data)), t_data)

        return line_x, line_y, line_z, line_t  # ✅ important for efficiency

    ani = FuncAnimation(
        fig,
        update,
        interval=50,                # ✅ ~20 FPS (smooth enough)
        blit=True,                 # ✅ HUGE speed boost
        cache_frame_data=False
    )


    # --- CLEAN EXIT ---
    def on_close(event):
        print("Closing CSV file...")
        csv_file.close()

    fig.canvas.mpl_connect('close_event', on_close)

    plt.show()

