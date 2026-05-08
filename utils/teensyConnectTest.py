import serial
import time

port = "COM7" 
baud = 115200

ser = serial.Serial(port, baud, timeout=1)
time.sleep(2) 
ser.reset_input_buffer()
ser.reset_output_buffer()

command = "Q\n"
ser.write(command.encode())
print("Sent:", command.strip())
time.sleep(2) 
command = "S 50 18000 1000 0 0 1 1000\n"
ser.write(command.encode())
print("Sent:", command.strip())

print("Waiting for response...")
while True:
    if ser.in_waiting > 0:
        response = ser.readline().decode(errors='ignore').strip()
        print("Teensy:", response)