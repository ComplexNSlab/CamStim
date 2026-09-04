#include <Wire.h>

// Two MPU6050 sensors on the same I2C bus.
#define MPU1_ADDR 0x68
#define MPU2_ADDR 0x69

struct MpuSample {
  int16_t ax;
  int16_t ay;
  int16_t az;
  int16_t gx;
  int16_t gy;
  int16_t gz;
};

MpuSample mpu1;
MpuSample mpu2;

const uint8_t TICK_PIN = 2;
const unsigned long STARTUP_HIGH_MS = 3000;
const unsigned long TICK_PERIOD_MS = 1000;
const unsigned long TICK_HIGH_MS = 100;

unsigned long lastPulseStartMs = 0;
unsigned long pulseHighStartMs = 0;
unsigned long startupHighStartMs = 0;
bool pulseActive = false;
bool recordingEnabled = false;
bool startupHighActive = false;

void handle_command(const char *cmd) {
  if (cmd[0] == 'S') {
    recordingEnabled = true;
    startupHighActive = true;
    startupHighStartMs = millis();
    pulseActive = false;
    lastPulseStartMs = startupHighStartMs;
    pulseHighStartMs = startupHighStartMs;
    digitalWrite(TICK_PIN, HIGH);
  } else if (cmd[0] == 'Q') {
    recordingEnabled = false;
    startupHighActive = false;
    pulseActive = false;
    digitalWrite(TICK_PIN, LOW);
  }
}

void poll_serial_commands() {
  static char commandBuffer[16];
  static uint8_t commandLen = 0;

  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (commandLen > 0) {
        commandBuffer[commandLen] = '\0';
        handle_command(commandBuffer);
        commandLen = 0;
      }
      continue;
    }

    if (commandLen < sizeof(commandBuffer) - 1) {
      commandBuffer[commandLen++] = c;
    }
  }
}

void wake_mpu(uint8_t addr) {
  Wire.beginTransmission(addr);
  Wire.write(0x6B);
  Wire.write(0);
  Wire.endTransmission(true);
}

bool read_mpu(uint8_t addr, MpuSample &out) {
  Wire.beginTransmission(addr);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) {
    return false;
  }

  uint8_t bytes_requested = 14;
  uint8_t bytes_read = Wire.requestFrom((int)addr, (int)bytes_requested, (int)true);
  if (bytes_read < bytes_requested) {
    while (Wire.available()) {
      Wire.read();
    }
    return false;
  }

  out.ax = (Wire.read() << 8) | Wire.read();
  out.ay = (Wire.read() << 8) | Wire.read();
  out.az = (Wire.read() << 8) | Wire.read();

  Wire.read();
  Wire.read();  // Skip temperature

  out.gx = (Wire.read() << 8) | Wire.read();
  out.gy = (Wire.read() << 8) | Wire.read();
  out.gz = (Wire.read() << 8) | Wire.read();
  return true;
}

void setup() {
  Wire.begin();
  Serial.begin(115200);

  pinMode(TICK_PIN, OUTPUT);
  digitalWrite(TICK_PIN, LOW);

  // Wake up both MPU-6050 sensors.
  wake_mpu(MPU1_ADDR);
  wake_mpu(MPU2_ADDR);
}

void loop() {
  poll_serial_commands();

  if (!recordingEnabled) {
    delay(20);
    return;
  }

  unsigned long now = millis();

  if (startupHighActive) {
    if (now - startupHighStartMs < STARTUP_HIGH_MS) {
      bool ok1 = read_mpu(MPU1_ADDR, mpu1);
      bool ok2 = read_mpu(MPU2_ADDR, mpu2);
      if (!ok1 || !ok2) {
        delay(20);
        return;
      }

      Serial.print(mpu1.ax); Serial.print(" ");
      Serial.print(mpu1.ay); Serial.print(" ");
      Serial.print(mpu1.az); Serial.print(" ");
      Serial.print(mpu1.gx); Serial.print(" ");
      Serial.print(mpu1.gy); Serial.print(" ");
      Serial.print(mpu1.gz); Serial.print(" ");
      Serial.print(mpu2.ax); Serial.print(" ");
      Serial.print(mpu2.ay); Serial.print(" ");
      Serial.print(mpu2.az); Serial.print(" ");
      Serial.print(mpu2.gx); Serial.print(" ");
      Serial.print(mpu2.gy); Serial.print(" ");
      Serial.print(mpu2.gz); Serial.print(" ");
      Serial.println(1);

      delay(20);
      return;
    }

    startupHighActive = false;
    digitalWrite(TICK_PIN, LOW);
    lastPulseStartMs = now;
  }

  // Start a new pulse every second (non-blocking).
  if (!pulseActive && (now - lastPulseStartMs >= TICK_PERIOD_MS)) {
    lastPulseStartMs = now;
    pulseHighStartMs = now;
    pulseActive = true;
    digitalWrite(TICK_PIN, HIGH);
  }

  // End pulse after 100 ms, keeping loop free for continuous logging.
  if (pulseActive && (now - pulseHighStartMs >= TICK_HIGH_MS)) {
    pulseActive = false;
    digitalWrite(TICK_PIN, LOW);
  }

  int tickValue = pulseActive ? 1 : 0;

  bool ok1 = read_mpu(MPU1_ADDR, mpu1);
  bool ok2 = read_mpu(MPU2_ADDR, mpu2);
  if (!ok1 || !ok2) {
    delay(20);
    return;
  }

  // Output format (13 ints):
  // ax1 ay1 az1 gx1 gy1 gz1 ax2 ay2 az2 gx2 gy2 gz2 tick
  Serial.print(mpu1.ax); Serial.print(" ");
  Serial.print(mpu1.ay); Serial.print(" ");
  Serial.print(mpu1.az); Serial.print(" ");
  Serial.print(mpu1.gx); Serial.print(" ");
  Serial.print(mpu1.gy); Serial.print(" ");
  Serial.print(mpu1.gz); Serial.print(" ");
  Serial.print(mpu2.ax); Serial.print(" ");
  Serial.print(mpu2.ay); Serial.print(" ");
  Serial.print(mpu2.az); Serial.print(" ");
  Serial.print(mpu2.gx); Serial.print(" ");
  Serial.print(mpu2.gy); Serial.print(" ");
  Serial.print(mpu2.gz); Serial.print(" ");
  Serial.println(tickValue);

  delay(20);
}
