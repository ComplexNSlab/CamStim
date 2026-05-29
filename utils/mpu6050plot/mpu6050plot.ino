#include <Wire.h>

const int MPU = 0x68; // MPU6050 I2C address

int16_t AccX, AccY, AccZ;
int16_t GyroX, GyroY, GyroZ;

const uint8_t TICK_PIN = 2;
const unsigned long TICK_PERIOD_MS = 1000;
const unsigned long TICK_HIGH_MS = 100;

unsigned long lastPulseStartMs = 0;
unsigned long pulseHighStartMs = 0;
bool pulseActive = false;

void setup() {
  Wire.begin();
  Serial.begin(115200);

  pinMode(TICK_PIN, OUTPUT);
  digitalWrite(TICK_PIN, LOW);

  // Wake up MPU-6050
  Wire.beginTransmission(MPU);
  Wire.write(0x6B);
  Wire.write(0);
  Wire.endTransmission(true);
}

void loop() {
  unsigned long now = millis();

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

  Wire.beginTransmission(MPU);
  Wire.write(0x3B);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU, 14, true);

  AccX = Wire.read() << 8 | Wire.read();
  AccY = Wire.read() << 8 | Wire.read();
  AccZ = Wire.read() << 8 | Wire.read();

  Wire.read(); Wire.read(); // skip temp

  GyroX = Wire.read() << 8 | Wire.read();
  GyroY = Wire.read() << 8 | Wire.read();
  GyroZ = Wire.read() << 8 | Wire.read();

  Serial.print(AccX); Serial.print(" ");
  Serial.print(AccY); Serial.print(" ");
  Serial.print(AccZ); Serial.print(" ");
  Serial.print(GyroX); Serial.print(" ");
  Serial.print(GyroY); Serial.print(" ");
  Serial.print(GyroZ); Serial.print(" ");
  Serial.println(tickValue);

  delay(20);
}
