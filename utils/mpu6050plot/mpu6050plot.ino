#include <Wire.h>

const int MPU = 0x68; // MPU6050 I2C address

int16_t AccX, AccY, AccZ;
int16_t GyroX, GyroY, GyroZ;

unsigned long lastTime = 0;

void setup() {
  Wire.begin();
  Serial.begin(115200);

  // Wake up MPU-6050
  Wire.beginTransmission(MPU);
  Wire.write(0x6B);
  Wire.write(0);
  Wire.endTransmission(true);
}

void loop() {
  
 unsigned long now = millis();
 int tickValue = 0;
 
  if (now - lastTime >= 1000) {  // 1 second
    lastTime = now;

    digitalWrite(2, HIGH);   // pulse pin
    delay(10);
    digitalWrite(2, LOW);
    tickValue = 1;
  }

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
