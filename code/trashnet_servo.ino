#include <Servo.h>

const int SERVO_PIN = 9;
Servo sortingServo;

String categoryNames[6] = {"cardboard", "glass", "metal", "paper", "plastic", "trash"};
int servoAngles[6] = {0, 36, 72, 108, 144, 180};

int currentAngle = 90;
int targetAngle = 90;

void setup() {
  Serial.begin(9600);
  sortingServo.attach(SERVO_PIN);
  sortingServo.write(90);
  currentAngle = 90;
  
  Serial.println();
  Serial.println("========================================");
  Serial.println("TrashNet Arduino Servo Controller");
  Serial.println("========================================");
  Serial.println("Status: Ready");
  Serial.println("Waiting for commands from Python...");
  Serial.println();
  Serial.println("Angle mapping:");
  for (int i = 0; i < 6; i++) {
    Serial.print("  ");
    Serial.print(categoryNames[i]);
    Serial.print(" -> ");
    Serial.print(servoAngles[i]);
    Serial.println(" degrees");
  }
  Serial.println();
  Serial.println("Format: Send angle (0-180) as integer");
  Serial.println("========================================");
  Serial.println();
}

void loop() {
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');
    data.trim();
    targetAngle = data.toInt();
    
    if (targetAngle >= 0 && targetAngle <= 180) {
      Serial.print("[");
      Serial.print(millis());
      Serial.print("ms] Received: ");
      Serial.print(targetAngle);
      Serial.print(" degrees");
      
      bool foundCategory = false;
      for (int i = 0; i < 6; i++) {
        if (targetAngle == servoAngles[i]) {
          Serial.print(" -> ");
          Serial.print(categoryNames[i]);
          foundCategory = true;
          break;
        }
      }
      if (!foundCategory) {
        Serial.print(" (custom angle)");
      }
      Serial.println();
      
      moveServoSmooth(currentAngle, targetAngle);
      currentAngle = targetAngle;
      
      Serial.print("  -> Servo moved to: ");
      Serial.print(currentAngle);
      Serial.println(" degrees");
      Serial.println();
    } else {
      Serial.print("ERROR: Invalid angle: ");
      Serial.print(targetAngle);
      Serial.println(" degrees");
      Serial.println("  Valid range: 0-180 degrees");
      Serial.println();
    }
  }
}

void moveServoSmooth(int startAngle, int endAngle) {
  int step = (startAngle < endAngle) ? 1 : -1;
  for (int angle = startAngle; angle != endAngle; angle += step) {
    sortingServo.write(angle);
    delay(15);
  }
  sortingServo.write(endAngle);
  delay(100);
}
