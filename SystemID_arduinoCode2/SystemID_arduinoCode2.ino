// ME520 System ID
// Andrew North - 10/12/2021
// This program sets pin 13 to alternatly output 5V then 0V changing every second.
// Pin 13 voltage is measured with A0 pin and recorded with the serialmonitor.
//With help from tutorial: https://www.arduino.cc/en/Tutorial/BuiltInExamples/BlinkWithoutDelay

// Next we will try to measure both pin 13 output voltage & A1 input (Capacitor Voltage)

int analogPin0 = 0;  // "OUTPUT"
int analogPin1 = 1;  // <-------NEW  "INPUT"
int val0 = 0;
int val1 = 1;  // <-------NEW   "INPUT"

int voltageState = LOW;

// Using "unsigned long" for variable holding time.
unsigned long previousMillis = 0;

// constants won't change
// const long interval = 1000
long interval = 10000;   // interval at which to change voltage

void setup() {
  Serial.begin(9600);
  pinMode(13, OUTPUT);    // set digital pin 13 as output
}

void loop() {
  // millis() function returns number of milliseconds sense the 
  // board started running the current sketch
  unsigned long currentMillis = millis();
  
  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;

    if (voltageState == LOW) {
      voltageState = HIGH;
      interval = 2000; // 20 seconds
    } else {
      voltageState = LOW;
      interval = 2000;  // 5 seconds
    }

    digitalWrite(13, voltageState);
  }
  
  val0 = analogRead(analogPin0);   // analogRead returns an int (0 to 1023)
  val1 = analogRead(analogPin1);   // read the "INPUT"
  //val1 = analogRead(analogpin1);  // <-------NEW
  Serial.println(val0);
  Serial.print(",");
  Serial.println(val1);  // <-------NEW
}
