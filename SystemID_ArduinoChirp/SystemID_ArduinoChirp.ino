// ME520 System ID Assignment
// Andrew North & Taylor Ayars - 10/27/2021

const int analogOut = 5; // pin 5 outputs sine sweep or chirp function
const int analogIn = A0;
int in1 = 0;  // in1 variable records input to A0 pin
float out1 = 0; // out1 variable holds chirp output signal

double currentMillis = 0;
double currentTime = 0;
double deltaT = 0;
double previousTime = 0;
const long interval = 5; //seconds


// Defining the Chirp/Sine Sweep constant Variables
const int phi0 = 0; // start phase
const int f0 = 0.1;   // start frequency
const int f1 = 10;   // stop frequency
const int t1 = 9;   // stop time


void setup() {
  Serial.begin(9600);
}

void loop() {
  currentMillis = millis();
  currentTime = currentMillis/1000;  // currentTime in seconds
  deltaT = currentTime - previousTime; // detaT starts at 0 seconds, grows to "interval" time, then resets to 0

  if (deltaT <= interval) {
    out1 = 127*sin(2*PI*((f1-f0)/t1*deltaT+f0)*deltaT)+127; // defines chirp output based on deltaT input
    in1 = analogRead(analogIn); // read A0 pin. analogRead values from 0 to 1023
    analogWrite(analogOut, out1);  // analogWrite values from 0 to 255
  } else {
    previousTime = currentTime;  // reset chirp output
    }
//  Serial.print("Time: ");
  Serial.print(deltaT);
  Serial.print(",");
//  Serial.print("out1: ");
  Serial.print(out1);
  Serial.print(",");
//  Serial.print("A0 Input:" );
  Serial.print(in1);
  Serial.println();

}
