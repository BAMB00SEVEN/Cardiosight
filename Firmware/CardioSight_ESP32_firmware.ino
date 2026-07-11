/*
  =====================================================================
  CardioSight — ECG Acquisition, Transmission & Display Firmware
  Platform     : ESP32 Dev Board
  Sensor       : AD8232 Single-Lead ECG Front-End (or ADS1292R via SPI)
  Author       : <Your Name>
  College      : <Your College Name>
  =====================================================================

  FUNCTION:
  - Samples raw ECG from AD8232 analog output at a fixed rate (500 Hz)
  - Detects lead-off (electrode disconnect) using AD8232's LO+/LO- pins
  - Buffers ~4 seconds of samples and sends them to the Python ML
    backend (inference_server.py) over WiFi (HTTP POST)
  - Receives back a JSON prediction {label, confidence, heart_rate_bpm}
  - Displays heart rate + diagnostic label on the OLED
  - Drives a status LED (green=normal, red=abnormal) and buzzer alert
    when an arrhythmia / abnormal indication is returned

  LIBRARIES REQUIRED (Arduino IDE > Library Manager):
  - WiFi.h, HTTPClient.h (built-in with ESP32 board package)
  - ArduinoJson (for parsing the prediction response)
  - Adafruit_SSD1306 + Adafruit_GFX (OLED display)
  =====================================================================
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ---------------------- USER CONFIG ---------------------------------
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* SERVER_URL    = "http://<YOUR_PC_IP>:5000/predict"; // inference_server.py --mode wifi

// ---------------------- PIN MAP --------------------------------------
#define ECG_ANALOG_PIN   34    // AD8232 OUTPUT
#define LO_PLUS_PIN      32    // AD8232 LO+ (lead-off detect)
#define LO_MINUS_PIN     33    // AD8232 LO-
#define LED_GREEN_PIN    26
#define LED_RED_PIN      27
#define BUZZER_PIN       25
#define OLED_SDA         21
#define OLED_SCL         22
#define SCREEN_WIDTH     128
#define SCREEN_HEIGHT     64

// ---------------------- SAMPLING CONFIG --------------------------------
const int SAMPLE_RATE_HZ   = 500;
const int BUFFER_SECONDS   = 4;
const int BUFFER_SIZE      = SAMPLE_RATE_HZ * BUFFER_SECONDS;   // 2000 samples
float ecgBuffer[BUFFER_SIZE];
int bufferIndex = 0;
unsigned long lastSampleMicros = 0;
const unsigned long SAMPLE_INTERVAL_US = 1000000UL / SAMPLE_RATE_HZ;

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

String lastLabel = "Waiting...";
float  lastHR    = 0;
float  lastConfidence = 0;

void setup() {
  Serial.begin(115200);

  pinMode(LO_PLUS_PIN, INPUT);
  pinMode(LO_MINUS_PIN, INPUT);
  pinMode(LED_GREEN_PIN, OUTPUT);
  pinMode(LED_RED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);

  Wire.begin(OLED_SDA, OLED_SCL);
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println(F("OLED init failed"));
  }
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println("CardioSight Booting...");
  display.display();

  connectWiFi();
}

void loop() {
  // ---------- 1. Lead-off check ----------
  bool leadOff = (digitalRead(LO_PLUS_PIN) == HIGH) || (digitalRead(LO_MINUS_PIN) == HIGH);

  // ---------- 2. Sample ECG at fixed interval ----------
  unsigned long nowMicros = micros();
  if (!leadOff && (nowMicros - lastSampleMicros >= SAMPLE_INTERVAL_US)) {
    lastSampleMicros = nowMicros;
    int raw = analogRead(ECG_ANALOG_PIN);
    float voltage = (raw / 4095.0) * 3.3;
    ecgBuffer[bufferIndex++] = voltage;

    if (bufferIndex >= BUFFER_SIZE) {
      sendBufferForInference();
      bufferIndex = 0;  // reset buffer for next window
    }
  }

  // ---------- 3. Update display / alerts every loop ----------
  updateDisplay(leadOff);
  handleAlertOutputs();

  delay(2); // small yield; sampling handled via micros() timing above
}

// ---------------------- HELPER FUNCTIONS ------------------------------

void connectWiFi() {
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  Serial.println(WiFi.status() == WL_CONNECTED ? "\nConnected!" : "\nWiFi failed - check credentials.");
}

void sendBufferForInference() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi not connected - skipping inference call.");
    return;
  }

  // Build compact JSON payload: {"samples":[v1,v2,...]}
  String payload = "{\"samples\":[";
  for (int i = 0; i < BUFFER_SIZE; i++) {
    payload += String(ecgBuffer[i], 4);
    if (i < BUFFER_SIZE - 1) payload += ",";
  }
  payload += "]}";

  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");
  int httpCode = http.POST(payload);

  if (httpCode == 200) {
    String response = http.getString();
    StaticJsonDocument<256> doc;
    DeserializationError err = deserializeJson(doc, response);
    if (!err) {
      lastLabel      = doc["label"].as<String>();
      lastHR         = doc["heart_rate_bpm"].as<float>();
      lastConfidence = doc["confidence"].as<float>();
      Serial.printf("Prediction: %s | HR: %.1f bpm | Conf: %.2f\n",
                    lastLabel.c_str(), lastHR, lastConfidence);
    }
  } else {
    Serial.printf("HTTP POST failed, code: %d\n", httpCode);
  }
  http.end();
}

void updateDisplay(bool leadOff) {
  display.clearDisplay();
  display.setCursor(0, 0);
  display.setTextSize(1);

  if (leadOff) {
    display.println("Lead-off detected!");
    display.println("Check electrodes.");
    display.display();
    return;
  }

  display.println("CardioSight");
  display.printf("HR: %.0f bpm\n", lastHR);
  display.printf("Status: %s\n", lastLabel.c_str());
  display.printf("Conf: %.0f%%\n", lastConfidence * 100);
  display.display();
}

void handleAlertOutputs() {
  bool abnormal = (lastLabel != "Normal" && lastLabel != "Waiting...");

  digitalWrite(LED_GREEN_PIN, abnormal ? LOW : HIGH);
  digitalWrite(LED_RED_PIN, abnormal ? HIGH : LOW);
  digitalWrite(BUZZER_PIN, abnormal ? HIGH : LOW);
}
