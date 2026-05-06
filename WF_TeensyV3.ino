/*
██╗    ██╗██╗██████╗ ███████╗███████╗██╗███████╗██╗     ██████╗     ██╗███╗   ███╗ █████╗  ██████╗ ██╗███╗   ██╗ ██████╗     ████████╗███████╗███████╗███╗   ██╗███████╗██╗   ██╗
██║    ██║██║██╔══██╗██╔════╝██╔════╝██║██╔════╝██║     ██╔══██╗    ██║████╗ ████║██╔══██╗██╔════╝ ██║████╗  ██║██╔════╝     ╚══██╔══╝██╔════╝██╔════╝████╗  ██║██╔════╝╚██╗ ██╔╝
██║ █╗ ██║██║██║  ██║█████╗  █████╗  ██║█████╗  ██║     ██║  ██║    ██║██╔████╔██║███████║██║  ███╗██║██╔██╗ ██║██║  ███╗       ██║   █████╗  █████╗  ██╔██╗ ██║███████╗ ╚████╔╝ 
██║███╗██║██║██║  ██║██╔══╝  ██╔══╝  ██║██╔══╝  ██║     ██║  ██║    ██║██║╚██╔╝██║██╔══██║██║   ██║██║██║╚██╗██║██║   ██║       ██║   ██╔══╝  ██╔══╝  ██║╚██╗██║╚════██║  ╚██╔╝  
╚███╔███╔╝██║██████╔╝███████╗██║     ██║███████╗███████╗██████╔╝    ██║██║ ╚═╝ ██║██║  ██║╚██████╔╝██║██║ ╚████║╚██████╔╝       ██║   ███████╗███████╗██║ ╚████║███████║   ██║   
 ╚══╝╚══╝ ╚═╝╚═════╝ ╚══════╝╚═╝     ╚═╝╚══════╝╚══════╝╚═════╝     ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝        ╚═╝   ╚══════╝╚══════╝╚═╝  ╚═══╝╚══════╝   ╚═╝   
_________________________________________________________________________________________________________________________________________________________________________________
Teensy-Based Widefield Imaging TTL Controller - coordinates synchronized TTL outs for widefield imaging experiments using the RenAndStimPi system
  * Compatible with Teensy 4.x hardware
  * Updated by JS on 28-05-2025
  * Developed for the RenAndStimPi project.

This code receives UART commands from the WF terminal to set parameters and start the LEDS.
  - LEDS will fire in an alternating fashion based on the chosen FPS .
  - Whether the camera TTL (held high) and/or the timestamping TTLs start via an external TTL or when "X" is passed is configured by the user.
  - Timestamping TTLs should be offset (after each LED TTL) to reduce jitter.
  - The code will send a "final" TTL pulse on pin 7 to represent the end of the session (i.e., upon pressing Q)

  LEDs alternate continuously after configuration using the 'S' UART command.
  TTLs can be optionally emitted to the Task Teensy on either LED 1 only or both LEDs,
  and can start based on UART command or an external TTL trigger.
  The camera TTL can also be started either via command or an external TTL input,
  and is held HIGH for the duration of the task.

UART Command Format (send via serial terminal or script):

    1. "S <fps> <led_pulse_us> <offset_us> <camera_start_mode> <ttls_start_mode> <dual_mode> <Timestamping_TTL_Pulse_us>""

    Parameters:
      <fps>                = Pulse frequency for LED toggling (Hz)
      <led_pulse_us>       = Pulse width for each LED (in microseconds)
      <offset_us>          = Delay (in µs) from LED ON to when TTL is sent to Task Teensy
      <camera_start_mode>  = 0 = Camera TTL starts with external TTL (pin 9)
                             1 = Camera TTL starts when 'X' command is received
      <ttls_start_mode>    = 0 = Task Teensy TTLs start with external TTL (pin 9)
                             1 = Task Teensy TTLs start immediately after 'S'
                             2 = Task Teensy TTLs start when 'X' command is received
      <dual_mode>          = 0 = TTL to Task Teensy fires only on LED 1
                             1 = TTL fires on both LED 1 and LED 2 pulses
      <TTL_Width>          = Pulse width for timestamping TTLs in microseconds (after the offset time)
  
    2. "X" = starts the camera TTL (held high), if not set to an external TTL start. Can also be configured to start timesamping (see above)

    3. "Q" = Stop all TTLs and reset session

Example S for testing: S 50 18000 1000 0 0 1 1000

NOTE: The current version of this code does not use hardware timers, to first see if the Teensy is fast enough as is. Happy to add this functionality later if needed.
*/
#define LED1_PIN 2          // TTL to LED 1
#define LED2_PIN 3          // TTL to LED 2
#define CAMERA_PIN 33       // Camera TTL output (held high until Q is passed)
#define FRAME_TTL_PIN 5     // Timestamp TTLs (i.e., for the Task Teensy or DAQ) - pulsed after the specified offset time
#define END_TTL_PIN 7       // One-time TTL to mark session end time (upon pressing Q)
#define EXT_TRIGGER_PIN 23  // Input pin if you want the camera and the timestamping ttls to start via external ttl instead of uart

// Persistent edge-detection state (for external triggering of the cam/timestamp ttls)
bool ext_trigger_latched = false;
bool last_ext_trigger_state = false;

//Configuration parameters (below are defaults, these are set via the 'S' command)
int F_LED = 20; // LED frequency in Hz
int E_LED = 5000; // Duration of LED pulse in microseconds
int OFFSET = 1000; // Delay from LED ON to TTL to Task Teensy (microseconds)
int camera_start_mode = 0; // 0 = external TTL, 1 = 'X' command
int ttls_start_mode = 0; // 0 = external TTL, 1 = immediate ('S'), 2 = 'X'
int dual_mode = 1; // 0 = TTL on LED1 only, 1 = TTLs on both LEDs
int TTL_WIDTH = 1000;  // V4: Default duration for timestamping TTLs (us)


bool config_received = false; // Indicates if 'S' command has been processed
bool ttl_enabled = false; // Flag for sending Timestamping TTLs
bool camera_enabled = false; // Flag for Camera TTL activation 
bool running = false; // Flag indicating active pulse train

unsigned long last_pulse_time = 0; // Time of the last LED pulse
// unsigned long start_time = 0;       // Time when task began (used for debugging)
bool pulse_led1 = true; // Toggles between LED1 and LED2
void stopWaveforms(); // Forward declaration to prevent "not declared in scope" error

void setup() {
  Serial.begin(115200);

// Configuring outputs
  pinMode(LED1_PIN, OUTPUT);
  pinMode(LED2_PIN, OUTPUT);
  pinMode(CAMERA_PIN, OUTPUT);
  pinMode(FRAME_TTL_PIN, OUTPUT);
  pinMode(END_TTL_PIN, OUTPUT);
  pinMode(EXT_TRIGGER_PIN, INPUT); // For the external TTL input 

// Making sure all pins start low
  digitalWrite(LED1_PIN, LOW);
  digitalWrite(LED2_PIN, LOW);
  digitalWrite(CAMERA_PIN, LOW);
  digitalWrite(FRAME_TTL_PIN, LOW);
  digitalWrite(END_TTL_PIN, LOW);

  Serial.println("WFTeensy ready. Awaiting configuration...");
}

void loop() {
  handleSerial(); // Looking for UART commands

  bool ext_trigger_now = digitalRead(EXT_TRIGGER_PIN); // again for external triggering

  if (!ext_trigger_latched && ext_trigger_now && !last_ext_trigger_state) {
    // = Rising edge detected from external TTL
    ext_trigger_latched = true;
    Serial.println("External TTL detected (rising edge).");

    if (camera_start_mode == 0 && !camera_enabled) {
      camera_enabled = true;
      digitalWrite(CAMERA_PIN, HIGH);
      Serial.println("Camera TTL set HIGH (via external TTL).");
    }

    if (ttls_start_mode == 0 && !ttl_enabled) {
      ttl_enabled = true;
      Serial.println("Timestamping TTLs enabled (via external TTL).");
    }
  }
  last_ext_trigger_state = ext_trigger_now;

  if (running) {
    unsigned long now = millis();
    unsigned long interval = 1000 / F_LED;

    if (now - last_pulse_time >= interval) {
      last_pulse_time = now;
      triggerPulse(pulse_led1);
      pulse_led1 = !pulse_led1;
    }
  }
}

void handleSerial() { // Handles the UART input Commands
  if (!Serial.available()) return;

  String cmd = Serial.readStringUntil('\n');
  cmd.trim();

  if (cmd.startsWith("S")) { // Starts the LED train
    parseConfig(cmd);
    config_received = true;
    running = true;

    Serial.println("Config received. LED pulse trains started.");

    if (ttls_start_mode == 1) {
      ttl_enabled = true;
      Serial.println("Timestamping TTLs Enabled (config set this to start on 'S').");
    }
  }
  else if (cmd == "X") {
    if (!config_received) {
      Serial.println("Error: No configuration loaded yet.");
      return;
    }

    if (camera_start_mode == 1 && !camera_enabled) {
      camera_enabled = true;
      digitalWrite(CAMERA_PIN, HIGH);
      Serial.println("Camera TTL set HIGH (via 'X').");
    }

    if (ttls_start_mode == 2 && !ttl_enabled) {
      ttl_enabled = true;
      Serial.println("Timestamping TTLs enabled via 'X'.");
    }
  }
  else if (cmd == "Q") {
    stopWaveforms();
    Serial.println("Session ended.");
  }
  else {
    Serial.println("Unknown command.");
  }
}

void parseConfig(String cmd) { // This parses the S config string 
  cmd = cmd.substring(1); // removes the 'S'
  cmd.trim();

  int tokens[7]; // since using 6 params in S
  int index = 0;

  while (cmd.length() > 0 && index < 7) {
    int spaceIndex = cmd.indexOf(' ');
    if (spaceIndex == -1) spaceIndex = cmd.length();
    tokens[index++] = cmd.substring(0, spaceIndex).toInt();
    cmd = cmd.substring(spaceIndex + 1);
    cmd.trim();
  }

  F_LED = tokens[0];
  E_LED = tokens[1];
  OFFSET = tokens[2];
  camera_start_mode = tokens[3];
  ttls_start_mode = tokens[4];
  dual_mode = tokens[5];
  TTL_WIDTH = tokens[6]; // For camera ttl durations

  Serial.print("F_LED: "); Serial.println(F_LED);
  Serial.print("E_LED: "); Serial.println(E_LED);
  Serial.print("OFFSET: "); Serial.println(OFFSET);
  Serial.print("camera_start_mode: "); Serial.println(camera_start_mode);
  Serial.print("ttls_start_mode: "); Serial.println(ttls_start_mode);
  Serial.print("dual_mode: "); Serial.println(dual_mode);
  Serial.print("TTL_WIDTH: "); Serial.println(TTL_WIDTH);
}

void triggerPulse(bool use_led1) { //Sends one LED pulse and TTL if configured
// Resetting both LEDs first to prevent overlap
  digitalWrite(LED1_PIN, LOW);
  digitalWrite(LED2_PIN, LOW);

// Selecting LED to pulse this frame
  int led_pin = use_led1 ? LED1_PIN : LED2_PIN;
  digitalWrite(led_pin, HIGH); // Turns on selected TTL

  bool should_fire_ttl = false; // Decides whether Timestamping TTL should be fired

  if (ttl_enabled) {
    if (dual_mode == 1 || (dual_mode == 0 && use_led1)) {
      should_fire_ttl = true;
    }
  }

  if (should_fire_ttl) {
    int ttl_pulse_width = TTL_WIDTH;

    delayMicroseconds(OFFSET);
    digitalWrite(FRAME_TTL_PIN, HIGH);
    delayMicroseconds(ttl_pulse_width);
    digitalWrite(FRAME_TTL_PIN, LOW);
    delayMicroseconds(E_LED-ttl_pulse_width-OFFSET);
    //Serial.println("TTL to Task Teensy fired."); // debugging 
  } else {
    delayMicroseconds(E_LED);
  }

  digitalWrite(led_pin, LOW); // Turns off LED after 1 pulse
}

void stopWaveforms() { // Ends the session & turns of all outputs
  running = false;
  ttl_enabled = false;
  camera_enabled = false;

  digitalWrite(LED1_PIN, LOW);
  digitalWrite(LED2_PIN, LOW);
  digitalWrite(CAMERA_PIN, LOW);
  digitalWrite(FRAME_TTL_PIN, LOW);

  digitalWrite(END_TTL_PIN, HIGH); // Pulses the final End TTL for 1ms
  delayMicroseconds(1000);
  digitalWrite(END_TTL_PIN, LOW);

// Allows external triggering to work if another S is passed without resetting the script
  ext_trigger_latched = false;
  last_ext_trigger_state = false; 
}
