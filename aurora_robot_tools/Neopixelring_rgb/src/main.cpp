
/*
This program is a simple example of using the Adafruit NeoPixel library to control a strip of NeoPixels.by changeing the color of the strip based on serial input. The program defines several states 
(OFF, RED, GREEN, BLUE, WHITE, PARTY) and changes the color of the strip based on the current state. The program also includes a PartY mode that cycles through different colors and a rainbow effect.
the folowing serial inputs will be acepted:
0 - OFF
1 - RED 
2 - GREEN
3 - BLUE
4 - WHITE
5 - PARTY
6 - QR_mode (to turn on a single LED for QR code)
7 - settings (to change the waiting time for the color wipe function)

it will alwys run to one cycle of the party mode and then  continiue from begining. 
*/
// ********************************************************************************includes*************************************************************** 
#include <Arduino.h>
#include <Adafruit_NeoPixel.h>

// *********************************************************************************Defines************************************************************
#define PIXEL_PIN    20 // Digital IO pin connected to the NeoPixels.
#define PIXEL_COUNT 24  // Number of NeoPixels
#define PIXEL_PIN_side 27
#define PIXEL_COUNT_side 8
typedef enum // Define the states of the NeoPixel strip
{
  initialisation, // Initialisation state
  OFF,           // Off state
  idl,           // Idle state
  RED,           // Red state
  GREEN,         // Green state
  BLUE,          // Blue state
  WHITE,         // White state
  PARTY,         // Party mode state
  QR_mode,   // lighting for QR code
  settings       // Settings state for changing waiting time
}
state_TE;

int w_time = 10; // wait time for the color wipe function

// Declare our NeoPixel strip object:
Adafruit_NeoPixel strip(PIXEL_COUNT, PIXEL_PIN, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel strip_side(PIXEL_COUNT_side, PIXEL_PIN_side, NEO_GRB + NEO_KHZ800);

//***********************************************************************************Function Prototypes************************************************************************************
void rainbow(int); // Function to create a rainbow effect on the NeoPixel strip
void colorWipe(uint32_t, int); // Function to change the color of the strip with a wipe effect
void colorWipe_side(uint32_t color, int wait) ; // Function to change the color of the strip_side with a wipe effect
int Read_Serial_Input(int); // Function to read serial input and return the input value
state_TE get_new_state(state_TE, int); // Function to determine the next state based on the current state and input
int change_waiting_time(); // Function to change the waiting time based on user input
void turn_on_LED_for_QR(); // Function to turn on a single LED

//***********************************************************************************Setup************************************************************************************
void setup() 
{
  // Initialize the NeoPixel strip
  strip.begin(); // Initialize NeoPixel strip object (REQUIRED)
  strip.show();  // Initialize all pixels to 'off'
  strip_side.begin(); // Initialize NeoPixel strip object (REQUIRED)
  strip_side.show();  // Initialize all pixels to 'off'
 // Initialize Serial communication
  Serial.begin(9600);
  Serial.setTimeout(10);
}

//***********************************************************************************Main Code************************************************************************************
void loop() 
{
  static state_TE currentState = initialisation;
  switch (currentState) // Check the current state
  {
    case initialisation: // If the state is init
      colorWipe(strip.Color(255, 0, 0), 10);// Turn on the strip to red
      delay(500); // Delay for 1 second
      colorWipe(strip.Color(0, 255, 0), 10); // Turn on the strip to green
      delay(500); // Delay for 1 second
      colorWipe(strip.Color(0, 0, 255), 10); // Turn on the strip to blue
      delay(100); // Delay for 1 second
      colorWipe(strip.Color(255, 255, 255), 10); // Turn on the strip to white
      delay(100); // Delay for 1 second
      colorWipe(strip.Color(0, 0, 0), 0); // Turn off the strip
      colorWipe_side(strip_side.Color(0, 0, 0), 10); // Turn off the strip_side
      delay(100); // Delay for 1 second
      currentState = idl; // Change the state to idle
      break;
    case idl: // If the state is idle
      //input = Read_Serial_Input(input); // Read the serial input and change the state accordingly
      currentState = get_new_state(currentState, Read_Serial_Input(99)); // Get the new state from the input
      break;
    
    case OFF: // If the state is OFF
      colorWipe(strip.Color(0, 0, 0),w_time); // Turn off the strip
      colorWipe_side(strip_side.Color(0, 0, 0),w_time); // Turn off the strip_side
      currentState = idl; // Change the state to idle
      break;
    case RED: // If the state is RED
      colorWipe(strip.Color(255, 0, 0), w_time); // Turn on the strip to red
      currentState = get_new_state(currentState, Read_Serial_Input(1)); // Get the new state from the input
      break;
    case GREEN: // If the state is GREEN
      colorWipe(strip.Color(0, 255, 0), w_time); // Turn on the strip to green
      currentState = get_new_state(currentState, Read_Serial_Input(2)); // Get the new state from the input
      break;
    case BLUE: // If the state is BLUE
      colorWipe(strip.Color(0, 0, 255), w_time); // Turn on the strip to blue
      currentState = get_new_state(currentState, Read_Serial_Input(3)); // Get the new state from the input
      break;
    case WHITE: // If the state is WHITE
      colorWipe(strip.Color(255, 255, 255), w_time); // Turn on the strip to white
      currentState = get_new_state(currentState, Read_Serial_Input(4)); // Get the new state from the input
      break;
    case PARTY: // If the state is PARTY
      /*colorWipe(strip.Color(0, 0, 0), 50); // Turn off the strip
      delay(1000); // Delay for 1 second
      colorWipe(strip.Color(255, 0, 0), 50);
      delay(1000); // Delay for 1 second
      colorWipe(strip.Color(0, 255, 0), 50); // Turn on the strip to green
      delay(1000); // Delay for 1 second
      colorWipe(strip.Color(0, 0, 255), 50); // Turn on the strip to blue
      delay(1000); // Delay for 1 second
      colorWipe(strip.Color(255, 255, 255), 50); // Turn on the strip to white
      delay(1000); // Delay for 1 second */
      rainbow(10); // Call the rainbow function
      delay(10); // Delay for 1 second 
      currentState = get_new_state(currentState, Read_Serial_Input(5)); // Get the new state from the input
      break;
    case QR_mode: // If the state is single LED
      //colorWipe_side(strip_side.Color(255, 255, 255), 50); // Turn off the strip
      turn_on_LED_for_QR()  ; // Turn on the strip_side
      currentState = get_new_state(currentState, Read_Serial_Input(6)); // Get the new state from the input
      break;
    case settings: // If the state is settings 
      w_time = change_waiting_time();
      currentState =  idl; // Change the state to idle
      break;
    default:
      currentState = idl; // If no match found, change the state to idle
      break;
  }
}

// *************************************************************************************Function definitions:*****************************************************************************************************************
/*
 * Function: colorWipe
 * Description: This function changes the color of the strip to the specified color
 *              and waits for the specified time before changing to the next color.
 * Parameters:
 *   - color: The color to change the strip to (in RGB format).
 *   - wait: The time to wait before changing to the next color (in milliseconds).
 *************************************************************************************/
void colorWipe(uint32_t color, int wait) 
{
  for(int i=0; i<strip.numPixels(); i++) 
  { // For each pixel in strip...
    strip.setPixelColor(i, color);         //  Set pixel's color (in RAM)
    strip.show();                          //  Update strip to match
    delay(wait);                           //  Pause for a moment
  }
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
void colorWipe_side(uint32_t color, int wait) 
{
  for(int i=0; i<strip_side.numPixels(); i++) 
  { // For each pixel in strip...
    strip_side.setPixelColor(i, color);         //  Set pixel's color (in RAM)
    strip_side.show();                          //  Update strip to match
    delay(wait);                           //  Pause for a moment
  }
}

// Rainbow cycle along whole strip. Pass delay time (in ms) between frames.
void rainbow(int wait) 
{
  // Hue of first pixel runs 1 complete loops through the color wheel.
  // Color wheel has a range of 65536 but it's OK if we roll over, so
  // just count from 0 to 3*65536. Adding 256 to firstPixelHue each time
  // means we'll make 3*65536/256 = 768 passes through this outer loop:
  for(long firstPixelHue = 0; firstPixelHue < 1*65536; firstPixelHue += 256) 
  {
    for(int i=0; i<strip.numPixels(); i++) 
    { // For each pixel in strip...
      // Offset pixel hue by an amount to make one full revolution of the
      // color wheel (range of 65536) along the length of the strip
      // (strip.numPixels() steps):
      int pixelHue = firstPixelHue + (i * 65536L / strip.numPixels());
      // strip.ColorHSV() can take 1 or 3 arguments: a hue (0 to 65535) or
      // optionally add saturation and value (brightness) (each 0 to 255).
      // Here we're using just the single-argument hue variant. The result
      // is passed through strip.gamma32() to provide 'truer' colors
      // before assigning to each pixel:
      strip.setPixelColor(i, strip.gamma32(strip.ColorHSV(pixelHue)));
    }
    strip.show(); // Update strip with new contents
    delay(wait);  // Pause for a moment
  }
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// This function reads the serial input and returns the value
int Read_Serial_Input(int i)  
{
  char string[10] = {0};  // Buffer to store the input string
  // delay(3000);            // Delay to allow the user to input the value
  if (Serial.available() > 0) // Check if there is any input
  {
    Serial.readBytesUntil('\n', string, sizeof(string) - 1);  // Read the input string
   //Serial.println(string); // print the input string for debugging purposes
    sscanf(string, "%d", &i); // Convert the input string to integer

    return i; // Return the integer value
  }
  else
  {
    return i; // Return 0 if there is no input
  }
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
void turn_on_LED_for_QR()
{
  //colorWipe(strip.Color(0, 0, 0), 50); // Turn off the strip
  strip_side.setPixelColor(0, strip_side.Color(255, 255,255)); // 
  strip_side.setPixelColor(1, strip_side.Color(255, 255,255)); // 
  strip_side.setPixelColor(2, strip_side.Color(255, 255,255)); // 
  strip_side.setPixelColor(3, strip_side.Color(255, 255,255)); // 
  strip_side.show();
}
// This function changes the waiting time based on user input
int change_waiting_time()
{
  char string[10] = {0};  // Buffer to store the input string
  bool condition = true; // Condition to check if the input is valid
  int i = 500; // Default waiting time
  while (condition == true) // Wait for the user to input a value
  {
    Serial.println("Please enter the waiting time in ms: "); // Prompt the user to enter a value
    if (Serial.available()>0)
    {
      Serial.readBytesUntil('\n', string, sizeof(string) - 1);  // Read the input string
      Serial.print("The waiting time is set to: "); // Print the waiting time for debugging purposes
      Serial.println(string); // print the input string for debugging purposes
      sscanf(string, "%d", &i); // Convert the input string to integer
      condition = false; // Set the condition to false to exit the loop
    }
    delay(5000); // Delay for 5 seconds to allow the user to input the value
  }
  return i; // Return the waiting time
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// This function gets the new state based on the input case
state_TE get_new_state(state_TE next_state, int input_case) 
{
  switch (input_case) // Check the case and return the new state
  {
    case 0: // If the case is 1
      return next_state = OFF; // Return RED state
      break;
    case 1: // If the case is 2
      return next_state = RED; // Return OFF state
      break;
    case 2: // If the case is 3
      return next_state = GREEN; // Return GREEN state
      break;
    case 3: // If the case is 4
      return next_state = BLUE; // Return BLUE state
      break;
    case 4: // If the case is 4
      return next_state = WHITE; // Return WHITE state
      break;
    case 5: // If the case is 5
      return next_state = PARTY; // Return Party state
      break;
    case 6: // If the case is 6
      return next_state = QR_mode; // Return single LED state
      break;
    case 7: // If the case is 7
      return next_state = settings; // Return settings state
      break;
    default:
      return next_state = idl; // Return OFF state if no match found
      break;
  }
}
//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// user defined color function not working yet. 
// Interupt handling for stopping the party mode or the settings mode.
void interrupthandler()
{
  
}