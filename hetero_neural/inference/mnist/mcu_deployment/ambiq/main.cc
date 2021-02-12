/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "am_mcu_apollo.h"
#include "am_bsp.h"
#include "am_util.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/examples/micro_mnist/model/mnist_model.h"
#include "tensorflow/lite/micro/examples/micro_mnist/model/mnist_test_data.h"
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

//*****************************************************************************
//
// Init function for the STimer.
//
//*****************************************************************************
void
stimer_init(void)
{
  //
  // Configure the STIMER and run
  //
  am_hal_stimer_config(AM_HAL_STIMER_CFG_CLEAR | AM_HAL_STIMER_CFG_FREEZE);
  am_hal_stimer_config(AM_HAL_STIMER_XTAL_32KHZ);

}


// Helper fn to log the shape and datatype of a tensor
void printTensorDetails(TfLiteTensor* tensor,
                        tflite::ErrorReporter* error_reporter) {
  error_reporter->Report("Type [%s] Shape :", TfLiteTypeGetName(tensor->type));
  for (int d = 0; d < tensor->dims->size; ++d) {
    error_reporter->Report("%d [ %d]", d, tensor->dims->data[d]);
  }
  error_reporter->Report("");
}

static int boardSetup(void)
{
    // Set the clock frequency.
    am_hal_clkgen_control(AM_HAL_CLKGEN_CONTROL_SYSCLK_MAX, 0);

    // Set the default cache configuration
    am_hal_cachectrl_config(&am_hal_cachectrl_defaults);
    am_hal_cachectrl_enable();

    // Configure the board for low power operation.
    am_bsp_low_power_init();

    // Initialize the printf interface for ITM/SWO output.
    am_bsp_uart_printf_enable(); // Enable UART - will set debug output to UART
    //am_bsp_itm_printf_enable(); // Redirect debug output to SWO

    // Setup LED's as outputs
    //am_hal_gpio_pinconfig(AM_BSP_GPIO_LED_RED, g_AM_HAL_GPIO_OUTPUT_12);
    //am_hal_gpio_pinconfig(AM_BSP_GPIO_LED_BLUE, g_AM_HAL_GPIO_OUTPUT_12);
    //am_hal_gpio_pinconfig(AM_BSP_GPIO_LED_GREEN, g_AM_HAL_GPIO_OUTPUT_12);
    //am_hal_gpio_pinconfig(AM_BSP_GPIO_LED_YELLOW, g_AM_HAL_GPIO_OUTPUT_12);

    // Set up button 14 as input (has pullup resistor on hardware)
    //am_hal_gpio_pinconfig(AM_BSP_GPIO_14, g_AM_HAL_GPIO_INPUT);

    // Turn on the LEDs
    //am_hal_gpio_output_set(AM_BSP_GPIO_LED_RED);
    //am_hal_gpio_output_set(AM_BSP_GPIO_LED_BLUE);
    //am_hal_gpio_output_set(AM_BSP_GPIO_LED_GREEN);
    //am_hal_gpio_output_set(AM_BSP_GPIO_LED_YELLOW);

    return 0;
}

int main(int argc, char* argv[]) {

  uint32_t                      ui32StartTime, ui32StopTime;
  uint32_t                      ui32BurstModeDelta, ui32NormalModeDelta;
  am_hal_burst_avail_e          eBurstModeAvailable;
  am_hal_burst_mode_e           eBurstMode;

  // Set up logging.
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  error_reporter->Report("Setting up clock\n");

  //am_devices_led_array_init(am_bsp_psLEDs, AM_BSP_NUM_LEDS);
  //am_devices_led_array_out(am_bsp_psLEDs, AM_BSP_NUM_LEDS, 0x00000000);

  //am_devices_led_off(am_bsp_psLEDs, AM_BSP_GPIO_LED_RED);
  //am_devices_led_off(am_bsp_psLEDs, AM_BSP_GPIO_LED_YELLOW);
  //am_devices_led_off(am_bsp_psLEDs, AM_BSP_GPIO_LED_GREEN);


  //am_devices_led_on(am_bsp_psLEDs, AM_BSP_GPIO_LED_RED);
  //am_devices_led_on(am_bsp_psLEDs, AM_BSP_GPIO_LED_YELLOW);
  //am_devices_led_on(am_bsp_psLEDs, AM_BSP_GPIO_LED_GREEN);

    // Set the clock frequency.
   // am_hal_clkgen_control(AM_HAL_CLKGEN_CONTROL_SYSCLK_MAX, 0);


    // Set the default cache configuration
    //am_hal_cachectrl_config(&am_hal_cachectrl_defaults);
    //am_hal_cachectrl_enable();

    // Configure the board for low power operation.
    //am_bsp_low_power_init();


    boardSetup();

    am_util_stdio_terminal_clear();

    am_util_stdio_printf("SparkFun Edge Project Template\n");
    am_util_stdio_printf("Compiled on %s, %s\n\n", __DATE__, __TIME__);
    am_util_stdio_printf("SparkFun Edge Debug Output (UART)\r\n");
    am_bsp_uart_string_print("Hello, World!\r\n");  // Sting_print has less overhead than printf (and less risky behavior since no varargs)
    am_bsp_uart_string_print("This project is meant to serve as a template for making your own apps as a makefile project\r\n");  

    uint32_t pin14Val = 0; // Default to 0 to illustrate pull-up on hardware
    am_hal_gpio_state_read( AM_BSP_GPIO_14, AM_HAL_GPIO_INPUT_READ, &pin14Val);
    am_util_stdio_printf("Value on button 14 is: %d\r\n", pin14Val);

    // Disable debug
    //am_bsp_debug_printf_disable();
    
    // Go to Deep Sleep.
    //am_hal_sysctrl_sleep(AM_HAL_SYSCTRL_SLEEP_DEEP);



    // Check that the Burst Feature is available.
    if (AM_HAL_STATUS_SUCCESS == am_hal_burst_mode_initialize(&eBurstModeAvailable))
    {
        if (AM_HAL_BURST_AVAIL == eBurstModeAvailable)
        {
            am_util_stdio_printf("Apollo3 Burst Mode is Available\n");
        }
        else
        {
            am_util_stdio_printf("Apollo3 Burst Mode is Not Available\n");
        }
    }
    else
    {
        am_util_stdio_printf("Failed to Initialize for Burst Mode operation\n");
    }

    // Put the MCU into "Burst" mode.
    if (AM_HAL_STATUS_SUCCESS == am_hal_burst_mode_enable(&eBurstMode))
    {
        if (AM_HAL_BURST_MODE == eBurstMode)
        {
            am_util_stdio_printf("Apollo3 operating in Burst Mode (96MHz)\n");
        }
    }
    else
    {
       am_util_stdio_printf("Failed to Enable Burst Mode operation\n");
    }


    // Make sure we are in "Normal" mode.
    if (AM_HAL_STATUS_SUCCESS == am_hal_burst_mode_disable(&eBurstMode))
    {
        if (AM_HAL_NORMAL_MODE == eBurstMode)
        {
           am_util_stdio_printf("Apollo3 operating in Normal Mode (48MHz)\n");
        }
    }
    else
    {
       am_util_stdio_printf("Failed to Disable Burst Mode operation\n");
    }

  
    //am_hal_clkgen_control(AM_HAL_CLKGEN_CONTROL_SYSCLK, 0); //use this to get 48 MHz
    am_hal_clkgen_control(AM_HAL_CLKGEN_CONTROL_SYSCLK_DIV2, 0); //use this to get 24 MHz

  //setup timer

  // Initialize the STimer.
  stimer_init();


  // Execute the example algorithm.
  am_util_stdio_printf("Started tensorflow\n");


  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  error_reporter->Report(
      "Parsing MNIST classifier model FlatBuffer, size %d bytes.",
      mnist_dense_model_tflite_len);
  const tflite::Model* model = ::tflite::GetModel(mnist_dense_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        model->version(), TFLITE_SCHEMA_VERSION);
    return 1;
  }

  // This pulls in all the operation implementations we need.
  tflite::ops::micro::AllOpsResolver resolver;

  // Create an area of memory to use for input, output, & intermediate arrays.
  // The size of this will depend on the model you're using, currently
  // determined by experimentation.
  const int tensor_arena_size = 30 * 1024;
  uint8_t tensor_arena[tensor_arena_size];

  // Build an interpreter to run the model with.
  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                       tensor_arena_size, error_reporter);
  interpreter.AllocateTensors();

  // Get information about the models input and output tensors.
  TfLiteTensor* model_input = interpreter.input(0);
  error_reporter->Report("Details of input tensor:");
  printTensorDetails(model_input, error_reporter);
  TfLiteTensor* model_output = interpreter.output(0);
  error_reporter->Report("Details of output tensor:");
  printTensorDetails(model_output, error_reporter);

  // perform inference on each test sample and evalute accuracy of model
  int accurateCount = 0;
  const int inputTensorSize = 28 * 28;
  error_reporter->Report("Total number of samples is %d\n",mnistSampleCount);

  for (int s = 0; s < mnistSampleCount; ++s) {
    // Set value of input tensor
    for (int d = 0; d < inputTensorSize; ++d) {
      model_input->data.uint8[d] = mnistInput[s][d];
    }


  // Capture the start time.
  ui32StartTime = am_hal_stimer_counter_get();

  

    // perform inference and repeat 10 times
    for (int r = 0; r < 10; r++) {

    	TfLiteStatus invoke_status = interpreter.Invoke();
    	if (invoke_status != kTfLiteOk) {
      		error_reporter->Report("Invoke failed.\n");
     		 return 1;
    	}
	else {
		error_reporter->Report("Invoke succesful.\n");
	}
    }



  // Stop the timer and calculate the elapsed time.
  ui32StopTime = am_hal_stimer_counter_get();


    error_reporter->Report("Model estimate [%d] [%d] [%d] [%d] [%d] [%d] [%d] [%d] [%d] [%d] ",
                           model_output->data.uint8[0],model_output->data.uint8[1],model_output->data.uint8[2],model_output->data.uint8[3],
                           model_output->data.uint8[4],model_output->data.uint8[5],model_output->data.uint8[6],model_output->data.uint8[7],
			   model_output->data.uint8[8],model_output->data.uint8[9]);

     int v_uint8=0;
     int idx = 0;
     for (int i = 0; i < 10; i++) {
          unsigned int vi = model_output->data.uint8[i];
          if(vi > v_uint8){
             idx = i;
             v_uint8 = vi;
          }
        }

     error_reporter->Report("training label [%d]",mnistOutput[s]);

    //error_reporter->Report("Model estimate [%d] training label [%d]",
    //                       model_output->data.uint8[0], mnistOutput[s]);

    if (idx == mnistOutput[s]) {
      ++accurateCount;
    }
  }

  error_reporter->Report("Test set accuracy was %d percent\n",
                         ((accurateCount * 100) / mnistSampleCount));

  

    // Setup LED's as outputs
   //#ifdef AM_BSP_NUM_LEDS
   //#endif

    // Turn off the LEDs
    am_hal_gpio_output_clear(AM_BSP_GPIO_LED_RED);
    am_hal_gpio_output_clear(AM_BSP_GPIO_LED_BLUE);
    am_hal_gpio_output_clear(AM_BSP_GPIO_LED_GREEN);
    am_hal_gpio_output_clear(AM_BSP_GPIO_LED_YELLOW);

  error_reporter->Report("MNIST classifier example completed successfully.\n"); 

  // Calculate the Burst Mode delta time.
  ui32NormalModeDelta = ui32StopTime - ui32StartTime;
  am_util_stdio_printf("Start : %d \n", (uint32_t)(ui32StartTime));
  am_util_stdio_printf("End : %d \n", (uint32_t)(ui32StopTime));
  am_util_stdio_printf("Delta : %d \n", (uint32_t)(ui32NormalModeDelta));
  am_util_stdio_printf("Execution time is : %d millisec\n", (uint32_t)(1000*ui32NormalModeDelta / 32000));


  //am_devices_led_toggle(am_bsp_psLEDs, AM_BSP_GPIO_LED_BLUE);
  //am_devices_led_toggle(am_bsp_psLEDs, AM_BSP_GPIO_LED_BLUE);
  //am_devices_led_toggle(am_bsp_psLEDs, AM_BSP_GPIO_LED_BLUE);
  //am_devices_led_toggle(am_bsp_psLEDs, AM_BSP_GPIO_LED_BLUE);
  //am_devices_led_toggle(am_bsp_psLEDs, AM_BSP_GPIO_LED_BLUE);
  //am_devices_led_toggle(am_bsp_psLEDs, AM_BSP_GPIO_LED_BLUE);


  am_hal_sysctrl_sleep(AM_HAL_SYSCTRL_SLEEP_DEEP);

  return 0;
}
