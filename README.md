# subthreshold_hetero_mcu_ncs
 heterogenous neural network with subthreshold mcu and Intel NCS devices

These folders contain support files for paper "Energy-efficient neural networks with near-threshold processors and hardware accelerators".

These instructions assume that you are familiar and have used Tensorflowlite for microcontrollers and the Intel OpenVINO toolset previously.

contents:

hetero_neural -> inference -> mcu_deployment -> ambiq 
folder contains makefile, scriptfiles and c code for tensorflowlite 1.14 for miconcontrollers. 
The TFlite 1.14 should be installed an the script file will compile and generate a binary ready 
for deployment in a sparkfun Ambiq development board.

hetero_neural -> inference -> mcu_deployment -> etacompute
folder contains makefile, scriptfiles and c code for tensorflowlite 1.14 for miconcontrollers. 
The TFlite 1.14 should be installed an the script file will compile and generate a binary ready 
for deployment in a Etacompute ECM3531 development board. 

hetero_neural -> inference -> ncs2_deployment
folder contains makefile, scriptfiles and c code to obtain a binary that will use an Intel NCS2 (Neural Compute Stick)
to accelerate the neural network. It is targetting openvino_2020.1.023 that should be installed and ready to use before trying to generate 
compile.  


hetero_neural -> training -> mnist
folder contains script files and python files to train the mnist classification model for microcontrollers. 
The script mnist_mcu.sh calls two python files that that train the mnist network and write keras model h5 and then frozen graph pb. Then the pb file
is converted to a tflite file with tflite_convert and the tflite to C file with xdd: xxd -i ./tflite/first_model_mcu.tflite > ./tflite/mnist_model.cc

The file mnist_ncs.sh shows and example how the frozen graph pb file can be converted to OpenVino code using the OpenVino model optimizer: 
model_optimizer/mo_tf.py 

hetero_neural -> training -> motion_classification -> data

Training and testing data use during model generation

hetero_neural -> training -> motion_classification -> full_model_ncs

script and python files to train the full model for motion classification and obtain implementation for Intel NCS2 device

hetero_neural -> training -> motion_classification -> two_classes_mcu

script and python files to train the simplified 2 classes model for motion classification and obtain tflite files and C files for mcu implementation. 
