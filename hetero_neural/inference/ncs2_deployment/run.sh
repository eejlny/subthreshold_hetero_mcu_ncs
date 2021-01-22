source /hdd2/programs/intel/openvino_2020.1.023/bin/setupvars.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./lib


##./main -i ./eight_17.jpg -d CPU -m /hdd2/programs/Vitis-AI/tensorflow/keras_mnist/ncs/frozen_graph.xml


##./main -i ./eight_17.jpg -d CPU -m /hdd2/programs/Vitis-AI/tensorflow/keras_mnist_tf113/ncs/frozen_graph.xml

##./main  /hdd2/programs/Vitis-AI/tensorflow/keras_mnist_tf113/ncs/frozen_graph.xml ./eight_17.jpg CPU

./main  \
 /hdd2/programs/Vitis-AI/tensorflow/deepconvlstm-activity-detection-tf113/MachineLearningDemos/motion_detector/ncs/first_model.xml \
 /hdd2/programs/Vitis-AI/tensorflow/deepconvlstm-activity-detection-tf113/MachineLearningDemos/motion_detector/ncs/second_model.xml \
 ./data2/body_acc_x_test.txt ./data2/body_acc_y_test.txt ./data2/body_acc_z_test.txt \
 ./data2/body_gyro_x_test.txt ./data2/body_gyro_y_test.txt ./data2/body_gyro_z_test.txt \
 ./data2/total_acc_x_test.txt ./data2/total_acc_y_test.txt ./data2/total_acc_z_test.txt \
 CPU CPU

#./main  /hdd2/programs/Vitis-AI/tensorflow/keras_mnist_tf113/ncs/first_model_tpu.xml  /hdd2/programs/Vitis-AI/tensorflow/keras_mnist_tf113/ncs/second_model_tpu.xml ./images_jpeg/nine_383.jpg CPU


##./main -i ./imgs/elephant.jpg -d CPU -m /hdd1/eejlny/projects/openvino/models/keras/frozen_model.xml
