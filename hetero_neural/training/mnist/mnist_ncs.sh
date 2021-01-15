#python3 mnist.py

python3 split_tpu.py

#input range (0,255) then mean = 0, std_dev = 1
#input range (-1,1) then mean = 127.5, std_dev = 127.5
#input range (0,1) then mean = 0, std_dev = 255

#convert first h5 to pb
python3 /hdd2/programs/Vitis-AI/tensorflow/keras_to_tensorflow/keras_to_tensorflow.py --input_model=./second_model_tpu.h5 --output_model=./second_model_tpu.pb

#convert second h5 to pb
python3 /hdd2/programs/Vitis-AI/tensorflow/keras_to_tensorflow/keras_to_tensorflow.py --input_model=./first_model_tpu.h5 --output_model=./first_model_tpu.pb


python3 /hdd2/programs/intel/openvino_2020.1.023/deployment_tools/model_optimizer/mo_tf.py \
	--input_model ./first_model_tpu.pb \
	--output_dir ./ncs \
	--input_shape "[1,28,28,1]" \
	--data_type FP32

python3 /hdd2/programs/intel/openvino_2020.1.023/deployment_tools/model_optimizer/mo_tf.py \
	--input_model ./second_model_tpu.pb \
	--output_dir ./ncs \
	--input_shape "[1,128]" \
	--data_type FP32


