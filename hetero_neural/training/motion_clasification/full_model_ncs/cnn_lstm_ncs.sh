python3  merged_dcl.py

#input range (0,255) then mean = 0, std_dev = 1
#input range (-1,1) then mean = 127.5, std_dev = 127.5
#input range (0,1) then mean = 0, std_dev = 255

#python3 /hdd2/programs/openvino/openvino_2019.1.133/deployment_tools/model_optimizer/mo.py \
#	--input_model ./cnn_lstm_tpu.pb \
#	--output_dir ./ncs \
#	--input_shape "[1,210,1]" \
#	--data_type FP32 \
#	--framework tf

#python3 split_tpu.py

#convert h5 to pb
#convert monolitic model
python3 /hdd2/programs/Vitis-AI/tensorflow/keras_to_tensorflow/keras_to_tensorflow.py --input_model=./data/UCI_HAR_Dataset_output/merged_dcl.h5 --output_model=./data/UCI_HAR_Dataset_output/merged_dcl.pb


#convert first model to pb
#python3 /hdd2/programs/Vitis-AI/tensorflow/keras_to_tensorflow/keras_to_tensorflow.py --input_model=./first_model.h5 --output_model=./data/UCI_HAR_Dataset_output/first_model.pb
#convert second model to pb
#python3 /hdd2/programs/Vitis-AI/tensorflow/keras_to_tensorflow/keras_to_tensorflow.py --input_model=./second_model.h5 --output_model=./data/UCI_HAR_Dataset_output/second_model.pb



#openvino optimize
#python3 /hdd2/programs/intel/openvino_2020.1.023/deployment_tools/model_optimizer/mo.py \
#	--input input_1,input_2,input_3 \
#	--input_model ./data/UCI_HAR_Dataset_output/first_model.pb \
#       --input_shape "[1,128,3],[1,128,3],[1,128,3]" \
#	--output concatenate/concat \
#	--output_dir ./ncs \
#	--data_type FP32 \
#	--framework tf

#python3 /hdd2/programs/intel/openvino_2020.1.023/deployment_tools/model_optimizer/mo.py \
#	--input input_4 \
#	--input_model ./data/UCI_HAR_Dataset_output/second_model.pb \
#        --input_shape "[1,384]" \
#	--output softmax/Softmax \
#	--output_dir ./ncs \
#	--data_type FP32 \
#	--framework tf



#convert h5 to pb
#python3 /hdd2/programs/Vitis-AI/tensorflow/keras_to_tensorflow/keras_to_tensorflow.py --input_model=./second_cnn_lstm_tpu.h5 --output_model=./second_model_ecg.pb

#python3 /hdd2/programs/intel/openvino_2020.1.023/deployment_tools/model_optimizer/mo.py \
#	--input main_input1 \
#	--input_model ./data/UCI_HAR_Dataset_output/merged_dcl.pb \
#       --input_shape "[1,128,3]" \
#	--output softmax/Softmax \
#	--output_dir ./ncs \
#	--data_type FP32 \
#	--framework tf


#openvino optimize
python3 /hdd2/programs/intel/openvino_2020.1.023/deployment_tools/model_optimizer/mo.py \
	--input main_input1,main_input2,main_input3 \
	--input_model ./data/UCI_HAR_Dataset_output/merged_dcl.pb \
        --input_shape "[1,128,3],[1,128,3],[1,128,3]" \
 	--output softmax/Softmax \
	--output_dir ./ncs \
	--data_type FP16 \
	--framework tf

#fix 0 dimensions problem for Myriad and LSTM

python3 dim1d.py ./ncs/merged_dcl.xml


#python3 /hdd2/programs/intel/openvino_2020.1.023/deployment_tools/model_optimizer/mo.py \
#	--input_model ./second_model_ecg.pb \
#	--output_dir ./ncs \
#	--input_shape "[1,52,256]" \
#	--data_type FP16 \
#	--framework tf






#python3 split_tpu.py

#python3 second_model_tpu.py

#python3 first_model_tpu.py

#convert to tf lite first model dpu

#tflite_convert --output_file=./tflite/first_cnnlstm.tflite \
#--graph_def_file=./frozen_graph_first_model_tpu.pb --inference_type=QUANTIZED_UINT8 \
#--input_shapes=1,210,1 --input_arrays=input_1 \
#--output_arrays=max_pooling1d/Squeeze \
#--default_ranges_min=0 --default_ranges_max=255 --mean_values=0 --std_dev_values=1 \
#--change_concat_input_ranges=false \
#--allow_nudging_weights_to_use_fast_gemm_kernel=true \
#--allow_custom_ops

#python3 /hdd2/programs/openvino/openvino_2019.1.133/deployment_tools/model_optimizer/mo.py \
#	--input_model ./frozen_graph_second_model_tpu.pb \
#	--output_dir ./ncs \
#	--input_shape "[1,210,1]" \
#	--data_type FP32 \
#	--framework tf

#tflite_convert --output_file=./tflite/first_cnnlstm.tflite \
#--graph_def_file=./first_cnn_lstm_tpu.pb --inference_type=QUANTIZED_UINT8 \
#--input_shapes=1,210,1 --input_arrays=lstmsplit/input_2  \
#--output_arrays=lstmsplit/dense/BiasAdd \
#--default_ranges_min=0 --default_ranges_max=255 --mean_values=0 --std_dev_values=1 \
#--change_concat_input_ranges=false \
#--allow_nudging_weights_to_use_fast_gemm_kernel=true \
#--allow_custom_ops



#edgetpu_compiler ./tflite/cnnlstm.tflite
