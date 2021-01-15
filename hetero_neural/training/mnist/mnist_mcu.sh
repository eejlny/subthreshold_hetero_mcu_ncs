python3 mnist_mcu.py

#input range (0,255) then mean = 0, std_dev = 1
#input range (-1,1) then mean = 127.5, std_dev = 127.5
#input range (0,1) then mean = 0, std_dev = 255

#python3 split_tpu.py

#python3 second_model_tpu.py

python3 first_model_mcu.py

#tflite_convert --output_file=./tflite/first_model_mcu.tflite --graph_def_file=./frozen_graph_first_model_mcu.pb \
#--inference_type=QUANTIZED_UINT8 --input_shapes=1,28,28,1 --input_arrays=flatten_input  --output_arrays=dense_1/Softmax \
#--default_ranges_min=0 --default_ranges_max=255 --mean_values=0 --std_dev_values=1 \
#--change_concat_input_ranges=false --allow_nudging_weights_to_use_fast_gemm_kernel=true \
#--allow_custom_ops

tflite_convert --output_file=./tflite/first_model_mcu.tflite --graph_def_file=./frozen_graph_first_model_mcu.pb \
--inference_type=QUANTIZED_UINT8 --inference_input_type=QUANTIZED_UINT8 --input_shapes=1,28,28,1 --input_arrays=conv2d_input  --output_arrays=dense_1/Softmax \
--default_ranges_min=0 --default_ranges_max=255 --mean_values=0 --std_dev_values=1 \
--change_concat_input_ranges=false --allow_nudging_weights_to_use_fast_gemm_kernel=true \
--allow_custom_ops


#python3 write_cc.py

xxd -i ./tflite/first_model_mcu.tflite > ./tflite/mnist_model.cc



