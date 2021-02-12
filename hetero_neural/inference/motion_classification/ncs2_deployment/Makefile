hellomake: main.cpp
	g++  -O3 -o main --std=c++11 main.cpp -I. \
            -I/hdd2/programs/intel/openvino_2020.1.023/opencv/include/ \
            -I/hdd2/programs/intel/openvino_2020.1.023/deployment_tools/inference_engine/include/ \
            -I/hdd2/programs/intel/openvino_2020.1.023/inference_engine/include/cpp \
	    -I/hdd2/programs/intel/openvino_2020.1.023/deployment_tools/inference_engine/samples/cpp/common \
	    -L/hdd2/programs/intel/openvino_2020.1.023/deployment_tools/ngraph/lib/ -lngraph \
            -L/hdd2/programs/intel/openvino_2020.1.023/deployment_tools/inference_engine/lib/intel64 -linference_engine \
            -L/home/eejlny/inference_engine_cpp_samples_build/intel64/Release/lib/ -ldl -lpthread  \
	    -L/hdd2/programs/intel/openvino_2020.1.023/deployment_tools/inference_engine/external/tbb/lib -liomp5 -lgflags  \
            -L/hdd2/programs/intel/openvino_2020.1.023/opencv/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs