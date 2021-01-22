// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>
#include <string>
#include <samples/common.hpp>

#include <inference_engine.hpp>
#include <details/os/os_filesystem.hpp>
#include <samples/ocv_common.hpp>
#include <samples/classification_results.h>

using namespace InferenceEngine;

#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
#define tcout std::wcout
#define file_name_t std::wstring
#define WEIGHTS_EXT L".bin"
#define imread_t imreadW
#define ClassificationResult_t ClassificationResultW
#else
#define tcout std::cout
#define file_name_t std::string
#define WEIGHTS_EXT ".bin"
#define imread_t cv::imread
#define ClassificationResult_t ClassificationResult
#endif

//#define one_model_only 1
#define linecount 0 //specify the line to process

//input 70 -> 6
//input 0 -> 5

const std::string WHITESPACE = " \n\r\t\f\v";

std::string ltrim(const std::string& s)
{
	size_t start = s.find_first_not_of(WHITESPACE);
	return (start == std::string::npos) ? "" : s.substr(start);
}

std::string rtrim(const std::string& s)
{
	size_t end = s.find_last_not_of(WHITESPACE);
	return (end == std::string::npos) ? "" : s.substr(0, end + 1);
}

std::string trim(const std::string& s)
{
	return rtrim(ltrim(s));
}


cv::Mat loadFromTXT(std::string input_image_path1, std::string input_image_path2, std::string input_image_path3, int opencv_type)
{

    cv::Mat m;

    std::cout << "Loading data from 3 TXT files starting with " << input_image_path1  << std::endl;

    std::ifstream file1(input_image_path1);
    std::ifstream file2(input_image_path2);
    std::ifstream file3(input_image_path3);

    if (file1&&file2&&file3)
    {

	file1.seekg(0,std::ios::end);
        std::streampos length1 = file1.tellg();
        file1.seekg(0,std::ios::beg);
        
	std::vector<char>       buffer1(length1);
        file1.read(&buffer1[0],length1);

        std::stringstream       ss1;
        ss1.rdbuf()->pubsetbuf(&buffer1[0],length1);


	file2.seekg(0,std::ios::end);
        std::streampos length2 = file2.tellg();
        file2.seekg(0,std::ios::beg);
        
	std::vector<char>       buffer2(length2);
 	file2.read(&buffer2[0],length2);

        std::stringstream       ss2;
        ss2.rdbuf()->pubsetbuf(&buffer2[0],length2);


	file3.seekg(0,std::ios::end);
        std::streampos length3 = file3.tellg();
        file3.seekg(0,std::ios::beg);
        
	std::vector<char>       buffer3(length3);

 	file3.read(&buffer3[0],length3);

        std::stringstream       ss3;
        ss3.rdbuf()->pubsetbuf(&buffer3[0],length3);



        std::string line1;
        std::string line2;
        std::string line3;
	int lcounter;

	for(lcounter=0;lcounter < linecount;lcounter++) 
	{
	   getline(ss1, line1);
	   getline(ss2, line2);
	   getline(ss3, line3);
	}

        //for(int lcounter=0;lcounter < linecount;lcounter++) 
        //{
	   getline(ss1, line1);
	   getline(ss2, line2);
	   getline(ss3, line3);

   
	   line1 = trim(line1);
	   line2 = trim(line2);
	   line3 = trim(line3);


           std::stringstream ssline1(line1);
           std::stringstream ssline2(line2);
           std::stringstream ssline3(line3);
           std::string val1,val2,val3;

	   //std::cout << " line " << line1 << std::endl;
           //while (getline(ssline1, val1,' ')&&getline(ssline2, val2, ' ')&&getline(ssline3, val3, ' '))
           //{
              // Check if s consists only of whitespaces
	      //if ((std::all_of(val1.begin(),val1.end(),isspace))||(std::all_of(val2.begin(),val2.end(),isspace))||(std::all_of(val3.begin(),val3.end(),isspace)))
	      //std::cout << " val1 " << val1 << " val2 " << val2 << " val3 " << val3  << std::endl;
           //   dvals.push_back(stod(val1));
           //   dvals.push_back(stod(val2));
           //   dvals.push_back(stod(val3));
           //}
           std::vector<double> dvals1;
     	   while (getline(ssline1, val1,' '))
           {
              // Check if s consists only of whitespaces
	      //if ((std::all_of(val1.begin(),val1.end(),isspace))||(std::all_of(val2.begin(),val2.end(),isspace))||(std::all_of(val3.begin(),val3.end(),isspace)))
	      //std::cout << " val1 " << val1 << " val2 " << val2 << " val3 " << val3  << std::endl;
              dvals1.push_back(stod(val1));

           }
           cv::Mat mline1(dvals1, true);
           cv::transpose(mline1, mline1);

	   m.push_back(mline1);
	   std::cout << "size of m " << m.size() << std::endl;
   	   std::cout << "size of mline " << mline1.size() << std::endl;

           std::vector<double> dvals2;
     	   while (getline(ssline2, val2,' '))
           {
              // Check if s consists only of whitespaces
	      //if ((std::all_of(val1.begin(),val1.end(),isspace))||(std::all_of(val2.begin(),val2.end(),isspace))||(std::all_of(val3.begin(),val3.end(),isspace)))
	      //std::cout << " val1 " << val1 << " val2 " << val2 << " val3 " << val3  << std::endl;
              dvals2.push_back(stod(val2));    

           }
           cv::Mat mline2(dvals2, true);
           cv::transpose(mline2, mline2);

	   m.push_back(mline2);
	   std::cout << "size of m " << m.size() << std::endl;
   	   std::cout << "size of mline " << mline2.size() << std::endl;

           std::vector<double> dvals3;
     	   while (getline(ssline3, val3,' '))
           {
              // Check if s consists only of whitespaces
	      //if ((std::all_of(val1.begin(),val1.end(),isspace))||(std::all_of(val2.begin(),val2.end(),isspace))||(std::all_of(val3.begin(),val3.end(),isspace)))
	      //std::cout << " val1 " << val1 << " val2 " << val2 << " val3 " << val3  << std::endl;
              dvals3.push_back(stod(val3));
           }
           cv::Mat mline3(dvals3, true);
           cv::transpose(mline3, mline3);

	   m.push_back(mline3);
	   std::cout << "size of m " << m.size() << std::endl;
   	   std::cout << "size of mline " << mline3.size() << std::endl;


	   //exit(0);
    
	 
	//}
	std::cout << " reshaping " << std::endl;

        int ch = CV_MAT_CN(opencv_type);

	std::cout << " ch is " << ch << std::endl;

        //m = m.reshape(ch);

        std::cout << "rows : " << m.rows << std::endl;
        std::cout << "cols : " << m.cols << std::endl;

	cv::transpose(m, m);

        std::cout << "rows : " << m.rows << std::endl;
        std::cout << "cols : " << m.cols << std::endl;


        m.convertTo(m, opencv_type);
        return m;
   }
   else
   {
	std::cout << "Error no files " << std::endl;
        exit(1);
   }
}


#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
cv::Mat imreadW(std::wstring input_image_path) {
    cv::Mat image;
    std::ifstream input_image_stream;
    input_image_stream.open(
        input_image_path.c_str(),
        std::iostream::binary | std::ios_base::ate | std::ios_base::in);
    if (input_image_stream.is_open()) {
        if (input_image_stream.good()) {
            std::size_t file_size = input_image_stream.tellg();
            input_image_stream.seekg(0, std::ios::beg);
            std::vector<char> buffer(0);
            std::copy(
                std::istream_iterator<char>(input_image_stream),
                std::istream_iterator<char>(),
                std::back_inserter(buffer));
            image = cv::imdecode(cv::Mat(1, file_size, CV_8UC1, &buffer[0]), cv::IMREAD_COLOR);
        } else {
            tcout << "Input file '" << input_image_path << "' processing error" << std::endl;
        }
        input_image_stream.close();
    } else {
        tcout << "Unable to read input file '" << input_image_path << "'" << std::endl;
    }
    return image;
}


int wmain(int argc, wchar_t *argv[]) {
#else
int main(int argc, char *argv[]) {
#endif
    try {
        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (argc != 14) {
            tcout << "Usage : ./hello_classification <path_to_model1> <path_to_model2> <9*path_to_data> <device_name1> <device_name2>" << std::endl;
            return EXIT_FAILURE;
        }

        const file_name_t input_model1{argv[1]};
	const file_name_t input_model2{argv[2]};
        const file_name_t input_image_path1{argv[3]};
	const file_name_t input_image_path2{argv[4]};
	const file_name_t input_image_path3{argv[5]};
        const file_name_t input_image_path4{argv[6]};
	const file_name_t input_image_path5{argv[7]};
	const file_name_t input_image_path6{argv[8]};
        const file_name_t input_image_path7{argv[9]};
	const file_name_t input_image_path8{argv[10]};
	const file_name_t input_image_path9{argv[11]};



#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        const std::string device_name = InferenceEngine::details::wStringtoMBCSstringChar(argv[4]);
#else
        const std::string device_name1{argv[12]};

        const std::string device_name2{argv[13]};
#endif
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Load inference engine instance -------------------------------------
        Core ie;
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 2. Read IR Generated by ModelOptimizer (.xml and .bin files) ------------
        CNNNetwork network1 = ie.ReadNetwork(input_model1, input_model1.substr(0, input_model1.size() - 4) + WEIGHTS_EXT);
        CNNNetwork network2 = ie.ReadNetwork(input_model2, input_model2.substr(0, input_model2.size() - 4) + WEIGHTS_EXT);


	//cout << "shape " << network1.inputs['data'].shape << endl;

        network1.setBatchSize(1);
        network2.setBatchSize(1);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Configure input & output ---------------------------------------------
        // --------------------------- Prepare input blobs -----------------------------------------------------
        InputsDataMap inputsInfo(network1.getInputsInfo());
        InputInfo::Ptr input_info1;
        std::string input_name1;
        InputInfo::Ptr input_info2;
        std::string input_name2;
        InputInfo::Ptr input_info3;
        std::string input_name3;
	int load_input = 0;

       for (auto & item : inputsInfo) {

	std::cout << "Setting input "<< load_input << std::endl;

	if (load_input == 0)
	{
		input_info1 = item.second;
        	input_name1 = item.first;
	        /* Mark input as resizable by setting of a resize algorithm.
        	 * In this case we will be able to set an input blob of any shape to an infer request.
       		  * Resize and layout conversions are executed automatically during inference */
        	//input_info1->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
        	input_info1->setLayout(Layout::CHW);
        	input_info1->setPrecision(Precision::FP32);
	}
	else if (load_input == 1)
	{
        	input_info2 = item.second;
        	input_name2 = item.first;
        	input_info2->setLayout(Layout::CHW);
        	input_info2->setPrecision(Precision::FP32);

	}
	else if (load_input == 2)
	{
        	input_info3 = item.second;
        	input_name3 = item.first;
         	input_info3->setLayout(Layout::CHW);
        	input_info3->setPrecision(Precision::FP32);

	}

	load_input ++;


	}


        std::cout << "Input1 layer name is " << input_name1 << std::endl;
        std::cout << "Input2 layer name is " << input_name2 << std::endl;
        std::cout << "Input3 layer name is " << input_name3 << std::endl;



        InputInfo::Ptr input_info4 = network2.getInputsInfo().begin()->second;
        std::string input_name4 = network2.getInputsInfo().begin()->first;

        /* Mark input as resizable by setting of a resize algorithm.
         * In this case we will be able to set an input blob of any shape to an infer request.
         * Resize and layout conversions are executed automatically during inference */
        //input_info2->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
        //input_info2->setLayout(Layout::CHW);
        input_info4->setPrecision(Precision::FP32);


        // --------------------------- Prepare output blobs ----------------------------------------------------
        DataPtr output_info1 = network1.getOutputsInfo().begin()->second;
        std::string output_name1 = network1.getOutputsInfo().begin()->first;
	std::cout << "Output1 layer name is " << output_name1 << std::endl;


        output_info1->setPrecision(Precision::FP32);

        DataPtr output_info2 = network2.getOutputsInfo().begin()->second;
        std::string output_name2 = network2.getOutputsInfo().begin()->first;
	std::cout << "Output2 layer name is " << output_name2 << std::endl;


        output_info2->setPrecision(Precision::FP32);

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Loading model to the device ------------------------------------------
        ExecutableNetwork executable_network1 = ie.LoadNetwork(network1, device_name1);
        ExecutableNetwork executable_network2 = ie.LoadNetwork(network2, device_name2);

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Create infer request -------------------------------------------------
        InferRequest infer_request1 = executable_network1.CreateInferRequest();
        InferRequest infer_request2 = executable_network2.CreateInferRequest();

        ConstInputsDataMap cInputInfo = executable_network1.GetInputsInfo();
        /** Stores all input blobs data **/
        std::cout << "Inputinfo size " << std::to_string(cInputInfo.size()) << std::endl; 
    

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Prepare input --------------------------------------------------------
        /* Read input image to a blob and set it to an infer request without resize and layout conversions. */


    	InputsDataMap inputInfo(network1.getInputsInfo());
        int input_index = 0;
	for (auto & item : inputInfo)
        {
		cv::Mat image;
		std::cout << "Preparing input " << (input_index) << std::endl;
		if (input_index == 0)
		{
			image = loadFromTXT(input_image_path1,input_image_path2,input_image_path3,CV_32F);
		}
		if (input_index == 1)
		{
			image = loadFromTXT(input_image_path4,input_image_path5,input_image_path6,CV_32F);
		}
		if (input_index == 2)
		{
			image = loadFromTXT(input_image_path7,input_image_path8,input_image_path9,CV_32F);
		}
		
		Blob::Ptr inputBlob = infer_request1.GetBlob(item.first);
        	SizeVector dims = inputBlob->getTensorDesc().getDims();
        	/** Fill input tensor with images. First b channel, then g and r channels **/
        	size_t num_channels = dims[0];
        	size_t image_size = dims[2] * dims[1];
 		std::cout << "    channels    = " << dims[0]    << std::endl;
        	std::cout << "    dim 1   = " << dims[1]   << std::endl;
        	std::cout << "    dim 2   = " << dims[2]   << std::endl;

                MemoryBlob::Ptr minput = as<MemoryBlob>(inputBlob);
                if (!minput) {
                  std::cout << "We expect MemoryBlob from inferRequest, but by fact we were not able to cast inputBlob to MemoryBlob" << std::endl;
                  return 1;
               }
               // locked memory holder should be alive all time while access to its buffer happens
               auto minputHolder = minput->wmap();

               auto data = minputHolder.as<PrecisionTrait<Precision::FP32>::value_type *>();
                 for (int row = 0; row < dims[1]; row++) {
       			for (int col = 0; col < dims[2]; col++) {

			//for(size_t ch =0;ch < num_channels;ch++) {
                        //float fpixel = (imagesData.at(image_id).get()[pid*num_channels + ch])/255.0;
			//data[image_id * image_size * num_channels + ch * image_size + pid] = fpixel;
			//int index = (int)pid;
			//data[ch*image_size+pid] = image.at<float>(0,pid*num_channels + ch);
			//data[pid] = image.at<float>(pid);
			data[row*dims[2]+col] = image.at<float>(row,col);
			//data[row*dims[2]+col] = 0.0;

			//std::cout << "data at row " << row << " col " << col << " is " << data[row*dims[2]+col] << std::endl;
			//}
			}
                    }
		if (input_index == 0)
		{
			std::cout << "blob set " << input_index << std::endl;
	       		infer_request1.SetBlob(input_name1, inputBlob);  // infer_request accepts input blob of any size
		}
		if (input_index == 1)
		{
			//std::cout << " Error too many inputs in model " << std::endl;
			//exit(1);
			std::cout << "blob set " << input_index << std::endl;
	       		infer_request1.SetBlob(input_name2, inputBlob);  // infer_request accepts input blob of any size
		}
		if (input_index == 2)
		{
			//std::cout << " Error too many inputs in model " << std::endl;
			//exit(1);
			std::cout << "blob set " << input_index << std::endl;
	       		infer_request1.SetBlob(input_name3, inputBlob);  // infer_request accepts input blob of any size
		}
		input_index++;

	}
   

	std::cout << "Starting inference " << std::endl;


        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 7. Do inference --------------------------------------------------------
        /* Running the request synchronously */

        typedef std::chrono::high_resolution_clock Time;
        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
        typedef std::chrono::duration<float> fsec;

        double total = 0.0;
        /** Start inference & calc performance **/
        auto t0 = Time::now();


        infer_request1.Infer();

 


        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 8. Process output ------------------------------------------------------
        

	//Blob::Ptr output1 = infer_request1.GetBlob(input_name1);
	//Blob::Ptr output1 = infer_request1.GetBlob("softmax/Softmax"); //Make sure we get the rigt layer as output
        Blob::Ptr output1 = infer_request1.GetBlob(output_name1);

 


	#ifdef one_model_only

       		auto t1 = Time::now();
        	fsec fs = t1 - t0;
        	ms d = std::chrono::duration_cast<ms>(fs);
        	total = d.count();

    		std::cout << "Total execution time " << total << " ms " << std::endl;    

 		SizeVector dims = output1->getTensorDesc().getDims();
        	size_t num_channels = dims[0];
        	size_t image_size = dims[2] * dims[1];
 		std::cout << " out   channels    = " << dims[0]    << std::endl;
        	std::cout << " out   dim 1   = " << dims[1]   << std::endl;
       		std::cout << " out   dim 2   = " << dims[2]   << std::endl;


		float* data_buffer = static_cast<float*>(output1->buffer());
		for (size_t pid = 0; pid < 6; pid++) {
			std::cout << "one model only out at " << pid+1 << " is " << data_buffer[pid] << std::endl;
		}

        	// Print classification results
        	//ClassificationResult_t classificationResult(output1, {input_image_path1});
        	//classificationResult.print();

	#else


		infer_request2.SetBlob(input_name4, output1);  // second infer request

        	/* Running the request synchronously */
        	infer_request2.Infer();

       		auto t1 = Time::now();
        	fsec fs = t1 - t0;
        	ms d = std::chrono::duration_cast<ms>(fs);
        	total = d.count();

    		std::cout << "Total execution time " << total << " ms " << std::endl;    

        	Blob::Ptr output2 = infer_request2.GetBlob(output_name2);

       		float* data_buffer = static_cast<float*>(output2->buffer());
		for (size_t pid = 0; pid <6; pid++) {
			std::cout << "two model pipeline out at " << pid+1 << " is " << data_buffer[pid] << std::endl;
		}


        	// Print classification results
        	//ClassificationResult_t classificationResult(output2, {input_image_path1});
        	//classificationResult.print();
	#endif
        // -----------------------------------------------------------------------------------------------------
    } catch (const std::exception & ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "This sample is an API example, for any performance measurements "
                 "please use the dedicated benchmark_app tool" << std::endl;
    return EXIT_SUCCESS;
}
