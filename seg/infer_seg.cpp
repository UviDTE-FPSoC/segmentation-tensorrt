#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <NvInfer.h>
#include <memory>
#include <NvOnnxParser.h>
#include <vector>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <algorithm>
#include <numeric>
#include <time.h>
#include <stdio.h>

// utilities ----------------------------------------------------------------------------------------------------------
//reset cumulative buffers for mean, max, min and variance
void reset_cumulative(unsigned long long int * total,
  unsigned long long int* min, unsigned long long int * max,
  unsigned long long int * variance)
{
  *total = 0;
  *min = ~0;
  *max = 0;
  *variance = 0;
}

//update cumulative buffers for mean, max, min and variance with new measurement
void update_cumulative(unsigned long long int * total,
  unsigned long long int* min, unsigned long long int * max,
  unsigned long long int * variance, unsigned long long int ns_begin,
  unsigned long long int ns_end, unsigned long long int clk_read_delay)
{
  unsigned long long int tmp = ( ns_end < ns_begin ) ?
    1000000000 - (ns_begin - ns_end) - clk_read_delay :
    ns_end - ns_begin - clk_read_delay ;

  *total = *total + tmp;
  *variance = *variance + tmp*tmp;

  if (tmp < *min) *min = tmp;
  if (tmp > *max) *max = tmp;

//  printf("total %lld, begin %lld, end %lld\n", *total, ns_begin, ns_end);
}

//calculate the variance from the cumulative
unsigned long long variance (unsigned long long variance ,
  unsigned long long total, unsigned long long rep_tests)
{

  float media_cuadrados, quef1, quef2, cuadrado_media, vari;

  media_cuadrados = (variance/(float)(rep_tests-1));
  quef1 = (total/(float)rep_tests);
  quef2=(total/(float)(rep_tests-1));
  cuadrado_media = quef1 * quef2;
  vari = media_cuadrados - cuadrado_media;
/*
  printf("media_cuadrados %f,",media_cuadrados );
  printf("quef1 %f,",quef1 );
  printf("quef2 %f,",quef2 );
  printf("cuadrado_media %f,",cuadrado_media );
  printf("variance %f\n",vari );
*/
  return (unsigned long long) vari;

  //return ((variance/(rep_tests-1))-(total/rep_tests)*(total/(rep_tests-1)));
}

// class to log errors, warnings, and other information during the build and inference phases
class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) override {
        // remove this 'if' if you need more logged info
        if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) {
            std::cout << msg << "\n";
        }
    }
} gLogger;

// destroy TensorRT objects if something goes wrong
struct TRTDestroy
{
    template <class T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

template <class T>
using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;

// calculate size of tensor
size_t getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    return size;
}

// preprocessing stage ------------------------------------------------------------------------------------------------
std::string * preprocessImage(std::fstream& file, float* gpu_input, const nvinfer1::Dims& dims)
{
    // get input dimensions 
    auto batch_size = dims.d[0];
    auto input_width = dims.d[3];
    auto input_height = dims.d[2];
    auto channels = dims.d[1];
    auto input_size = cv::Size(input_width, input_height);
    int from_to[] = {0,2,1,1,2,0};
    
    // masks vector (max size 16 as it is the max batch size for inference in this network)
    static std::string  masks[8];

    std::vector<std::string> csv_row;
    std::string line, word;

    for (size_t b = 0; b < batch_size; ++b)
    {
	    // read input image from csv
	    getline(file,line);
	    csv_row.clear();
        std::stringstream str(line);
        while(getline(str, word, ','))
        {
		csv_row.push_back(word);
	    }
	    std::string image_path = "/media/data/train_images/" + csv_row[0];
	    // store mask file
        masks[b] = csv_row[2];

        // read image and transform to RGB
	    cv::Mat frame_bgr = cv::imread(image_path);
	    cv::Mat frame_rgb(frame_bgr.size(),frame_bgr.type());
	    cv::mixChannels(&frame_bgr,1,&frame_rgb,1,from_to,3);

	    if (frame_bgr.empty())
	    {
		std::cerr << "Input image " << image_path << " load failed\n";
		return masks;
	    }

	    // cv::namedWindow("Input Image", cv::WINDOW_AUTOSIZE);
	    // cv::imshow("Input Image",frame_rgb);
	    // cv::waitKey();

	    cv::Mat frame_rgb_float;
	    cv::Mat frame_rgb_resized;
        // normalize
	    frame_rgb.convertTo(frame_rgb_float, CV_32FC3, 1.f / 255.f);
	    // resize
	    cv::resize(frame_rgb_float, frame_rgb_resized, input_size, 0, 0, cv::INTER_LINEAR);
            
	    cv::cuda::GpuMat gpu_frame;

	    // upload image to GPU
	    gpu_frame.upload(frame_rgb_resized);

      // copy image to input buffer as (n)chw
        std::vector<cv::cuda::GpuMat> chw;
	    for (size_t i = 0; i < channels; ++i)
	    {
		chw.emplace_back(cv::cuda::GpuMat(input_size, CV_32FC1, gpu_input + (b*3+i) * input_width * input_height));
	    }
	    cv::cuda::split(gpu_frame, chw);
    }

    return masks;
}

// post-processing stage ----------------------------------------------------------------------------------------------
void postprocessResults(float *gpu_output, const nvinfer1::Dims &dims, std::string *masks)
{
    auto batch_size = dims.d[0];
    // std::vector<float> thresh_upper{0.7,0.7,0.7,0.7};
    // std::vector<float> thresh_lower{0.4,0.5,0.4,0.5};
    // std::vector<int> min_area{180,260,200,500};

    // copy results from GPU to CPU
    std::vector<float> cpu_output(getSizeByDim(dims));
    cudaMemcpy(cpu_output.data(), gpu_output, cpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::vector<float> v_output(128*4*800);
    cv::Mat output = cv::Mat(128*4,800, CV_32FC1);
    
    for (size_t b = 0; b < batch_size; ++b)
    {
        // store as a single image with all 4 defects stacked
        for (size_t i = 0; i <128*800; ++i)
        {
            v_output.at(i) = cpu_output.at(i*4 + 128*4*800*b)*255;
            v_output.at(i+128*800) = cpu_output.at(i*4+1 + 128*4*800*b)*255;
            v_output.at(i+128*800*2) = cpu_output.at(i*4+2 + 128*4*800*b)*255;
            v_output.at(i+128*800*3) = cpu_output.at(i*4+3 + 128*4*800*b)*255;
        }
        // copy to Mat and save
        memcpy(output.data, v_output.data(), v_output.size()*sizeof(float));
        std::string tmp_name = "./masks_result/" + *(masks+b);
        // cv::imwrite(tmp_name, output);
    }
}

// initialize TensorRT engine and parse ONNX model --------------------------------------------------------------------
void parseOnnxModel(const std::string& model_path, TRTUniquePtr<nvinfer1::ICudaEngine>& engine,
                    TRTUniquePtr<nvinfer1::IExecutionContext>& context)
{
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TRTUniquePtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(gLogger)};
    TRTUniquePtr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(explicitBatch)};
    TRTUniquePtr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, gLogger)};
    TRTUniquePtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
    // parse ONNX
    if (!parser->parseFromFile(model_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO)))
    {
        std::cerr << "ERROR: could not parse the model.\n";
        return;
    }
    // allow TensorRT to use up to 1GB of GPU memory for tactic selection.
    config->setMaxWorkspaceSize(1ULL << 30);
    // use FP16 mode if possible
    // if (builder->platformHasFastFp16())
    // {
    // config->setFlag(nvinfer1::BuilderFlag::kFP32);
    // }
    // Explicit batch size
    builder->setMaxBatchSize(1);
    // generate TensorRT engine optimized for the target platform
    engine.reset(builder->buildEngineWithConfig(*network, *config));
    context.reset(engine->createExecutionContext());
}

// main pipeline ------------------------------------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cerr << "usage: " << argv[0] << " model.onnx\n";
        return -1;
    }
    std::string model_path(argv[1]);

    // initialize TensorRT engine and parse ONNX model
    TRTUniquePtr<nvinfer1::ICudaEngine> engine{nullptr};
    TRTUniquePtr<nvinfer1::IExecutionContext> context{nullptr};
    parseOnnxModel(model_path, engine, context);

    // get sizes of input and output and allocate memory required for input data and for output data
    std::vector<nvinfer1::Dims> input_dims; // we expect only one input
    std::vector<nvinfer1::Dims> output_dims; // and one output
    std::vector<void*> buffers(engine->getNbBindings()); // buffers for input and output data
    for (size_t i = 0; i < engine->getNbBindings(); ++i)
    {
        auto binding_size = getSizeByDim(engine->getBindingDimensions(i)) * sizeof(float);
        cudaMalloc(&buffers[i], binding_size);
        if (engine->bindingIsInput(i))
        {
            input_dims.emplace_back(engine->getBindingDimensions(i));
        }
        else
        {
            output_dims.emplace_back(engine->getBindingDimensions(i));
        }
    }
    if (input_dims.empty() || output_dims.empty())
    {
        std::cerr << "Expect at least one input and one output for network\n";
        return -1;
    }

    std::fstream file("../valid_files.csv", std::ios::in);
    std::ofstream output_file;
    output_file.open("output_bin.txt");

    //---------Save intermediate results in speed tests------------//
    unsigned long long int total_clk, min_clk, max_clk, var_clk, clk_read_avrg;
    unsigned long long int total_inf, min_inf, max_inf, var_inf;
    //-----------CLOCK INIT------------//
    clockid_t clk = CLOCK_REALTIME;
    struct timespec clk_struct_begin, clk_struct_end;

    if (clock_getres(clk, &clk_struct_begin))
		 printf("Failed in checking CLOCK_REALTIME resolution");
    else
		 printf("CLOCK_REALTIME resolution is %ld ns\n", clk_struct_begin.tv_nsec );

    printf("Measuring clock read overhead\n");
    reset_cumulative(&total_clk, &min_clk, &max_clk, &var_clk);
    clock_gettime(clk, &clk_struct_begin);
		for(int i = 0; i<5000; i++)
	{
		clock_gettime(clk, &clk_struct_begin);
		clock_gettime(clk, &clk_struct_end);

		update_cumulative(&total_clk, &min_clk, &max_clk, &var_clk,
			clk_struct_begin.tv_nsec, clk_struct_end.tv_nsec, 0);
	}
    printf("Clock Statistics for %d consecutive reads\n", 5000);
    printf("Average, Minimum, Maximum, Variance\n");
    printf("%lld,%lld,%lld,%lld\n", clk_read_avrg = total_clk/5000,
		min_clk, max_clk, variance (var_clk , total_clk, 5000));

    std::string *masks;

    reset_cumulative(&total_inf, &min_inf, &max_inf, &var_inf);

    if(file.is_open())
    {
    // discard first batch from measurement
    masks = preprocessImage(file, (float *) buffers[0], input_dims[0]);
    context->enqueue(1, buffers.data(), 0, nullptr);
    postprocessResults((float *) buffers[1], output_dims[0], masks);

    for (size_t bs = 0; bs < 100; ++bs)
    {
        // preprocess input data
        masks = preprocessImage(file, (float *) buffers[0], input_dims[0]);
        // inference//read a value from clock to eliminate "first-read-jitter"
        clock_gettime(clk, &clk_struct_begin);
        clock_gettime(clk, &clk_struct_begin);
        context->enqueue(1, buffers.data(), 0, nullptr);
        clock_gettime(clk, &clk_struct_end);
        update_cumulative(&total_inf, &min_inf, &max_inf, &var_inf,
                clk_struct_begin.tv_nsec, clk_struct_end.tv_nsec, clk_read_avrg);
        // postprocess results
        postprocessResults((float *) buffers[1], output_dims[0], masks); 
        
        }
    }
    printf("Average, Minimum, Maximum, Variance\n");
    printf("%lld,%lld,%lld,%lld\n", total_inf/100,
		min_inf, max_inf, variance (var_inf , total_inf, 100));

    for (void* buf : buffers)
    {
        cudaFree(buf);
    }

    output_file.close();

    return 0;
}
