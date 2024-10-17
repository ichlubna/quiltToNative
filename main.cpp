#include <CL/opencl.hpp>
#include <stdexcept>
#include <string>
#include <fstream>
#include <sstream>
#include <exception>
#include "arguments/arguments.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

class Params
{
    public:
    std::string inputImage;
    std::string outputImage;
    float focus;
    int cols;
    int rows;
    int width;
    int height;
    float tilt;
    float subp;
    float pitch;
    float center;
    float viewPortion;
};

void process(Params params)
{
    std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0)
        throw std::runtime_error("No OpenCL platforms available");
    cl::Context context(CL_DEVICE_TYPE_DEFAULT);
    std::ifstream file("kernel.cl");
    std::stringstream kernelContent;
    kernelContent << file.rdbuf();
    file.close();
    cl::Program program(context, kernelContent.str(), true);
    cl::CommandQueue queue(context);
  
    std::cerr << "Loading input image" << std::endl;
    int imageWidth, imageHeight, imageChannels;
    int imageChannelsGPU = 4;
    unsigned char *imageData = stbi_load(params.inputImage.c_str(), &imageWidth, &imageHeight, &imageChannels, imageChannelsGPU);
    if (imageData == nullptr)
        throw std::runtime_error("Failed to load image");
    
    std::cerr << "Allocating GPU memory" << std::endl;
    const cl::ImageFormat imageFormat(CL_RGBA, CL_UNSIGNED_INT8);
	cl::Image2D inputImageGPU(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imageFormat, imageWidth, imageHeight, 0, imageData);
	cl::Image2D outputImageGPU(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, imageFormat, params.width, params.height, 0, nullptr);

    stbi_image_free(imageData);

    std::cerr << "Processing on GPU" << std::endl;
    auto kernel = cl::compatibility::make_kernel<cl::Image2D&,cl::Image2D&, int, int, float, float, float, float, float, float>(program, "kernelMain"); 
    cl_int buildErr = CL_SUCCESS; 
    auto buildInfo = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
    for (auto &pair : buildInfo)
        if(!pair.second.empty() && !std::all_of(pair.second.begin(),pair.second.end(),isspace))
            std::cerr << pair.second << std::endl;
    cl::EnqueueArgs kernelArgs(queue, cl::NDRange(params.width, params.height));
    kernel(kernelArgs, inputImageGPU, outputImageGPU, params.rows, params.cols, params.tilt, params.pitch, params.center, params.viewPortion, params.subp, params.focus);
    queue.finish();

    std::cerr << "Storing the result" << std::endl;
    cl::array<size_t, 3> origin{0, 0, 0};
    cl::array<size_t, 3> size{static_cast<size_t>(params.width), static_cast<size_t>(params.height), 1};
    std::vector<unsigned char> outData;
    outData.resize(params.width * params.height * imageChannelsGPU);
    if(queue.enqueueReadImage(outputImageGPU, CL_TRUE, origin, size, 0, 0, outData.data()) != CL_SUCCESS)
        throw std::runtime_error("Cannot download the result");
    stbi_write_png(params.outputImage.c_str(), params.width, params.height, imageChannelsGPU, outData.data(), params.width * imageChannelsGPU);
}

int main(int argc, char *argv[])
{
    std::string helpText =  "This program takes a quilt image and produces the native Looking Glass image. All parameters below need to be specified according to the display model.\n"
                            "--help, -h Prints this help\n"
                            "-i input quilt image - 8-BIT RGBA\n"
                            "-o output native image\n"
                            "-rows number of rows in the quilt\n"
                            "-cols number of cols in the quilt\n"
                            "-width horizonal resolution of the display\n"
                            "-height vertical resolution of the display\n"
                            "-tilt display calibration\n"
                            "-pitch display calibration\n"
                            "-center display calibration\n"
                            "-subp display calibration\n"
                            "-viewPortion display calibration\n"
                            "-focus focusing value\n";
    Arguments args(argc, argv);
    if(args.printHelpIfPresent(helpText))
        return 0;
    if(argc < 2)
    {
        std::cerr << "Use --help" << std::endl;
        return 0;
    }

    Params params;
    params.inputImage = static_cast<std::string>(args["-i"]);
    params.outputImage = static_cast<std::string>(args["-o"]);
    params.cols = static_cast<int>(args["-cols"]);
    params.rows = static_cast<int>(args["-rows"]);
    params.width = static_cast<int>(args["-width"]);
    params.height = static_cast<int>(args["-height"]);
    params.tilt = static_cast<float>(args["-tilt"]);
    params.pitch = static_cast<float>(args["-pitch"]);
    params.center = static_cast<float>(args["-center"]);
    params.viewPortion = static_cast<float>(args["-viewPortion"]);
    params.subp = static_cast<float>(args["-subp"]);
    params.focus = static_cast<float>(args["-focus"]);

    try
    {
        process(params);
    }

    catch(const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
