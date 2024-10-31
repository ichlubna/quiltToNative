#include <CL/opencl.hpp>
#include <filesystem>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <fstream>
#include <sstream>
#include <exception>
#include "arguments/arguments.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

constexpr int imageChannelsGPU = 4;

class Params
{
    public:
    std::string inputPath;
    std::string outputPath;
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

void storeGPUImage(cl::CommandQueue queue, cl::Image2D image, std::string path)
{
    size_t width{0}, height{0};
    image.getImageInfo(CL_IMAGE_WIDTH, &width);
    image.getImageInfo(CL_IMAGE_HEIGHT, &height);
    std::vector<unsigned char> outData;
    outData.resize(width * height * imageChannelsGPU);
    if(queue.enqueueReadImage(image, CL_TRUE, cl::array<size_t, 3>{0, 0, 0}, cl::array<size_t, 3>{static_cast<size_t>(width), static_cast<size_t>(height), 1}, 0, 0, outData.data()) != CL_SUCCESS)
        throw std::runtime_error("Cannot download the result");
    stbi_write_png(path.c_str(), width, height, imageChannelsGPU, outData.data(), width * imageChannelsGPU);
}

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

    std::cerr << "Loading images and allocating GPU memory" << std::endl;
    const cl::ImageFormat imageFormat(CL_RGBA, CL_UNSIGNED_INT8);
    unsigned char *imageData{nullptr}; 
    int viewCount = params.cols * params.rows;
    int imageWidth{0}, imageHeight{0}, imageChannels{0};
    int viewWidth{0}, viewHeight{0}, viewChannels{0};
    if (std::filesystem::is_directory(params.inputPath))
    {
        for (const auto& file : std::filesystem::directory_iterator(params.inputPath))
        {
            unsigned char *imageData = stbi_load(file.path().c_str(), &viewWidth, &viewHeight, &viewChannels, 0);
            if (imageData == nullptr)
                throw std::runtime_error("Failed to load image");
            break;
        }
        imageWidth = viewWidth*params.cols;
        imageHeight = viewHeight*params.rows;
    }
    else
    {
        imageData = stbi_load(params.inputPath.c_str(), &imageWidth, &imageHeight, &imageChannels, imageChannelsGPU);
        if (imageData == nullptr)
            throw std::runtime_error("Failed to load image " + params.inputPath);
    }
   
	cl::Image2D inputImageGPU(context, CL_MEM_READ_ONLY, imageFormat, imageWidth, imageHeight, 0, nullptr);
    
    if (std::filesystem::is_directory(params.inputPath))
    {
        int counter = 0;
        auto iterator = std::filesystem::directory_iterator(params.inputPath);
        std::vector<std::filesystem::path> files; 
        for (const auto& file : std::filesystem::directory_iterator(params.inputPath))
            files.push_back(file);
        std::sort(files.begin(), files.end());
        for (const auto& file : files)
        {
            imageData = stbi_load(file.c_str(), &viewWidth, &viewHeight, &viewChannels, imageChannelsGPU);
            if (imageData == nullptr)
                throw std::runtime_error("Failed to load image " + file.string());
            cl::array<size_t, 3> origin{0, 0, 0};
            int x = counter % params.cols;
            int y = params.rows - 1 - (counter / params.cols);
            origin[0] = x*viewWidth;
            origin[1] = y*viewHeight;
            cl::array<size_t, 3> size{static_cast<size_t>(viewWidth), static_cast<size_t>(viewHeight), 1};
            if(queue.enqueueWriteImage(inputImageGPU, CL_TRUE, origin, size, 0, 0, imageData) != CL_SUCCESS)
                throw std::runtime_error("Cannot upload the image " + file.string() + " to GPU");
            stbi_image_free(imageData);
            counter++;
            if(counter > viewCount)
            {
                std::cerr << "The number of input files is higher than the expected quilt size. Using only the first " << viewCount << " files" << std::endl;
                break;
            }
        }
        if(counter < viewCount-1)
            throw std::runtime_error("The number of input images is lower than the expected quilt size");
        std::cerr << "Storing the quilt" << std::endl;
        storeGPUImage(queue, inputImageGPU, std::filesystem::path(params.outputPath) / "quilt.png");
    }
    else
    {
        if(queue.enqueueWriteImage(inputImageGPU, CL_TRUE, cl::array<size_t, 3>{0, 0, 0}, cl::array<size_t, 3>{static_cast<size_t>(imageWidth), static_cast<size_t>(imageHeight), 1}, 0, 0, imageData) != CL_SUCCESS)
            throw std::runtime_error("Cannot upload the quilt to GPU");
        stbi_image_free(imageData);
    }
         
	cl::Image2D outputImageGPU(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, imageFormat, params.width, params.height, 0, nullptr);


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
    storeGPUImage(queue, outputImageGPU, std::filesystem::path(params.outputPath) / "output.png");
}

int main(int argc, char *argv[])
{
    std::string helpText =  "This program takes a quilt image and produces the native Looking Glass image. All parameters below need to be specified according to the display model.\n"
                            "--help, -h Prints this help\n"
                            "-i input quilt image or directory - 8-BIT RGBA, all views having the same resolution\n"
                            "-o output directory - results stored as output.png and quilt.png\n"
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
    params.inputPath = static_cast<std::string>(args["-i"]);
    params.outputPath = static_cast<std::string>(args["-o"]);
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
