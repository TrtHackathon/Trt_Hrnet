
#include <map>
#include <string>
#include <vector>
#include <fstream>
#include "trt_engine.h"
#include <unordered_map>
#include "opencv2/opencv.hpp"

int main(int argc, char**argv) {
    if (argc < 1) {
        std::cerr << "--trt_model, required!" << std::endl
            << "--input_height, default 384" << std::endl
            << "--input_width, default 288" << std::endl
            << "--map_height, default 96" << std::endl
            << "--map_width, default 72" << std::endl;
        exit(0);
    }

    std::map<std::string, std::string> args = {
        {"--trt_model",""},
        {"--input_height","384"},
        {"--input_width","288"},
        {"--map_height","96"},
        {"--map_width","72"}
    };

    for (int idx = 1; idx < argc; idx += 2) {
        if (args.count(argv[idx]) == 0) {
            std::cerr << argv[idx] << " is unknown!" << std::endl;
            exit(0);
        }
    }

    for (auto& arg : args) {
        if (arg.second == "") {
            std::cout << arg.first << " is not set!" << std::endl;
            exit(0);
        }
    }
    TRT::TrtEngine _trt_engine;
    _trt_engine.init(0, args["--trt_model"]);
    _trt_engine.warmup(_trt_engine.get_maximum_batch_size());

    //code for test value with python
    int input_height = std::stoi(args["--input_height"]);
    int input_width = std::stoi(args["--input_width"]);
    int map_height = std::stoi(args["--map_height"]);
    int map_width = std::stoi(args["--map_width"]);

    while (true) {

        std::string image_path = "";
        std::cout << "input image path" << std::endl;
        std::cin >> image_path;

        auto image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "read image fail!" << std::endl;
            continue;
        }

        auto trt_buf = _trt_engine.get_buf();
        auto buf = (float*)trt_buf.get();

        auto scale_h = image.rows / (float)input_height;
        auto scale_w = image.cols / (float)input_width;

        cv::Mat dst;
        cv::resize(image,dst,cv::Size(input_width, input_height));
        auto data = (unsigned char*)dst.data;
        for (int idx = 0; idx < dst.cols * dst.rows * dst.channels(); idx++) {
            buf[idx] = (float)data[idx];
        }

        /*std::ifstream ifs("D://hrnet//deep-high-resolution-net.pytorch//model2onnx//input.txt",std::ios::binary | std::ios::in);
        if (ifs.fail()) {
            std::cerr << "fail to open input val" << std::endl;
            exit(0);
        }

        size_t vlen = 1 * 3 * 384 * 288 * sizeof(float);
        ifs.read(buf, vlen);*/
        //auto image = cv::imread("test.jpg");
        

        std::unordered_map<std::string, char*> input = { {"input",(char*)buf} };
        std::unordered_map<std::string, std::vector<float>> output;
        _trt_engine.predict(1, input, output);

        std::vector<float>& indexs = output["indices"];
        for (auto val : indexs) {
            int height = (int)(val / map_width) * 4 * scale_h;
            int width = (int)((int)(val + 0.5) % map_width) * 4 * scale_w;
            std::cout << height << " " << width << std::endl;
            cv::circle(image, cv::Point(width, height), 3, (255, 0, 255), -1);
        }
        cv::imwrite("xx.jpg",image);
        /*for (auto& p : output) {
            /*int idx = 0;
            for (auto& val : p.second) {
                if (val > 0.9) {
                    std::cout << idx<<" --> "<< val << std::endl;
                }
                ++idx;
            }
            for (int idx = 0; idx < 17; ++idx) {
                std::cout << p.second[idx] << " ";
            }
            std::cout << std::endl;
        }*/
        /*
        std::vector<float>& indexs = output["indices"];
        for (auto val : indexs) {
            int height = (val / 72) * 4;
            int width = ((int)(val + 0.5) % 72) * 4;
            std::cout << height << " " << width << std::endl;
            cv::circle(image, cv::Point(width, height), 3, (255, 0, 255), -1);
        }

        cv::imshow("", image);
        cv::waitKey(0);
        */
    }

    return 0;
}
