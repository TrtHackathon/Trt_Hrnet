
#include <iostream>
#include <fstream>
#include "logger.h"
#include <string>
#include <unordered_map>

#include <NvInfer.h>
#include <NvOnnxParser.h>

using namespace nvinfer1;
using namespace nvonnxparser;

int main(int argc, char** argv) {
  if (argc < 1) {
    std::cerr
      << "--model, required!" << std::endl
      << "--output, required!" << std::endl
      << "--fp16, default false" << std::endl
      << "--max_batch_size, default 1" << std::endl;

    exit(0);
  }

  std::unordered_map<std::string, std::string> args = {
    {"--fp16","false"},
    {"--max_batch_size","64"},
    {"--model",""},
    {"--output","pose_hrnet_w48_384x288.trt"},
  };

  for (int idx = 1; idx < argc; idx += 2) {
    if (args.count(argv[idx]) == 0) {
      std::cerr << "args "<< argv[idx] <<" is unkown!" << std::endl;
      exit(0);
    }
    args[argv[idx]] = std::string(argv[idx + 1]);
  }

  for (auto& arg : args) {
    if (arg.second == "") {
      std::cerr << "args " << arg.first << " is not set!" << std::endl;
      exit(0);
    }
  }

  int maxBatchSize = std::stoi(args["--max_batch_size"]);
  std::string onnx_file = args["--model"];

  Logger gLogger;
  IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
  INetworkDefinition* network = builder->createNetworkV2(1);
  std::cout << network->hasImplicitBatchDimension() << std::endl;

  auto parser = createParser(*network, gLogger.getTRTLogger());
  parser->parseFromFile(onnx_file.c_str(), static_cast<int>(gLogger.getReportableSeverity()));

  auto bconfig = builder->createBuilderConfig();
  if (args["--fp16"] == "true")
    bconfig->setFlag(BuilderFlag::kFP16);
  bconfig->setMaxWorkspaceSize(1 << 30);

  builder->setMaxBatchSize(maxBatchSize);
  auto engine = builder->buildEngineWithConfig(*network, *bconfig);

  int nbio = engine->getNbBindings();
  //std::cout << engine->getNbBindings() << std::endl;

  for (int idx = 0; idx < nbio; ++idx) {
    auto name = engine->getBindingName(idx);
    auto dim = engine->getBindingDimensions(idx);
    if (engine->bindingIsInput(idx)) {
      std::cout << "input name -> " << name << " shape -> ";
      for (auto id = 0; id < dim.nbDims; ++id) {
        std::cout << dim.d[id] << " ";
      }
      std::cout << std::endl;
    }
    else {
      std::cout << "output name -> " << name << " shape -> ";
      for (auto id = 0; id < dim.nbDims; ++id) {
        std::cout << dim.d[id] << " ";
      }
      std::cout << std::endl;
    }
  }

  std::cout << "Number Layers " << engine->getNbLayers() << std::endl;
  IHostMemory* trtModelStream = engine->serialize();

  std::string trt_file = args["--output"];
  std::cout << "Saving model " << trt_file << std::endl;
  std::ofstream ofs(trt_file, std::ios::binary);
  if (ofs.fail()) {
    std::cout << trt_file << " open fail!" << std::endl;
    exit(0);
  }
  ofs.write((char*)trtModelStream->data(), trtModelStream->size());
  ofs.close();
  std::cout << "Build TRT Model Success!" << std::endl;

  parser->destroy();
  engine->destroy();
  network->destroy();
  builder->destroy();

  return 0;
}
