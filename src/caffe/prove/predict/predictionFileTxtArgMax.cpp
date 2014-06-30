#include <cuda_runtime.h>

#include <cstring>
#include <cstdlib>
#include <vector>

#include <string>
#include <iostream>
#include <stdio.h>
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;

class Timer
{
  public:
  Timer() : start_(0), time_(0) {}

  void start()
  {
    start_ = cv::getTickCount();
  }

  void stop()
  {
    CV_Assert(start_ != 0);
    int64 end = cv::getTickCount();
    time_ += end - start_;
    start_ = 0;
  }

  double time()
  {
    double ret = time_ / cv::getTickFrequency();
    time_ = 0;
    return ret;
  }

  private:
  int64 start_, time_;
};

int main(int argc, char** argv) {

  Timer t;

  if (argc < 4 || argc > 6) {
    LOG(ERROR) << "test_net net_proto pretrained_net_proto iterations "
        << "[CPU/GPU] [Device ID]";
    return 1;
  }
  Caffe::set_phase(Caffe::TEST);

  //Setting CPU or GPU
  if (argc >= 5 && strcmp(argv[4], "GPU") == 0) {
    Caffe::set_mode(Caffe::GPU);
    int device_id = 0;
    if (argc == 6) {
      device_id = atoi(argv[5]);
    }
    Caffe::SetDevice(device_id);
    LOG(ERROR) << "Using GPU #" << device_id;
  } else {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }
  //get the net
  Net<float> caffe_test_net(argv[1]);
  //get trained net
  caffe_test_net.CopyTrainedLayersFrom(argv[2]);
  // Run ForwardPrefilled
  float loss;
  const vector<Blob<float>*>& result = caffe_test_net.ForwardPrefilled(&loss);

  LOG(ERROR)<< "Output result size: "<< result.size();
  // Now result will contain the argmax results.
  const float* argmaxs = result[1]->cpu_data();
  for (int i = 0; i < result[1]->num(); ++i) {
    LOG(ERROR)<< " Image: "<< i << " class:" << argmaxs[i];
  }
  
  return 0;
}

