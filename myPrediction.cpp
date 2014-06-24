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

int main(int argc, char** argv) {

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

  /*
  float max = 0;
  float max_i = 0;
  for (int i = 0; i < 1000; ++i) {
    float value = result[0]->cpu_data()[i];
    if (max < value){
      max = value;
      max_i = i;
    }
    LOG(ERROR) << "value: " << value << " i " << i;
  }
  LOG(ERROR) << "max: " << max << " i " << max_i;
  */

  const float* argmaxs = result[1]->cpu_data();
  for (int i = 0; i < result[1]->num(); ++i) {
    LOG(INFO) << " Image: "<< i << " predicted class: " << argmaxs[i];
  }

  return 0;
}

