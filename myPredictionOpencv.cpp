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
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

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
//  const vector<Blob<float>*>& result = caffe_test_net.ForwardPrefilled(&loss);

  // Run AddImagesAndLabels and Forward
  cv::Mat image = cv::imread("/home/ladydisaster/libs/caffe/examples/images/cat.jpg"); // or cat.jpg
  int batch_size = 6;
  vector<cv::Mat> images(batch_size, image);
  vector<int> labels(batch_size, 0);

  const shared_ptr<ImageDataLayer<float> > image_data_layer =
      boost::static_pointer_cast<ImageDataLayer<float> >(
          caffe_test_net.layer_by_name("data"));
  image_data_layer->AddImagesAndLabels(images, labels);
  vector<Blob<float>* > dummy_bottom_vec;
  std::cout << "Inizio forward " << std::endl;
  t.start();
  const vector<Blob<float>*>& result = caffe_test_net.Forward(dummy_bottom_vec, &loss);

  t.stop();
  std::cout << "Eseguito in " << t.time() << std::endl;

  /*
  LOG(INFO)<< "Output result size: "<< result.size();
  // Now result will contain the argmax results.
  const float* argmaxs = result[1]->cpu_data();
  for (int i = 0; i < result[1]->num(); ++i) {
    LOG(INFO)<< " Image: "<< i << " class:" << argmaxs[i];
  }*/
  return 0;
}

