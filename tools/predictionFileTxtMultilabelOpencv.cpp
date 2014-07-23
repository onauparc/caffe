#include <cuda_runtime.h>

#include <cstring>
#include <cstdlib>
#include <vector>

#include <string>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;

int main(int argc, char** argv) {

  //Timer t;

  if (argc < 4 || argc > 7) {
    LOG(ERROR) << "test_net net_proto pretrained_net_proto iterations "
        << "[CPU/GPU] [Device ID] filename";
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
  cv::Mat image = cv::imread("/home/claudia/caffe/data/pettorine_dataset/50x50/real/dentro/000030_GT_009964.tiff");
  vector<cv::Mat> images(2, image);
  const shared_ptr<ImageDataLayer<float> > image_data_layer =
      boost::static_pointer_cast<ImageDataLayer<float> >(
          caffe_test_net.layer_by_name("data"));
  image_data_layer->AddImages(images);
  vector<Blob<float>* > dummy_bottom_vec;
  const vector<Blob<float>*>& result = caffe_test_net.Forward(dummy_bottom_vec, &loss);

  LOG(ERROR)<< "Output result size: "<< result.size();

  const float* bottom_data = result[0]->cpu_data();
  const float* bottom_label = result[1]->cpu_data();
 
  // Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  int count = result[0]->count();
  int max = -1;
  int max_prob = -1;
  stringstream ss;
   
  int i = 0;
  for (int ind = 0; ind < count; ++ind) {
    // Accuracy
    int label = static_cast<int>(bottom_label[ind]);
//    cout << "Prob: " << bottom_data[ind] << " label: " << label << endl;
    if (bottom_data[ind] > 0)
    {
//      cout << ind << ": " << ind%10 << endl;
      if (bottom_data[ind] > max_prob) {
        max_prob = bottom_data[ind];
        max = i%10;
      }
    }
    if ((i+1)%10 == 0 ) {
//      cout << "-------Pred: " << max << endl;
      ss << max << " ";
      max = -1;
      max_prob = -1;
      //cout << endl;
    }
    i++;
    if ((ind+1)%41 == 0 ) {
      ss << " - " << (bottom_data[ind] > 0);
      cout << "Predicted: " << ss.str() << endl;
      ss.str("");
      max = -1;
      max_prob = -1;
      i = 0;
    }
  }

  return 0;
}

