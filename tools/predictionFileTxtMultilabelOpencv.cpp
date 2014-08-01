#include <cuda_runtime.h>

#include <cstring>
#include <cstdlib>
#include <vector>

#include <string>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <fstream>
#include <cfloat>

#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;


void readLabels(const char* file_list, vector<string>& labels, vector<string>& paths);
string convertLabel(const vector<int>& labels);

string convertLabel(const vector<int>& labels){
  stringstream ss;
  for (int k = 0; k < 4; k++){
    for (int i = 0; i < 10; i++){
      if (labels[i+k*10] == 1){
        ss << i;
        break;
      }
    }
  }
  return ss.str();
}

void readLabels(const char* file_list, vector<string>& converted_labels, vector<string>& paths){
  //read labes and images
  std::string line;
  vector<string> labels;
  std::ifstream infile(file_list, std::ifstream::in);
  while (std::getline(infile, line))
  {
    vector<int> labels;
    std::stringstream ss(line); 
    string path;
    ss >> path;
    paths.push_back(path);
    for (int i = 0; i < 41; ++i)
    {
      int label;
      ss >> label;
      labels.push_back(label);
    }
    converted_labels.push_back(convertLabel(labels));
  }
}

int main(int argc, char** argv) {

  //Timer t;

  if (argc != 4) {
    LOG(ERROR) << "predictionFileTxtMultilabelOpencv net_proto pretrained_model file_list";
    return 1;
  }
  string net_proto = argv[1];
  string pretrained_model = argv[2];
  char* file_list = argv[3];

  Caffe::set_phase(Caffe::TEST);
  //Setting GPU
  Caffe::set_mode(Caffe::GPU);
  int device_id = 0;
  Caffe::SetDevice(device_id);
  LOG(ERROR) << "Using GPU #" << device_id;

  //get the net
  Net<float> caffe_test_net(net_proto);
  //get trained net
  caffe_test_net.CopyTrainedLayersFrom(pretrained_model);

  //read and conver labels
  vector<string> labels;
  vector<string> paths;
  readLabels(file_list, labels, paths);
  int num_images = labels.size();

  //read opencv images
  vector<cv::Mat> images;
  for (int i = 0; i < paths.size(); ++i){
    images.push_back(cv::imread(paths[i]));  
  }

  const shared_ptr<ImageDataLayer<float> > image_data_layer =
      boost::static_pointer_cast<ImageDataLayer<float> >(
          caffe_test_net.layer_by_name("data"));
  //add blur
  /*for (int i = 0; i < images.size(); ++i)
  {
    GaussianBlur(images[i], images[i], Size(3,3), 1);
  }*/
  
  //add images to layer
  image_data_layer->AddImages(images);
  //make prediction 
  vector<Blob<float>* > dummy_bottom_vec;
  float loss;
  const vector<Blob<float>*>& result = caffe_test_net.Forward(dummy_bottom_vec, &loss);

  LOG(ERROR)<< "Output result size: "<< result.size();

  const float* bottom_data = result[0]->cpu_data();
  const float* bottom_label = result[1]->cpu_data();

  int count = result[0]->count();
  int max = -1;
  float max_prob = -FLT_MAX;
  stringstream ss;
  
  int num_labels = 41;
  int ind = 0;
  int true_positives = 0;
  vector<string> predicted;
  for (int i = 0; i < num_images; ++i) {
    stringstream ss;
    //controllo il numero della pettorina
    for (int l = 0; l < num_labels-1; ++l) {
      ind = num_labels*i+l;
      if (bottom_data[ind] > max_prob) {
        max_prob = bottom_data[ind];
        max = l%10;
      }
      if ((l+1)%10 == 0) {
        ss << max;
        max = -1;
        max_prob = -FLT_MAX;
      }
    }
    //controllo se è una pettorina
    ind++;
    cout << paths[i] << endl;
    cout << labels[i] << ": " << ss.str();
    if (bottom_data[ind] <= 0){
      ss.str("NO TEXT");
    }
    else{
      if (labels[i].compare(ss.str()) == 0){
        cout << " <-- CORRETTO!!!";
        true_positives++;
      }
    }
    cout << endl;
    predicted.push_back(ss.str());
    //cout << "-----------------------------------------------------------------" << endl;
  }
  float accuracy = 0;
  if (true_positives > 0){
   accuracy = (float)true_positives/(float)num_images;
  }
  cout << "TP/TOTAL: " << true_positives << "/"<< num_images << endl;
  cout << "Accuracy: " << accuracy << endl;

  return 0;
}

