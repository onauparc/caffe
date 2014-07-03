// Copyright 2014 BVLC and contributors.


#include <fstream>
#include <iostream>
#include <sstream>

#include <string>
#include <vector>
#include <stdio.h>
#include <algorithm>

#include <glog/logging.h>
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using caffe::Datum;
using caffe::BlobProto;
using std::max;


int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc != 3) {
    LOG(ERROR) << "Usage: compute_opencv_mean file_list output_file";
    return 1;
  }
  char* file_list = argv[1];
  char* output_file = argv[2];

  std::ifstream infile(file_list, std::ifstream::in);
  std::vector<int> labels;
  std::vector<string> paths;

  //read labes and images
  std::string line;
  int check = false;
  cv::Mat image;
  int width;
  int height;
  string a_path;
  int a_label;
  while (std::getline(infile, line))
  {
    std::stringstream ss(line); 
    string path;
    int label;
    ss >> path;
    ss >> label;
    labels.push_back(label);
    paths.push_back(path);
    
    if (!check)
    {
      check = !check;
      image = cv::imread(path);
      width = image.cols;
      height = image.rows;
      a_path = path;
      a_label = label;
    }
  }

  Datum datum;
  ReadImageToDatum(a_path, 1, width, height, &datum);

  BlobProto sum_blob;
  sum_blob.set_num(1);
  sum_blob.set_channels(datum.channels());
  sum_blob.set_height(datum.height());
  sum_blob.set_width(datum.width());
  const int data_size = datum.channels() * datum.height() * datum.width();
  int size_in_datum = std::max<int>(datum.data().size(),
                                    datum.float_data_size());
  for (int i = 0; i < size_in_datum; ++i) {
    sum_blob.add_data(0.);
  }
  int count = 0;
  for (std::vector<std::string>::iterator it = paths.begin() ; it != paths.end(); ++it)
  {
    ReadImageToDatum(*it, labels[count], width, height, &datum);

    const string& data = datum.data();
    size_in_datum = std::max<int>(datum.data().size(),
        datum.float_data_size());
    CHECK_EQ(size_in_datum, data_size) << "Incorrect data field size " <<
        size_in_datum;
    if (data.size() != 0) {
      for (int i = 0; i < size_in_datum; ++i) {
        sum_blob.set_data(i, sum_blob.data(i) + (uint8_t)data[i]);
      }
    } else {
      for (int i = 0; i < size_in_datum; ++i) {
        sum_blob.set_data(i, sum_blob.data(i) +
            static_cast<float>(datum.float_data(i)));
      }
    }
    ++count;
    if (count % 10000 == 0) {
      LOG(ERROR) << "Processed " << count << " files.";
    }
  }
  for (int i = 0; i < sum_blob.data_size(); ++i) {
    sum_blob.set_data(i, sum_blob.data(i) / count);
  }
  // Write to disk
  LOG(INFO) << "Write to " << output_file;
  WriteProtoToBinaryFile(sum_blob, output_file);

  return 0;
}
