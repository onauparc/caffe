// Copyright 2014 BVLC and contributors.


#include <fstream>
#include <string>
#include <stdio.h>
#include <algorithm>

#include <glog/logging.h>
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/highgui/highgui.hpp"

using caffe::Datum;
using caffe::BlobProto;
using std::max;

using namespace std;

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc != 3) {
    LOG(ERROR) << "Usage: compute_opencv_mean file_list output_file";
    return 1;
  }
  char* file_list = argv[1];
  char* output_file = argv[2];

  //read image from file_list
  ifstream infile(file_list);
  int path, label;
  while (infile >> path >> label)
  {
    std::cout << "path " << path << "label " << label << std::endl;
    //cv::Mat image = cv::imread(path); 
  }
/*

  vector<cv::Mat> images(batch_size, image);
  cv::Mat image = cv::imread("/home/ladydisaster/libs/caffe/examples/images/cat.jpg"); // or cat.jpg
  int batch_size = 6;




  Datum datum;
  BlobProto sum_blob;
  int count = 0;
  // load first datum
  if (db_backend == "leveldb") {
    datum.ParseFromString(it->value().ToString());
  } else if (db_backend == "lmdb") {
    datum.ParseFromArray(mdb_value.mv_data, mdb_value.mv_size);
  } else {
    LOG(FATAL) << "Unknown db backend " << db_backend;
  }

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
  LOG(INFO) << "Starting Iteration";
  if (db_backend == "leveldb") {  // leveldb
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
      // just a dummy operation
      datum.ParseFromString(it->value().ToString());
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
  } else if (db_backend == "lmdb") {  // lmdb
    CHECK_EQ(mdb_cursor_get(mdb_cursor, &mdb_key, &mdb_value, MDB_FIRST),
        MDB_SUCCESS);
    do {
      // just a dummy operation
      datum.ParseFromArray(mdb_value.mv_data, mdb_value.mv_size);
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
    } while (mdb_cursor_get(mdb_cursor, &mdb_key, &mdb_value, MDB_NEXT)
        == MDB_SUCCESS);
  } else {
    LOG(FATAL) << "Unknown db backend " << db_backend;
  }

  if (count % 10000 != 0) {
    LOG(ERROR) << "Processed " << count << " files.";
  }
  for (int i = 0; i < sum_blob.data_size(); ++i) {
    sum_blob.set_data(i, sum_blob.data(i) / count);
  }
  // Write to disk
  LOG(INFO) << "Write to " << argv[2];
  WriteProtoToBinaryFile(sum_blob, argv[2]);

  // Clean up
  if (db_backend == "leveldb") {
    delete db;
  } else if (db_backend == "lmdb") {
    mdb_cursor_close(mdb_cursor);
    mdb_close(mdb_env, mdb_dbi);
    mdb_txn_abort(mdb_txn);
    mdb_env_close(mdb_env);
  } else {
    LOG(FATAL) << "Unknown db backend " << db_backend;
  }
*/
  return 0;
}
