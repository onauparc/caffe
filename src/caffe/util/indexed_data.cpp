#include "caffe/util/indexed_data.hpp"

namespace caffe {
   template <typename Dtype>
   SimpleSingleIndexedTextFile<Dtype>::SimpleSingleIndexedTextFile(const std::string& file_name) {
    std::ifstream input(file_name.c_str());
    Dtype tmp;
    while (input >> tmp)
      data_.push_back(tmp);
  }
  template <typename Dtype>
  index_type SimpleSingleIndexedTextFile<Dtype>::read(index_type index,
        Dtype* out, index_type length) {
    if (index >= data_.size())
      return 0;
      

    if (length > 0)
    {
      for (int c = 0; c < length; ++c) {
        *out = data_[(index*length)+c];
        out++;
      }
    }
    return 1;
  }
INSTANTIATE_CLASS(SimpleSingleIndexedTextFile);
}  // namespace caffe
