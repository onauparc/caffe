#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ConditionalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  concat_dim_ = this->layer_param_.concat_param().concat_dim();
  CHECK_GE(concat_dim_, 0) <<
    "concat_dim should be >= 0";
  CHECK_LE(concat_dim_, 1) <<
    "For now concat_dim <=1, it can only concat num and channels";
}

template <typename Dtype>
void ConditionalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Initialize with the first blob.
  count_ = bottom[0]->count();
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  for (int i = 1; i < bottom.size(); ++i) {
    count_ += bottom[i]->count();
    if (concat_dim_== 0) {
      num_ += bottom[i]->num();
    } else if (concat_dim_ == 1) {
      channels_ += bottom[i]->channels();
    } else if (concat_dim_ == 2) {
      height_ += bottom[i]->height();
    } else if (concat_dim_ == 3) {
      width_ += bottom[i]->width();
    }
  }
  top[0]->Reshape(num_, channels_, height_, width_);
  CHECK_EQ(count_, top[0]->count());
}

void softmax_casareccio(std::vector<float> y, std::vector<float>& value)
{
  //Vector y = mlp(x); // output of the neural network without softmax activation function
  double ysum = 0;
  for(int f = 0; f < y.size(); f++)
    ysum += exp((double)y[f]);
  
  for(int i = 0; i<value.size(); i++)
  {
    value[i] = (float)(exp((double)value[i])/ysum);
  }
}

template <typename Dtype>
void ConditionalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
        
        
        
  //const float* bottom_data_pettorina = bottom[0]->cpu_data();
  //const float* bottom_data_pool1 = bottom[1]->cpu_data();
  const Dtype* bottom_data_pettorina = bottom[0]->cpu_data();
  const Dtype* bottom_data_pool1 = bottom[1]->cpu_data();
  //const float* bottom_label = result[1]->cpu_data();
  vector<float> indicesTokeep;
  int max = -1;
  float max_prob = -FLT_MAX;

  int num_labels_pettorine = 2;
  int num_images = bottom[0]->num();
  LOG(ERROR) << "num_images: "<<num_images;
  int ind_pettorine = 0;
  for (int z = 0; z < num_images; ++z) {
    
    ind_pettorine = num_labels_pettorine*z;
    
    vector<float> is_pett;
    
    is_pett.push_back((float)bottom_data_pettorina[ind_pettorine]);
    is_pett.push_back((float)bottom_data_pettorina[ind_pettorine + 1]);
    LOG(ERROR) <<" bottom_data_pettorina[ind_pettorine]: "<<(float)bottom_data_pettorina[ind_pettorine];
    LOG(ERROR) <<" bottom_data_pettorina[ind_pettorine+1]: "<<(float)bottom_data_pettorina[ind_pettorine+1];
    
    softmax_casareccio(is_pett, is_pett);
    float max_isPett = *(std::max_element(is_pett.begin(),is_pett.end()));
    float max_isPett_index = distance(is_pett.begin(), max_element(is_pett.begin(), is_pett.end()));
    
    stringstream ss;
    float THRESH_PETT = 0.5;
     if(max_isPett >= THRESH_PETT && max_isPett_index == 1)
     {
      indicesTokeep.push_back(z);
      LOG(ERROR) << z<<") PRESO - max_isPett_index: "<<max_isPett_index<<"(PETTORINA) - value: "<<max_isPett;
     }
     else
      LOG(ERROR) << z<<") SCARTATO - max_isPett_index: "<<max_isPett_index<<"(NON PETTORINA) - value: "<<max_isPett;
   
  }
  
  top[0]->Reshape(indicesTokeep.size(), top[0]->channels(), top[0]->height(), top[0]->width());     
  Dtype* top_data = top[0]->mutable_cpu_data();
  int index = 0;
  for(size_t n = 0; n<indicesTokeep.size(); n++, index++)
  {
    int offset = indicesTokeep[n];
    top_data[index] = bottom_data_pool1[offset];
  }

}

template <typename Dtype>
void ConditionalLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(ConditionalLayer);
#endif

INSTANTIATE_CLASS(ConditionalLayer);
REGISTER_LAYER_CLASS(CONDITIONAL, ConditionalLayer);
}  // namespace caffe
