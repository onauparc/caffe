#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
    << "The data and label should have the same number of instances";
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels())
    << "The data and label should have the same number of channels";
  CHECK_EQ(bottom[0]->height(), bottom[1]->height())
    << "The data and label should have the same height";
  CHECK_EQ(bottom[0]->width(), bottom[1]->width())
    << "The data and label should have the same width";
  // Top will contain:
  // top[0] = Sensitivity or Recall (TP/P),
  // top[1] = Specificity (TN/N),
  // top[2] = Harmonic Mean of Sens and Spec, (2/(P/TP+N/TN)),
  // top[3] = Precision (TP / (TP + FP)),
  // top[4] = F1 Score (2 TP / (2 TP + FP + FN)),
  top[0]->Reshape(1, 5, 1, 1);
}

template <typename Dtype>

void MultiLabelAccuracyLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  Dtype true_positive = 0;
  Dtype false_positive = 0;
  Dtype true_negative = 0;
  Dtype false_negative = 0;
  int count_pos = 0;
  int count_neg = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  // Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  int count = bottom[0]->count();

  /*for (int ind = 0; ind < count; ++ind) {
    // Accuracy
    int label = static_cast<int>(bottom_label[ind]);
    if (label > 0) {
    // Update Positive accuracy and count
      true_positive += (bottom_data[ind] >= 0);
      false_negative += (bottom_data[ind] < 0);
      count_pos++;
    }
    if (label < 0) {
    // Update Negative accuracy and count
      true_negative += (bottom_data[ind] < 0);
      false_positive += (bottom_data[ind] >= 0);
      count_neg++;
    }
  }*/
  
  
  /*LOG(ERROR) << "ATTENZIONE, LA MULTILABEL ACCURACY E' CALCOLATA SUPPONENDO CHE LE CLASSI SIANO 41, CAMBIARE QUESTO VALORE IN MULTI_LABEL_ACCURACY_LAYER.CPP O RENDERLO DINAMICO";
  int num_labels_for_example = 41;
  for (int ind = 0; ind < (count/num_labels_for_example); ++ind) {
    //controllo prima se è pettorina/non pettorina
    int index_last_label = (ind*num_labels_for_example) + num_labels_for_example - 1;
    int last_label = static_cast<int>(bottom_label[index_last_label]);
    if(last_label < 0) //gt non pettorina
    {
      count_neg++;
      if(bottom_data[index_last_label] < 0) //predict non pettorina
      {
        true_negative += 1;
      }
      else if(bottom_data[index_last_label] >= 0)//predict pettorina
      {
        false_positive += 1;
      }
    }
    else if(last_label >= 0) //gt pettorina
    {
      if(last_label == 0)
        LOG(ERROR) << "LAST LABEL E' esattamente zero!";
      
      count_pos++;
      int index_next_labels = (ind*num_labels_for_example);
      int num_0 = std::distance(&bottom_data[index_next_labels], std::max_element(&bottom_data[index_next_labels],&bottom_data[index_next_labels] + 10));
      int num_1 = std::distance(&bottom_data[index_next_labels]+10, std::max_element(&bottom_data[index_next_labels]+10,&bottom_data[index_next_labels] + 20));
      int num_2 = std::distance(&bottom_data[index_next_labels]+20, std::max_element(&bottom_data[index_next_labels]+20,&bottom_data[index_next_labels] + 30));
      int num_3 = std::distance(&bottom_data[index_next_labels]+30, std::max_element(&bottom_data[index_next_labels]+30,&bottom_data[index_next_labels] + 40));
      //std::cout<<num_0<<num_1<<num_2<<num_3<<std::endl;
      
      if(bottom_label[index_next_labels + num_0] >= 0 && bottom_label[index_next_labels + 10 + num_1] >= 0 && bottom_label[index_next_labels + 20 + num_2] >= 0 && bottom_label[index_next_labels + 30 + num_3] >= 0)
      {
        true_positive += 1;
      }
      else
      {
        false_negative += 1;
      }
      
    }
    
  }*/
  LOG(ERROR) << "ATTENZIONE, LA MULTILABEL ACCURACY E' CALCOLATA SUPPONENDO CHE LE CLASSI SIANO 40, CAMBIARE QUESTO VALORE IN MULTI_LABEL_ACCURACY_LAYER.CPP O RENDERLO DINAMICO";
  int num_labels_for_example = 40;
  for (int ind = 0; ind < (count/num_labels_for_example); ++ind) {
    //controllo prima se è pettorina/non pettorina
    int index_first_label = (ind*num_labels_for_example);
    int index_last_label = (ind*num_labels_for_example) + num_labels_for_example - 1;
    int first_label = static_cast<int>(bottom_label[index_first_label]);
    count_neg = 1;
    if(first_label == 0) //gt non pettorina -- tutte le label della non pettorina sono 0, controllo solo la prima
    {
      /*
      count_neg++;
      bool found_false_negative = true;
      for(int index = index_first_label; index<=index_last_label; index++)
      {
        if(bottom_data[index] != 0) //predict non pettorina
        {
          false_positive += 1;
          found_false_negative = false;
          break;
        }
      }
      if(found_false_negative == true) //questo è vero se tutti le cifre sono esattamente 0
        true_negative += 1; //predict pettorina
        */
    }
    else //gt pettorina*/
    {
            
      count_pos++;
      int num_0 = std::distance(&bottom_data[index_first_label], std::max_element(&bottom_data[index_first_label],&bottom_data[index_first_label] + 10));
      int num_1 = std::distance(&bottom_data[index_first_label]+10, std::max_element(&bottom_data[index_first_label]+10,&bottom_data[index_first_label] + 20));
      int num_2 = std::distance(&bottom_data[index_first_label]+20, std::max_element(&bottom_data[index_first_label]+20,&bottom_data[index_first_label] + 30));
      int num_3 = std::distance(&bottom_data[index_first_label]+30, std::max_element(&bottom_data[index_first_label]+30,&bottom_data[index_first_label] + 40));
      //std::cout<<num_0<<num_1<<num_2<<num_3<<std::endl;
      
      if(bottom_label[index_first_label + num_0] >= 0 && bottom_label[index_first_label + 10 + num_1] >= 0 && bottom_label[index_first_label + 20 + num_2] >= 0 && bottom_label[index_first_label + 30 + num_3] >= 0)
      {
        true_positive += 1;
      }
      else
      {
        false_negative += 1;
      }
      
    }
    
  }
  Dtype sensitivity = (count_pos > 0)? (true_positive / count_pos) : 0;
  Dtype specificity = (count_neg > 0)? (true_negative / count_neg) : 0;
  Dtype harmmean = ((count_pos + count_neg) > 0)?
    2 / (count_pos / true_positive + count_neg / true_negative) : 0;
  Dtype precission = (true_positive > 0)?
    (true_positive / (true_positive + false_positive)) : 0;
  Dtype f1_score = (true_positive > 0)?
    2 * true_positive /
    (2 * true_positive + false_positive + false_negative) : 0;

  LOG(ERROR) << "true_positive/count_pos: " << true_positive<<"/"<<count_pos;
  LOG(ERROR) << "true_negative/count_neg: " << true_negative<<"/"<<count_neg;
  DLOG(INFO) << "Sensitivity: " << sensitivity;
  DLOG(INFO) << "Specificity: " << specificity;
  DLOG(INFO) << "Harmonic Mean of Sens and Spec: " << harmmean;
  DLOG(INFO) << "Precission: " << precission;
  DLOG(INFO) << "F1 Score: " << f1_score;
  top[0]->mutable_cpu_data()[0] = sensitivity;
  top[0]->mutable_cpu_data()[1] = specificity;
  top[0]->mutable_cpu_data()[2] = harmmean;
  top[0]->mutable_cpu_data()[3] = precission;
  top[0]->mutable_cpu_data()[4] = f1_score;
}

/*template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  Dtype true_positive = 0;
  Dtype false_positive = 0;
  Dtype true_negative = 0;
  Dtype false_negative = 0;
  int count_pos = 0;
  int count_neg = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  // Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  int count = bottom[0]->count();

  for (int ind = 0; ind < count; ++ind) {
    // Accuracy
    int label = static_cast<int>(bottom_label[ind]);
    if (label > 0) {
    // Update Positive accuracy and count
      true_positive += (bottom_data[ind] >= 0);
      false_negative += (bottom_data[ind] < 0);
      count_pos++;
    }
    if (label < 0) {
    // Update Negative accuracy and count
      true_negative += (bottom_data[ind] < 0);
      false_positive += (bottom_data[ind] >= 0);
      count_neg++;
    }
  }
  Dtype sensitivity = (count_pos > 0)? (true_positive / count_pos): 0;
  Dtype specificity = (count_neg > 0)? (true_negative / count_neg): 0;
  Dtype harmmean = ((count_pos + count_neg) > 0)?
    2 / (count_pos / true_positive + count_neg / true_negative) : 0;
  Dtype precission = (true_positive > 0)?
    (true_positive / (true_positive + false_positive)): 0;
  Dtype f1_score = (true_positive > 0)?
    2 * true_positive /
    (2 * true_positive + false_positive + false_negative) : 0;

  DLOG(INFO) << "Sensitivity: " << sensitivity;
  DLOG(INFO) << "Specificity: " << specificity;
  DLOG(INFO) << "Harmonic Mean of Sens and Spec: " << harmmean;
  DLOG(INFO) << "Precission: " << precission;
  DLOG(INFO) << "F1 Score: " << f1_score;

  top[0]->mutable_cpu_data()[0] = sensitivity;
  top[0]->mutable_cpu_data()[1] = specificity;
  top[0]->mutable_cpu_data()[2] = harmmean;
  top[0]->mutable_cpu_data()[3] = precission;
  top[0]->mutable_cpu_data()[4] = f1_score;
}*/

INSTANTIATE_CLASS(MultiLabelAccuracyLayer);
REGISTER_LAYER_CLASS(MULTI_LABEL_ACCURACY, MultiLabelAccuracyLayer);
}  // namespace caffe
