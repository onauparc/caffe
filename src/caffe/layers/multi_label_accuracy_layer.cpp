// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

#include <iostream>     // std::cout
#include <iterator>     // std::distance
#include <list>         // std::list
using std::max;


namespace caffe {

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::SetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
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
  // top[2] = Harmonic Mean of Sens and Spec, (2/(P/TP+N/TN))
  // top[3] = Precision (TP / (TP + FP))
  // top[4] = F1 Score (2 TP / (2 TP + FP + FN))
  (*top)[0]->Reshape(1, 5, 1, 1);
}

template <typename Dtype>
Dtype MultiLabelAccuracyLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
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
  
  
  LOG(ERROR) << "ATTENZIONE, LA MULTILABEL ACCURACY E' CALCOLATA SUPPONENDO CHE LE CLASSI SIANO 41, CAMBIARE QUESTO VALORE IN MULTI_LABEL_ACCURACY_LAYER.CPP O RENDERLO DINAMICO";
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
      
      /*
      count_pos++;
      bool found_false_negative = false;
      for(int ind_label_pett = 0; ind_label_pett < num_labels_for_example - 1; ind_label_pett++)
      {
        int index_next_label = (ind*num_labels_for_example) + ind_label_pett;
        int label = static_cast<int>(bottom_label[index_next_label]);
        
        if(label > 0)
        {
          if(bottom_data[index_next_label] < 0) //la label è discordante, quindi il numero cercato è sicuramente diverso
          {
            found_false_negative = true;
            break;
          }
        }
        else if(label < 0)
        {
          if(bottom_data[index_next_label] >= 0) //la label è discordante, però POTREBBE ANCORA ESSERE RECUPERATO IL NUMERO CORRETTO CON L'ARGMAX
          {
            found_false_negative = true;
            break;
          }
        }
        
      }
      if(found_false_negative == true)
      {
        false_negative += 1;
      }
      else //tutti i numeri sono corretti
      {
        true_positive += 1;
      }
      */
      
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
  (*top)[0]->mutable_cpu_data()[0] = sensitivity;
  (*top)[0]->mutable_cpu_data()[1] = specificity;
  (*top)[0]->mutable_cpu_data()[2] = harmmean;
  (*top)[0]->mutable_cpu_data()[3] = precission;
  (*top)[0]->mutable_cpu_data()[4] = f1_score;

  // MultiLabelAccuracy should not be used as a loss function.
  return Dtype(0);
}

INSTANTIATE_CLASS(MultiLabelAccuracyLayer);

}  // namespace caffe
