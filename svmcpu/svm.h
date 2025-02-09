/* file svm.h
 *  author
 *  date
 */

#ifndef SVM_H
#define SVM_H

#include <stdio.h>
#include <omp.h>
#include <map>

#include "dataset.h"

typedef unsigned int uint;
template<class T>
int MultipleMatrix(T *mx_a, size_t a_row, size_t a_cols,
                   T *mx_b, size_t b_row, size_t b_cols,
                   T *res);
class SVM {
 public:
  SVM() {
    weights_ = NULL;
    learning_rate_ = 1e-3;
    optim_ = "SGD";
    momentum_ = 0.5f;
    regularization_ = 1e-5;
    batch_size_ = 200;
    n_epochs_ = 55000;
    dataset_handle_ = NULL;
  };

  ~SVM() {
    if (weights_ != NULL) {
      delete weights_;
    }
    // this object should be delete ouside
    // if (dataset_handle_ != NULL) {
    //   delete dataset_handle_;
    // }
    weights_ = NULL;
    dataset_handle_ = NULL;
  };

  int InitData(Dataset *_dataset = NULL,
                    float _regularization = 1e-5,
                    float _learning_rate = 1e-3,
                    const std::string& _model_file = "");

  int TrainData(uint _batch_size,
                uint _n_epochs,
                std::string out_model_name = "");
  //
  // float EvaluateAccData(float*,
  //                  const std::vector<uint>&) const;
  // float EvaluateAccSoftmaxData(float*,
  //                  const std::vector<uint>&);
  virtual float Evaluate(float*, const std::vector<uint>&) = 0;
  virtual float LossFunc(float* data, std::vector<uint> labels, float* d_weights) = 0;
  // input data is matrix of batch_size x dim
  // float SVMLossVectorized(float* data, std::vector<uint> labels, float* d_weights);
  // float SoftmaxLossVectorized(float* data, std::vector<uint> labels, float* d_weights);
  int NormalData(float *data,
                 uint data_size,
                 float _min,
                 float _max,
                 float new_max = 1.f,
                 float new_min = -1.f);

  void PrintClass() {
    for(uint i = 0; i <num_class_; i++) {
      printf("%u -> %s \n", i, encode_label_[i].c_str());
    }
  }

 protected:
  float* weights_;
  uint num_class_;
  uint dim_;
  float learning_rate_;
  std::string optim_;
  float momentum_;
  float regularization_;
  uint batch_size_;
  uint n_epochs_;
  std::map<uint, std::string> encode_label_;
  Dataset *dataset_handle_;
};

class SoftmaxSVM : public SVM {
 public:
  float LossFunc(float* data, std::vector<uint> labels, float* d_weights);
  float Evaluate(float*, const std::vector<uint>&);
};

class VectorizedSVM : public SVM {
 public:
  float LossFunc(float* data, std::vector<uint> labels, float* d_weights);
  float Evaluate(float*, const std::vector<uint>&);
};

#endif //end file SVM_H
