/* file svm.h
 *  author
 *  date
 */

#ifndef SVM_H
#define SVM_H

#include <stdio.h>
#include <omp.h>
#include <map>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/complex.h>

#include "dataset.h"
#include "configuration.h"

typedef unsigned int uint;


template <typename T>
struct linear_index_to_row_index : public thrust::unary_function<T,T> {
  T C; // number of columns

  __host__ __device__
  linear_index_to_row_index(T C) : C(C) {}

  __host__ __device__
  T operator()(T i) {
    return i / C;
  }
};

class SVM {
 public:
  SVM() {
    learning_rate_ = 1e-3;
    optim_ = "SGD";
    momentum_ = 0.5f;
    regularization_ = 1e-5;
    batch_size_ = 200;
    n_epochs_ = 55000;
    dataset_handle_ = NULL;
    cublasCreate(&handle_);
  };

  ~SVM() {
    // this object should be delete ouside
    // if (dataset_handle_ != NULL) {
    //   delete dataset_handle_;
    // }
    dataset_handle_ = NULL;
  };

  int InitData(Dataset *_dataset = NULL,
                    float _regularization = 1e-5,
                    float _learning_rate = 1e-3,
                    const std::string& _model_file = "");

  int TrainData(uint _batch_size,
                uint _n_epochs,
                std::string out_model_name = "");

  virtual float Evaluate(float*, const std::vector<uint>&) = 0;
  virtual float LossFunc(thrust::device_vector<float>& _d_data,
                         std::vector<uint> labels,
                         thrust::device_vector<float>& _d_weights,
                         thrust::device_vector<float>& _d_score,
                         thrust::device_vector<float>& _row_indices,
                         thrust::device_vector<float>& _row_max) = 0;
  // input data is matrix of batch_size x dim
  // float SVMLossVectorized(float* data, std::vector<uint> labels, float* d_weights);
  // float SoftmaxLossVectorized(float* data, std::vector<uint> labels, float* d_weights);
  int DeviceNormalData(thrust::device_vector<float>& _d_data,
                 float _min,
                 float _max,
                 float new_max = 1.f,
                 float new_min = -1.f);

  int HostNormalData(float *data,
                      uint data_size,
                      float _min,
                      float _max,
                      float new_max = 1.f,
                      float new_min = -1.f);

  void PrintClass() {
    for(uint i = 0; i < num_class_; i++) {
      printf("%u -> %s \n", i, encode_label_[i].c_str());
    }
  }

 protected:
  thrust::device_vector<float> d_weights_;
  thrust::device_vector<uint> d_index_batch_;
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
  cublasHandle_t handle_;

};

class SoftmaxSVM : public SVM {
 public:
  float LossFunc(thrust::device_vector<float>& _d_data,
                 std::vector<uint> labels,
                 thrust::device_vector<float>& _d_weights,
                 thrust::device_vector<float>& _d_score,
                 thrust::device_vector<float>& _row_indices,
                 thrust::device_vector<float>& _row_max);

  float Evaluate(float*, const std::vector<uint>&);
};

#endif //end file SVM_H
