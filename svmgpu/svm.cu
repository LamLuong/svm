#include "svm.h"
#include <unistd.h>
#include <chrono>
#include <random>
#include <algorithm>
#include <climits>

#include "utils.h"

std::chrono::high_resolution_clock m_clock;
uint64_t GetTimeNanoSec();

__global__ void
transform_max_exp_array(float *A, float* B, uint n_col, uint b_size, int a_size);

__global__ void
transform_div_array(float *A, float* B, uint n_col, uint b_size, int a_size);

__global__ void
decreasement_label(float *A, uint* B, uint n_col, uint b_size, int a_size);
int SVM::InitData(Dataset *_dataset,
                  float _regularization,
                  float _learning_rate,
                  const std::string& _model_file) {

  if (!_model_file.empty()) {
    printf("Init for mode Testing \n");
    std::ifstream weights_data;
    weights_data.open(_model_file, std::ios::in | std::ios::binary);

    if (!weights_data.good()) {
      printf("Cannot open weights_data file \n");
      return -1;
    }
    weights_data.read((char*)&num_class_, sizeof(uint));
    weights_data.read((char*)&dim_, sizeof(uint));

    uint lengthlabel = 0;
    weights_data.read((char*)&lengthlabel, sizeof(uint));
    std::string list_label;
    list_label.resize(lengthlabel);
    weights_data.read((char*)list_label.c_str(), lengthlabel);

    uint initial_pos = 0;
    size_t pos = list_label.find(" ");
    uint label_code = 0;

    while (pos != std::string::npos) {
      encode_label_[label_code] = list_label.substr(initial_pos, pos - initial_pos);
      label_code++;
      initial_pos = pos + 1;
      pos = list_label.find(" ", initial_pos);
    }

    d_weights_.resize(dim_ * num_class_);
    float * h_weights = new float[dim_ * num_class_];
    weights_data.read((char*)h_weights, dim_ * num_class_ * sizeof(float));
    weights_data.close();
    thrust::copy(h_weights, h_weights + dim_ * num_class_, d_weights_.begin());

    if (h_weights)
      delete h_weights;
    h_weights = NULL;
    return 1;
  }

  if (_dataset == NULL) {
    printf("You need to input dataset \n");
    return -1;
  }

  printf("Init for mode Training \n");
  dataset_handle_ = _dataset;
  regularization_ = _regularization;
  learning_rate_ = _learning_rate;

  if (dataset_handle_->GetNumClass() <= 0 ||
      dataset_handle_->GetFeatureSize() <= 0) {

    printf("Invalid number of class of dimensions \n");
    return -1;
  }

  num_class_ = dataset_handle_->GetNumClass();
  dim_ = dataset_handle_->GetFeatureSize();

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> distrib(0.f, 1.f);

  float* h_weights = new float[dim_ * num_class_]; //weights_ is matrix of dim * num_class

  #pragma omp parallel for
  for (uint i = 0; i < dim_; ++i) {
    for (uint j = 0; j < num_class_; ++j) {
      *(h_weights + i *  num_class_ + j) = 0.001 * std::abs(distrib(gen));
    }
  }

  d_weights_.resize(dim_ * num_class_);
  thrust::copy(h_weights, h_weights + dim_ * num_class_, d_weights_.begin());

  if (h_weights)
    delete h_weights;
  h_weights = NULL;

  return 0;
}

int SVM::TrainData(uint _batch_size,
                   uint _n_epochs,
                   std::string out_model_name) {
  if (dataset_handle_ == NULL) {
    return -1;
  }

  batch_size_ = _batch_size;
  n_epochs_ = _n_epochs;

  std::random_device random_device;
  std::mt19937 gen_rand32(random_device());
  std::uniform_int_distribution<> dist(0, dataset_handle_->GetNumSample() - 1);

  auto gen_random_idx = [&dist, &gen_rand32]() {
                          return dist(gen_rand32);
                         };

  thrust::device_vector<float> d_gradient(dim_ * num_class_);
  thrust::device_vector<float> d_batch_data(batch_size_ * dim_);
  thrust::device_vector<float> d_score(batch_size_ * num_class_);
  thrust::device_vector<float> d_row_indices(batch_size_);
  thrust::device_vector<float> d_row_max(batch_size_);
  d_index_batch_.resize(batch_size_);
  std::vector<uint>h_index_batch;
  h_index_batch.resize(batch_size_);

  float d_learning_rate = learning_rate_;
  uint64_t start_time = GetTimeNanoSec();
  for (uint i = 0; i < n_epochs_; ++i) {
    std::vector<uint> index_from_batch;
    index_from_batch.resize(batch_size_);

    std::generate(std::begin(index_from_batch),
                  std::end(index_from_batch),
                  gen_random_idx);

    float* h_batch_data = new float[_batch_size * dim_];

    for(uint j = 0; j < _batch_size; ++j) {
      dataset_handle_->ReadData(h_batch_data + j * dim_,
        dataset_handle_->GetLabelFromIndex(index_from_batch[j]).position_);

      h_index_batch[j] = j * num_class_ +
           dataset_handle_->GetLabelFromIndex(index_from_batch[j]).label_;
    }

    thrust::copy(h_batch_data,
                 h_batch_data + _batch_size * dim_,
                 d_batch_data.begin());

     if (h_batch_data != NULL) {
       delete h_batch_data;
     }
     h_batch_data = NULL;

    DeviceNormalData(d_batch_data,
               dataset_handle_->GetMinVal(),
               dataset_handle_->GetMaxVal());

    thrust::copy(h_index_batch.begin(),
                 h_index_batch.end(),
                 d_index_batch_.begin());

    float curr_loss = 0.f;
    curr_loss = LossFunc(d_batch_data, index_from_batch,
                         d_gradient, d_score,
                         d_row_indices, d_row_max);

    if (i % 100 == 0) {
      printf("iteration %u / %u: loss %f  time: %lu (ms)\n",
              i,
              n_epochs_,
              curr_loss,
              (GetTimeNanoSec() - start_time) / 1000000);
      start_time = GetTimeNanoSec();
    }


    thrust::transform(d_weights_.begin(),
                      d_weights_.end(),
                      d_gradient.begin(),
                      d_weights_.begin(),
                      [=] __device__ (float a, float b) {
                        return a - d_learning_rate * b;
                      });
    usleep(Configuration::GetInstance()->n_sleep_);
  }

  if (out_model_name.empty()) {
    out_model_name = "./weights.bin";
  }
  std::ofstream bin_weights_file(out_model_name, std::ios::out | std::ios::binary);
  bin_weights_file.write((char *)&num_class_, sizeof(uint));
  bin_weights_file.write((char *)&dim_, sizeof(uint));
//  bin_weights_file.write((char *)&min_value_, sizeof(float));
//  bin_weights_file.write((char *)&max_value_, sizeof(float));

  std::string encode_class = dataset_handle_->GetEncodingClass();
  uint encode_class_length = encode_class.length();
  bin_weights_file.write((char *)&encode_class_length, sizeof(uint));
  bin_weights_file.write((char *)encode_class.c_str(), encode_class_length);

  float* h_weights = new float[dim_ * num_class_];
  thrust::copy(d_weights_.begin(), d_weights_.end(), h_weights);

  bin_weights_file.write((char *)h_weights, dim_ * num_class_ * sizeof(float));

  delete[] h_weights;
  bin_weights_file.close();
  printf("Model saved in %s \n", out_model_name.c_str());
  return 0;
}

float SoftmaxSVM::LossFunc(thrust::device_vector<float>& _d_data,
                           std::vector<uint> labels,
                           thrust::device_vector<float>& _d_gradient,
                           thrust::device_vector<float>& _d_score,
                           thrust::device_vector<float>& _row_indices,
                           thrust::device_vector<float>& _row_max) {
   float loss = 0.f;
   const float alpha = 1.0f;
   const float beta  = 0.0f;

   cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
              num_class_,
              batch_size_,
              dim_,
              &alpha,
              thrust::raw_pointer_cast(d_weights_.data()), num_class_,
              thrust::raw_pointer_cast(_d_data.data()), dim_,
              &beta,
              thrust::raw_pointer_cast(_d_score.data()), num_class_);

  thrust::reduce_by_key(
          thrust::make_transform_iterator(thrust::counting_iterator<int>(0),
            linear_index_to_row_index<int>(num_class_)),
          thrust::make_transform_iterator(thrust::counting_iterator<int>(0),
            linear_index_to_row_index<int>(num_class_)) + (num_class_ * batch_size_),
          _d_score.begin(),
          _row_indices.begin(),
          _row_max.begin(),
          thrust::equal_to<int>(),
          thrust::maximum<float>());

  transform_max_exp_array<<<batch_size_ * num_class_ / 1024 + 1, 1024>>>(
                        thrust::raw_pointer_cast(_d_score.data()),
                        thrust::raw_pointer_cast(_row_max.data()),
                        num_class_, batch_size_, batch_size_ * num_class_);

  thrust::reduce_by_key(
          thrust::make_transform_iterator(thrust::counting_iterator<int>(0),
            linear_index_to_row_index<int>(num_class_)),
          thrust::make_transform_iterator(thrust::counting_iterator<int>(0),
            linear_index_to_row_index<int>(num_class_)) + (num_class_ * batch_size_),
          _d_score.begin(),
          _row_indices.begin(),
          _row_max.begin(),
          thrust::equal_to<int>(),
          thrust::plus<float>());

  transform_div_array<<<batch_size_ * num_class_ / 1024 + 1, 1024>>>(
                        thrust::raw_pointer_cast(_d_score.data()),
                        thrust::raw_pointer_cast(_row_max.data()),
                        num_class_, batch_size_, batch_size_ * num_class_);

  float sum = thrust::transform_reduce(
    thrust::make_permutation_iterator(_d_score.begin(), d_index_batch_.begin()),
    thrust::make_permutation_iterator(_d_score.begin(), d_index_batch_.end()),
    [=] __device__ (float a) ->float {
      return -1 * log(a);
    }, 0.f,
    thrust::plus<float>());

  loss = loss + sum;

  loss = loss / batch_size_;
  loss += 0.5 * regularization_ * thrust::transform_reduce(
                                      d_weights_.begin(),
                                      d_weights_.end(),
                                      [=] __device__ (float a) ->float {
                                         return a * a;
                                      }, 0.f, thrust::plus<float>());

/*  thrust::transform(
    thrust::make_permutation_iterator(_d_score.begin(), d_index_batch_.begin()),
    thrust::make_permutation_iterator(_d_score.begin(), d_index_batch_.end()),
    _d_score.begin(),
    [=] __device__ (float a) ->float {
      return a - 1;
    });*/
  decreasement_label<<<batch_size_ * num_class_ / 1024 + 1, 1024>>>(
                        thrust::raw_pointer_cast(_d_score.data()),
                        thrust::raw_pointer_cast(d_index_batch_.data()),
                        num_class_,
                        batch_size_,
                        batch_size_ * num_class_);

  cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T,
             num_class_,
             dim_,
             batch_size_,
             &alpha,
             thrust::raw_pointer_cast(_d_score.data()), num_class_,
             thrust::raw_pointer_cast(_d_data.data()), dim_,
             &beta,
             thrust::raw_pointer_cast(_d_gradient.data()), num_class_);

   float d_batch_size = batch_size_;
   thrust::transform(_d_gradient.begin(),
                     _d_gradient.end(),
                     _d_gradient.begin(),
                     [=] __device__ (float a) ->float {
                        return a / d_batch_size;
                     });

  return loss;
}

float SoftmaxSVM::Evaluate(float* _input_data,
                      const std::vector<uint>& _real_label) {
  if (d_weights_.size() == 0 || _input_data == NULL) {
    printf("Invalid weights or data \n");
    return -1.f;
  }

  uint n_sample = _real_label.size();
  thrust::device_vector<float> d_input_data(n_sample * dim_);
  thrust::copy(_input_data, _input_data + n_sample * dim_, d_input_data.begin());

  thrust::device_vector<float> d_result(n_sample * num_class_);

  const float alpha = 1.0f;
  const float beta  = 0.0f;

  cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
             num_class_,
             n_sample,
             dim_,
             &alpha,
             thrust::raw_pointer_cast(d_weights_.data()), num_class_,
             thrust::raw_pointer_cast(d_input_data.data()), dim_,
             &beta,
             thrust::raw_pointer_cast(d_result.data()), num_class_);

  thrust::transform(d_result.begin(),
                    d_result.end(),
                    d_result.begin(),
                    [=] __device__ (float a) ->float {
                      return exp(a);
                    });

  float *h_result = new float[n_sample * num_class_];
  thrust::copy(d_result.begin(), d_result.end(), h_result);

  std::vector<uint> _predic_label;
  for (uint i = 0; i < n_sample; ++i) {
    float * max_pos = std::max_element(h_result + i * num_class_, h_result + (i + 1) * num_class_);
    uint pre_label = std::distance(h_result + i * num_class_, max_pos);

    _predic_label.push_back(pre_label);
    float curr_total = std::accumulate(h_result + i * num_class_,
                                       h_result + (i + 1) * num_class_,
                                       0.f);
  }

  uint count_diff = 0;
  for (uint i = 0; i < n_sample; ++i) {
    if (_predic_label[i] != _real_label[i]) {
      count_diff++;
    }
  }

  if (h_result != NULL) {
    delete h_result;
  }
  h_result = NULL;

  return 1.f - (float)count_diff / n_sample;
 }

int SVM::DeviceNormalData(thrust::device_vector<float>& _d_data,
                    float _min,
                    float _max,
                    float new_max,
                    float new_min) {
  if (_d_data.size() <= 0) {
    printf("Invalid data \n");
    return -1;
  }

  float new_range = new_max - new_min;

  thrust::transform(_d_data.begin(),
                    _d_data.end(),
                    _d_data.begin(),
                    [=] __device__ (float a) ->float {
                       return (a - _min) * new_range / (_max - _min) + new_min;
                    });

  return 0;
}

int SVM::HostNormalData(float *data,
                    uint data_size,
                    float _min,
                    float _max,
                    float new_max,
                    float new_min) {
  if (data == NULL) {
    printf("Invalid data \n");
    return -1;
  }

  float new_range = new_max - new_min;

  #pragma omp parallel for
  for(uint i = 0; i < data_size; ++i) {
    data[i] =
      (data[i] - _min) * new_range / (_max - _min) + new_min;
  }

  return 0;
}

__global__ void
transform_max_exp_array(float *A, float* B, uint a_col, uint b_size, int a_size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < a_size) {
        uint b_index = i / a_col;
        if (b_index < b_size) {
          A[i] = exp(A[i]  - B[b_index]);
        }
    }
}

__global__ void
transform_div_array(float *A, float* B, uint a_col, uint b_size, int a_size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < a_size) {
        uint b_index = i / a_col;
        if (b_index < b_size) {
          A[i] = A[i] / B[b_index];
        }
    }
}

__global__ void
decreasement_label(float *A, uint* B, uint a_col, uint b_size, int a_size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < a_size) {
      uint b_index = i / a_col;

      if (b_index < b_size && i % a_col == 0) {
        A[B[b_index]] = A[B[b_index]] - 1;
      }
  }
  __syncthreads();
}

uint64_t GetTimeNanoSec() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>
              (m_clock.now().time_since_epoch()).count();
}
