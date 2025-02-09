#include "svm.h"

#include <random>
#include <algorithm>
#include <climits>

#include "utils.h"

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
//    weights_data.read((char*)&min_value_, sizeof(float));
//    weights_data.read((char*)&max_value_, sizeof(float));

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

    weights_ = new float[dim_ * num_class_];
    weights_data.read((char*)weights_, dim_ * num_class_ * sizeof(float));
    weights_data.close();
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
//  min_value_ = dataset_handle_->GetMinVal();
//  max_value_ = dataset_handle_->GetMaxVal();

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> distrib(0.f, 1.f);

  weights_ = new float[dim_ * num_class_]; //weights_ is matrix of dim * num_class

  #pragma omp parallel for
  for (uint i = 0; i < dim_; ++i) {
    for (uint j = 0; j < num_class_; ++j) {
      *(weights_ + i *  num_class_ + j) = 0.001 * std::abs(distrib(gen));
    }
  }

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

  for (uint i = 0; i < n_epochs_; ++i) {
    std::vector<uint> index_from_batch;
    index_from_batch.resize(batch_size_);

    std::generate(std::begin(index_from_batch),
                  std::end(index_from_batch),
                  gen_random_idx);

    float* batch_data = new float[_batch_size * dim_];

    for(uint j = 0; j < _batch_size; ++j) {
      dataset_handle_->ReadData(batch_data + j * dim_,
        dataset_handle_->GetLabelFromIndex(index_from_batch[j]).position_);
    }

    NormalData(batch_data,
               _batch_size * dim_,
               dataset_handle_->GetMinVal(),
               dataset_handle_->GetMaxVal());
    float* gradient = new float[dim_ * num_class_];
    float curr_loss = LossFunc(batch_data, index_from_batch, gradient);

    if (i % 100 == 0) {
      printf("iteration %u / %u: loss %f\n", i, n_epochs_, curr_loss);
    }

    #pragma omp parallel for
    for (uint j = 0; j < dim_ * num_class_; ++j) {
      weights_[j] = weights_[j] - learning_rate_ * gradient[j];
    }

    // for (uint i = 0; i < batch_size_; ++i) {
    //   printf("%u  ", index_from_batch[i]);
    // }
    // printf("\n");
    if (gradient != NULL) {
      delete gradient;
    }
    gradient = NULL;

    if (batch_data != NULL) {
      delete batch_data;
    }
    batch_data = NULL;
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

  bin_weights_file.write((char *)weights_, dim_ * num_class_ * sizeof(float));

  bin_weights_file.close();
  printf("Model saved in %s \n", out_model_name.c_str());
  return 0;
}

float SoftmaxSVM::LossFunc(float* data,
                    std::vector<uint> list_position,
                    float* d_weights) {

  float loss = 0.f;

  float* scores = new float[batch_size_ * num_class_];
  MultipleMatrix(data, batch_size_, dim_,
                 weights_, dim_, num_class_,
                 scores);

  #pragma omp parallel for
  for (uint i = 0; i < batch_size_; ++i) {
    float score_axist1_m = *std::max_element(scores + i * num_class_,
                                     scores + (i + 1) * num_class_);
    float* curr_pointer = scores + i * num_class_;
    for (uint j = 0; j < num_class_; ++j) {
      *(curr_pointer  + j) = *(curr_pointer + j) - score_axist1_m;
    }
    curr_pointer = NULL;
  }

  #pragma omp parallel for
  for (uint i = 0; i < batch_size_ * num_class_; ++i) {
    *(scores + i) = std::exp(*(scores + i));
  }

  #pragma omp parallel for
  for (uint i = 0; i < batch_size_; ++i) {
    float exp_axist1_total = std::accumulate(scores + i * num_class_,
                                              scores + (i + 1) * num_class_,
                                              0.f);
    for (uint j = 0; j < num_class_; j++) {
      scores[i * num_class_ + j] /= exp_axist1_total;
    }
  }

  #pragma omp parallel for shared(loss)
  for (uint i = 0; i < batch_size_; ++i) {
    float get_tmp =
      scores[i * num_class_ +
             dataset_handle_->GetLabelFromIndex(list_position[i]).label_];

    #pragma omp atomic update
    loss += (-1 * std::log(get_tmp));
  }

  loss = loss / batch_size_;
  loss += 0.5 * regularization_ * std::accumulate(weights_,
                                    weights_ + dim_ * num_class_,
                                    0.f,
                                    [=](float a, float b) -> float {
                                      return a + b * b;
                                    });

  #pragma omp parallel for
  for (uint i = 0; i < batch_size_; ++i) {
    scores[i * num_class_ +
           dataset_handle_->GetLabelFromIndex(list_position[i]).label_] -= 1;
  }

  float* data_transpose = new float[dim_ * batch_size_];

  #pragma omp parallel for
  for (uint i = 0; i < dim_; ++i) {
    for (uint j = 0; j < batch_size_; ++j) {
      data_transpose[i * batch_size_ + j] =
        data[j *  dim_ + i];
    }
  }

  MultipleMatrix(data_transpose, dim_, batch_size_,
                 scores, batch_size_, num_class_,
                 d_weights);

  if (data_transpose != NULL) {
   delete data_transpose;
  }
  data_transpose = NULL;

  if (scores != NULL) {
   delete scores;
  }
  scores = NULL;

  for (uint i = 0; i < dim_ * num_class_; ++i) {
    d_weights[i] = d_weights[i] / batch_size_;
  }

  return loss;
}

float VectorizedSVM::LossFunc(float* data,
                    std::vector<uint> list_position,
                    float* d_weights) {

  float loss = 0.f;
//    float* d_weights = new float[dim_ * num_class_];
//  uint num_train = list_position.size();

  float* scores = new float[batch_size_ * num_class_];
  MultipleMatrix(data, batch_size_, dim_,
                 weights_, dim_, num_class_,
                 scores);
  float* correct_scores = new float[batch_size_];

  #pragma omp parallel for
  for (uint i = 0; i < batch_size_; ++i) {
    correct_scores[i] =
      scores[i * num_class_ +
             dataset_handle_->GetLabelFromIndex(list_position[i]).label_];
  }

  #pragma omp parallel for
  for (uint i = 0; i < batch_size_; ++i) {
    for (uint j = 0; j < num_class_; ++j) {
      scores[i * num_class_ + j] =
      std::max(0.f, scores[i * num_class_ + j] - correct_scores[i] + 1);
    }
  }

  if (correct_scores != NULL) {
    delete correct_scores;
  }
  correct_scores = NULL;

  #pragma omp parallel for
  for (uint i = 0; i < batch_size_; ++i) {
    scores[i * num_class_ +
        dataset_handle_->GetLabelFromIndex(list_position[i]).label_] = 0;
  }

  loss = std::accumulate(scores,
                  scores + batch_size_ * num_class_,
                  0.f) / batch_size_;

  loss += regularization_ * std::accumulate(weights_,
                                            weights_ + dim_ * num_class_,
                                            0.f,
                                            [=](float a, float b) -> float {
                                              return a + b * b;
                                            }) / batch_size_;

  #pragma omp parallel for
  for (uint i = 0; i < batch_size_ * num_class_; ++i) {
    if (scores[i] > 0) {
      scores[i] = 1;
    }
  }

  float* row_sum = new float[batch_size_];
  #pragma omp parallel for
  for (uint i = 0; i < batch_size_; ++i) {
    row_sum[i] = std::accumulate(scores + (i * num_class_),
                                scores + (i + 1) * num_class_,
                                0.f);
  }

  #pragma omp parallel for
  for (uint i = 0; i < batch_size_; ++i) {
    uint label = dataset_handle_->GetLabelFromIndex(list_position[i]).label_;
    scores[i * num_class_ + label] = -row_sum[i];
  }

  if (row_sum != NULL) {
    delete row_sum;
  }
  row_sum = NULL;

  float* data_transpose = new float[dim_ * batch_size_];

  #pragma omp parallel for
  for (uint i = 0; i < dim_; ++i) {
    for (uint j = 0; j < batch_size_; ++j) {
      data_transpose[i * batch_size_ + j] =
        data[j *  dim_ + i];
    }
  }

  MultipleMatrix(data_transpose, dim_, batch_size_,
                 scores, batch_size_, num_class_,
                 d_weights);

  if (data_transpose != NULL) {
    delete data_transpose;
  }
  data_transpose = NULL;

  if (scores != NULL) {
    delete scores;
  }
  scores = NULL;

  #pragma omp parallel for
  for (uint i = 0; i < dim_ * num_class_; i ++) {
    d_weights[i] = d_weights[i] / batch_size_;
    d_weights[i] = d_weights[i] + regularization_ * 2 * weights_[i];
  }

  return loss;
}

float VectorizedSVM::Evaluate(float* _input_data,
                      const std::vector<uint>& _real_label) {
  if (weights_ == NULL || _input_data == NULL) {
    printf("Invalid weights or data \n");
    return -1.f;
  }
  uint n_sample = _real_label.size();
  float* result = new float[n_sample * num_class_];

  MultipleMatrix(_input_data, n_sample, dim_,
                 weights_, dim_, num_class_,
                 result);

  std::vector<uint> _predic_label;
  for (uint i = 0; i < n_sample; ++i) {
    uint pre_label = std::distance(result + i * num_class_,
                  std::max_element(result + i * num_class_, result + (i + 1) * num_class_));
    _predic_label.push_back(pre_label);
  }

  uint count_diff = 0;
  for (uint i = 0; i < n_sample; ++i) {
    if (_predic_label[i] != _real_label[i]) {
      count_diff++;
    }
  }

  if (result != NULL) {
    delete result;
  }
  result = NULL;

  return 1.f - (float)count_diff / n_sample;
}

float SoftmaxSVM::Evaluate(float* _input_data,
                      const std::vector<uint>& _real_label) {
  if (weights_ == NULL || _input_data == NULL) {
    printf("Invalid weights or data \n");
    return -1.f;
  }
  uint n_sample = _real_label.size();
  float* result = new float[n_sample * num_class_];

  MultipleMatrix(_input_data, n_sample, dim_,
                 weights_, dim_, num_class_,
                 result);

  for (uint i = 0; i < n_sample * num_class_; ++i) {
    result[i] = std::exp(result[i]);
  }

  std::vector<uint> _predic_label;
  for (uint i = 0; i < n_sample; ++i) {
    float * max_pos = std::max_element(result + i * num_class_, result + (i + 1) * num_class_);
    uint pre_label = std::distance(result + i * num_class_, max_pos);
    _predic_label.push_back(pre_label);
    float curr_total = std::accumulate(result + i * num_class_,
                                       result + (i + 1) * num_class_,
                                       0.f);
  }

  uint count_diff = 0;
  for (uint i = 0; i < n_sample; ++i) {
    if (_predic_label[i] != _real_label[i]) {
      count_diff++;
    }
  }

  if (result != NULL) {
    delete result;
  }
  result = NULL;

  return 1.f - (float)count_diff / n_sample;
}

int SVM::NormalData(float *data,
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

template<class T>
int MultipleMatrix(T *mx_a, size_t a_row, size_t a_cols,
                   T *mx_b, size_t b_row, size_t b_cols,
                   T *res) {
  if (a_cols != b_row) {
    printf("matrix is not match \n");
    return -1;
  }

  if (res == NULL) {
    printf("You need to allocate memory for result\n");
    return -1;
  }

  T* mb_transpose = new T[b_cols * b_row];
  #pragma omp parallel for
  for (uint i = 0; i < b_cols; ++i) {
    for (uint j = 0; j < b_row; ++j) {
      mb_transpose[i * b_row + j] =
        mx_b[j *  b_cols + i];
    }
  }


//  res = new T[a_row * b_cols];
#pragma omp parallel shared(mx_a,mb_transpose,res)
{
  T *trans_temporary = new T[a_cols];
  T *trans_temporary_end = trans_temporary + a_cols;
  #pragma omp for
  for (uint i = 0; i < a_row; ++i) {
    T* current_ma_index = mx_a + i * a_cols;
    T* current_ma_index_end = current_ma_index + a_cols;
    T* current_res_index = res + i * b_cols;

    for (uint j = 0; j < b_cols; ++j) {
      std::transform(current_ma_index,
                     current_ma_index_end,
                     mb_transpose + j * b_row,
                     trans_temporary,
                     std::multiplies<T>());
      *(current_res_index + j) = std::accumulate(trans_temporary,
                                                 trans_temporary_end,
                                                   0.f);
      // float tmp = 0.f;
      // for (uint k = 0; k < b_row; ++k) {
      //   tmp = tmp + mx_a[zzz + k];
      // }

//      std::tra
//      res[mmm + j] = std::accumulate(zzz,
//                                     mmmm,
//                                     0.f) * std::accumulate(zzz,
 //                                                                   mmmm,
 //                                                                   0.f);
    }
  }
  delete trans_temporary;
}
  delete mb_transpose;
  return 0;
}
