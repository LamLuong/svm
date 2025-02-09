//#include <getopt.h>

#include "svm.h"
void print_usage();

int main(int argc, char* argv[]) {

  Dataset dataset;
  std::string binary_file_path = std::string(argv[1]) + "/features.bin";
  std::string index_file_path = std::string(argv[1]) + "/label_idx.txt";
  dataset.LoadData(index_file_path,
                   binary_file_path);

  SoftmaxSVM svm_handle;
  svm_handle.InitData(NULL, 0, 0, argv[2]);

  printf("handler dataset %u dimension of feature of %u samples\n",
          dataset.GetFeatureSize(), dataset.GetNumSample());
 // svm_handle.PrintClass();

  float* batch_data = new float[dataset.GetNumSample() * dataset.GetFeatureSize()];
  std::vector<uint> index_from_batch;
  index_from_batch.resize(dataset.GetNumSample());

  for (uint i = 0; i < index_from_batch.size(); i++) {
    index_from_batch[i] = i;
  }

  std::vector<uint> real_label;
  real_label.resize(dataset.GetNumSample());

  for(uint j = 0; j < dataset.GetNumSample(); ++j) {
    dataset.ReadData(batch_data + j * dataset.GetFeatureSize(),
      dataset.GetLabelFromIndex(index_from_batch[j]).position_);

    real_label[j] = dataset.GetLabelFromIndex(index_from_batch[j]).label_;
  }

  svm_handle.NormalData(batch_data,
                        dataset.GetNumSample() * dataset.GetFeatureSize(),
                        dataset.GetMinVal(),
                        dataset.GetMaxVal());

  float x = svm_handle.Evaluate(batch_data, real_label);
  printf("%f \n", x);

  if (batch_data != NULL) {
    delete batch_data;
  }
  batch_data = NULL;

  return 0;
}

void print_usage() {
  printf("Usage: rectangle [ap] -l num -b num\n");
}
