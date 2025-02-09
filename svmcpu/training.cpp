#include "svm.h"

void print_usage();

int main(int argc, char* argv[]) {

  Dataset dataset;
  std::string binary_file_path = std::string(argv[1]) + "/features.bin";
  std::string index_file_path = std::string(argv[1]) + "/label_idx.txt";

  dataset.LoadData(index_file_path,
                   binary_file_path);

  SoftmaxSVM svm_handle;
  // change reg = 1e-5 and learning rate to 1e-1 for softmax

  // we will set regularization value to 1e-5 and learning rate to 1e-1
  svm_handle.InitData(&dataset, 1e-5, 1e-1);
  // we will train with batch_size 500 and 100000 epochs
  svm_handle.TrainData(500, 100000);

  return 0;
}

void print_usage() {
  printf("Usage: rectangle [ap] -l num -b num\n");
}
