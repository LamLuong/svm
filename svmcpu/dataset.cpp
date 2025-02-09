#include "dataset.h"

#include <sstream>
#include <random>
#include <algorithm>

int Dataset::LoadData(const std::string& _index_file,
             const std::string& _feature_file) {

  std::ifstream index_data;
  index_data.open(_index_file, std::ifstream::in);
  if (!index_data.good()) {
    printf("Cannot open index file \n");
    return -1;
  }
  std::string line;
  uint current_label = 0;
  uint current_position = 0;
  while (std::getline(index_data, line)) {
    std::stringstream ss(line);
    uint tmp = 0;
    std::string class_name;
    uint encode_name = 0;
    ss >> class_name >> tmp;
    name_encoding_ = name_encoding_ + class_name + " ";
    total_sample_ += tmp;
    total_class_++;
    for(uint i = 0; i < tmp; i++) {
      label_position_.push_back({current_position, current_label});
      current_position++;
    }
    current_label++;
  }

  index_data.close();

  features_data_.open(_feature_file, std::ios::in | std::ios::binary);
  if (!features_data_.good()) {
    printf("Cannot open data file \n");
    return -1;
  }

  features_data_.seekg(-1 * (sizeof(float) + sizeof(float) + sizeof(unsigned int)),
                       std::ios::end);
  features_data_.read((char*)&min_value_, sizeof(float));
  features_data_.read((char*)&max_value_, sizeof(float));
  features_data_.read((char*)&feature_size_, sizeof(unsigned int));
  features_data_.seekg(0, std::ios::beg);

  std::random_device rd;
  auto rng = std::default_random_engine { rd() };

  std::shuffle(std::begin(label_position_),
               std::end(label_position_),
               rng);

  return 0;
}

void Dataset::ReadData(float *data, uint sample_index) {
  features_data_.seekg(sample_index * feature_size_ * sizeof(float),
                       std::ios::beg);
  features_data_.read((char*)data, feature_size_ * sizeof(float));
}
