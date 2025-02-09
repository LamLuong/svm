#ifndef DATASET_H
#define DATASET_H

#include <stdio.h>
#include <fstream>
#include <vector>

typedef unsigned int uint;
class Dataset {
 struct LabelPosition {
   uint position_;
   uint label_;
 };
 public:
  Dataset() {
    total_sample_ = 0;
    total_class_ = 0;
    min_value_ = -1000.f;
    max_value_ = 1000.f;
    feature_size_ = 0;
    name_encoding_ = "";
  }

  ~Dataset() {
    features_data_.close();
    label_position_.clear();
  }

  int LoadData(const std::string&, const std::string&);
  int NormalData(float *, uint);
  void ReadData(float *, uint);

  uint GetNumSample() {
    return total_sample_;
  }

  uint GetNumClass() {
    return total_class_;
  }

  float GetMinVal() {
    return min_value_;
  }

  float GetMaxVal() {
    return max_value_;
  }

  uint GetFeatureSize() {
    return feature_size_;
  }

  std::string GetEncodingClass() {
    return name_encoding_;
  }

  LabelPosition GetLabelFromIndex(uint _idx) {
    return { label_position_[_idx].position_, label_position_[_idx].label_};
  }

 private:
  std::ifstream features_data_;
  std::vector<LabelPosition> label_position_;
  std::string name_encoding_;
  uint total_sample_;
  uint total_class_;
  uint feature_size_;
  float min_value_;
  float max_value_;
};

#endif
