#include "configuration.h"
#include <sstream>

#define Q(x) #x
#define QUOTE(x) Q(x)

Configuration* Configuration::instance = new Configuration();

Configuration::Configuration() {
  batch_size_ = 100;
  n_sleep_ = 10;
  n_epochs_ = 100000;
  const char* config_path = "../config.ini";
  ReadConfig(config_path);
}

int Configuration::ReadConfig(const char* filename) {
  std::ifstream fin(filename);

  if(!fin.is_open()) {
    printf("could not open config file: %s \n", filename);
    return -1;
  }
  printf("Read config file successfully\n");
  std::string line;
  std::string::size_type sz;
  while (std::getline(fin, line)) {
    std::istringstream in(line);
    std::string option, val;
    in >> option >> val;

    if (option.compare("batch_size") == 0) {
      batch_size_ = std::atoi(val.c_str());
    } else if (option.compare("n_sleep") == 0) {
      n_sleep_ = std::atoi(val.c_str());
    } else if (option.compare("epochs") == 0) {
      n_epochs_ = std::atoi(val.c_str());
    }
  }
  return 0;
}
