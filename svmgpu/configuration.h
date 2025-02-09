/*file configuration.h
* author: lamluong
* date: 2019
*/
#ifndef CONFIGURATION_H
#define CONFIGURATION_H
/*
 * include common header file
 * init all path file
 * include constances and thresholds
 * Define structs
 * Define enums
 */

#include <libconfig.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

class Configuration {
 public:

  static Configuration* GetInstance() {
    if (!instance) {
      instance = new Configuration();
    }
  return instance;
  };

  ~Configuration() {};
 private:
  Configuration();
  int ReadConfig(const char* filename);

 public:
  int batch_size_;
  int n_sleep_;
  int n_epochs_;
 private:
	static Configuration* instance;
};

#endif // CONFIGURATION_H
