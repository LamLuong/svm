#include <math.h>
#include <chrono>
#include <fstream>
#include <sstream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>

cv::dnn::Net net;
std::chrono::high_resolution_clock m_clock;
uint64_t GetTimeNanoSec();
int GetFeature(const cv::Mat& faceImg, cv::Mat& _feature);
std::string GetStdoutFromCommand(std::string cmd);
std::string GetLabelFromPath(std::string path);

int main(int argc, char* argv[]) {
  if (argc < 5) {
    printf("too few argument \n");
    return -1;
  }
  std::string handle_folder = "find " + std::string(argv[3]) + " -name *.* | sort";
  std::string test = GetStdoutFromCommand(handle_folder);
  std::stringstream ss(test);
  std::string label_name = "";
  size_t count_element = 0;
  float max_val = -1000.f, min_val = 1000.f;
  unsigned int feature_size = 0;


  net = cv::dnn::readNetFromCaffe(argv[1], argv[2]);

#ifdef USE_CUDA_BACKEND
  net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
  net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
#else
  net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
  net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
#endif

  std::string binary_file_path = std::string(argv[4]) + "/features.bin";
  std::string index_file_path = std::string(argv[4]) + "/label_idx.txt";
  std::ofstream bin_feature_file(binary_file_path, std::ios::out | std::ios::binary);
  std::ofstream index_file(index_file_path);

  std::string line;
  while (std::getline(ss, line)) {

    if (label_name == "") {
      label_name = GetLabelFromPath(line);
    }

    if (label_name != GetLabelFromPath(line)) {
      printf("count element %lu \n", count_element);
      index_file << label_name << " " << count_element << std::endl;
      index_file.flush();
      label_name = GetLabelFromPath(line);
      count_element = 0;
    }

    cv::Mat input = cv::imread(line);
    cv::Mat out;
    GetFeature(input, out);

    double cur_min, cur_max;
    cv::minMaxLoc(out, &cur_min, &cur_max);
    if (cur_min < min_val) {
      min_val = (float)cur_min;
    }
    if (cur_max > max_val) {
      max_val = (float)cur_max;
    }

    if (feature_size == 0) {
      feature_size = (unsigned int)(out.rows * out.cols);
    }

//   for (uint l = 0; l < out.rows * out.cols; l++) {
//    printf("%f  %f \n", ((float*)out.data)[0], ((float*)out.data)[out.rows * out.cols - 1]);
//    }
//    printf("\n");
    bin_feature_file.write((char *) out.data, out.elemSize1() * out.rows * out.cols);
    count_element++;
  }

  bin_feature_file.write((char *)&min_val, sizeof(float));
  bin_feature_file.write((char *)&max_val, sizeof(float));
  bin_feature_file.write((char *)&feature_size, sizeof(unsigned int));

  index_file << label_name << " " << count_element << std::endl;

  bin_feature_file.close();
  index_file.close();
  printf("Min - max: %f - %f \n", min_val, max_val);
  printf("write done \n");

  return 0;
}

int GetFeature(const cv::Mat& faceImg, cv::Mat& _feature) {
  if (net.empty()) {
    printf("Net is empty \n");
    return -1;
  }

  cv::Mat image;
  image = faceImg.clone();

  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  image.convertTo(image, CV_32FC3);
  image = (image - 127.5) * 0.0078125;

  cv::Mat inputBlob = cv::dnn::blobFromImage(image, 1, cv::Size(112, 112));
  net.setInput(inputBlob);
  auto feature = net.forward();

  double norm = cv::norm(feature);
  _feature = feature.clone() / norm;
  return 0;
}

uint64_t GetTimeNanoSec() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>
              (m_clock.now().time_since_epoch()).count();
}

std::string GetLabelFromPath(std::string path) {
  std::string label_name = path.substr(0, path.find_last_of("/\\"));
  label_name = label_name.substr(label_name.find_last_of("/\\") + 1, label_name.length());
  return label_name;
}

std::string GetStdoutFromCommand(std::string cmd) {

  std::string data;
  FILE * stream;
  const int max_buffer = 256;
  char buffer[max_buffer];
  cmd.append(" 2>&1");

  stream = popen(cmd.c_str(), "r");
  if (stream) {
    while (!feof(stream))
      if (fgets(buffer, max_buffer, stream) != NULL) data.append(buffer);
    pclose(stream);
  }
  return data;
}
