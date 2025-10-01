#include <cstdio>
#include <vector>
#include <iostream>
#include <fstream>
#include <memory>
#include <chrono>
#include <algorithm>
#include <cmath>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include "headers/edgetpu_c.h"
#include "opencv2/opencv.hpp"

// This is an example that is minimal to read a model
// from disk and perform inference. There is no data being loaded
// that is up to you to add as a user.

using namespace std;
using namespace cv;

// MAKE SURE TO USE PROPER DIRECTORIES
#define IMAGE_PATH "/home/byeongyun/EAI_TfLite/04_tpu_yolo/image/sample.jpg"  // Replace with your YOLO test image
#define VOC_CLASSES {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"}

// YOLOv1 parameters
const int S = 7;  // Grid size
const int B = 2;  // Boxes per cell
const int C = 20; // Classes
const float CONF_THRESH = 0.2f;
const float IOU_THRESH = 0.5f;

// Box structure
struct Box {
    float x, y, w, h, conf;
    int class_id;
    float prob;
};

// IoU calculation
float iou(const Box& a, const Box& b) {
    float x1 = std::max(a.x - a.w / 2, b.x - b.w / 2);
    float y1 = std::max(a.y - a.h / 2, b.y - b.h / 2);
    float x2 = std::min(a.x + a.w / 2, b.x + b.w / 2);
    float y2 = std::min(a.y + a.h / 2, b.y + b.h / 2);
    float inter = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
    float area_a = a.w * a.h;
    float area_b = b.w * b.h;
    return inter / (area_a + area_b - inter + 1e-6f);
}

// NMS
std::vector<Box> nms(std::vector<Box> boxes, float iou_thresh) {
    std::sort(boxes.begin(), boxes.end(), [](const Box& a, const Box& b) {
        return a.prob > b.prob;
    });
    std::vector<Box> result;
    std::vector<bool> picked(boxes.size(), false);
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (picked[i]) continue;
        result.push_back(boxes[i]);
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (iou(boxes[i], boxes[j]) > iou_thresh) {
                picked[j] = true;
            }
        }
    }
    return result;
}

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

// Read image with opencv
void readImageCV(string filename, cv::Mat& input){
	cv::Mat cvimg = cv::imread(filename, cv::IMREAD_COLOR);
	if(cvimg.data == NULL){
		std::cout << "=== IMAGE DATA NULL ===\n";
		return;
	}

	cv::cvtColor(cvimg, cvimg, COLOR_BGR2RGB);
  cv::resize(cvimg, cvimg, cv::Size(448, 448));  // YOLOv1 input size
  
  cvimg.convertTo(cvimg, CV_32F, 1.0f / 255.0f);  // Normalize to [0,1] for float model; adjust if quantized
  input = cvimg;
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    fprintf(stderr, "minimal <model> <use tpu 0/1> <inference num>\n");
    return 1;
  }

  const char* filename = argv[1];
  bool use_tpu = std::stoi(argv[2]);
  int inference_num = std::stoi(argv[3]);

  if(use_tpu){
    std::cout << "Use TPU acceleration" << "\n";
  }
  else{
    std::cout << "No TPU acceleration" << "\n";
  }
  std::cout << "Inference " << inference_num << " times and get average latency" << "\n";
  

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter with the InterpreterBuilder.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Setup for Edge TPU device.
  if(use_tpu){
    size_t num_devices;
    std::unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(
        edgetpu_list_devices(&num_devices), &edgetpu_free_devices);

    if (num_devices == 0) {
      std::cerr << "No Edge TPU devices found\n";
      return 1;
    }
    const auto& device = devices.get()[0];

    // Create TPU delegate.
    auto* delegate =
      edgetpu_create_delegate(device.type, device.path, nullptr, 0);

    // Delegate graph.
    interpreter->ModifyGraphWithDelegate(delegate);
  }

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  printf("=== Pre-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());


  // Variables to measure invoke latency.
  struct timespec begin, end;
  double latency = 0;
  
  // Read input image   
  cv::Mat input;
  readImageCV(IMAGE_PATH, input);

  std::vector<std::string> voc_classes = VOC_CLASSES;

  for(int seq=0; seq<inference_num; ++seq){
    // Fill input buffers
    float* input_tensor = interpreter->typed_input_tensor<float>(0);  // Assume float32; change to int8_t if quantized
    // Copy input
    memcpy(input_tensor, input.data, 448 * 448 * 3 * sizeof(float));

    // Get start time
    clock_gettime(CLOCK_MONOTONIC, &begin);
    // Run inference
    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

    // Get end time
    clock_gettime(CLOCK_MONOTONIC, &end);
    double temp = (end.tv_sec - begin.tv_sec) + ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
    latency += temp;

    // Read output buffers
    float* output_tensor = interpreter->typed_output_tensor<float>(0);  // [1, 7, 7, 30]

    // Parse YOLOv1 output
    std::vector<Box> boxes;
    for (int y = 0; y < S; ++y) {
        for (int x = 0; x < S; ++x) {
            int cell_offset = (y * S + x) * (C + B * 5);
            // Find max class prob
            float max_prob = 0.f;
            int max_class = -1;
            for (int c = 0; c < C; ++c) {
                float prob = output_tensor[cell_offset + c];
                if (prob > max_prob) {
                    max_prob = prob;
                    max_class = c;
                }
            }
            // Boxes
            for (int b = 0; b < B; ++b) {
                int box_offset = cell_offset + C + b * 5;
                float conf = output_tensor[box_offset + 4];
                float prob = conf * max_prob;
                if (prob > CONF_THRESH) {
                    float cx = (output_tensor[box_offset] + x) / S;
                    float cy = (output_tensor[box_offset + 1] + y) / S;
                    float w = output_tensor[box_offset + 2];
                    float h = output_tensor[box_offset + 3];
                    boxes.push_back({cx, cy, w, h, conf, max_class, prob});
                }
            }
        }
    }

    // Apply NMS
    auto final_boxes = nms(boxes, IOU_THRESH);

    // Print detections
    std::cout << "Detections for inference " << seq << ":" << std::endl;
    for (const auto& box : final_boxes) {
        std::cout << "Class: " << voc_classes[box.class_id] << ", Prob: " << box.prob
                  << ", Box: (" << box.x - box.w/2 << ", " << box.y - box.h/2
                  << ", " << box.x + box.w/2 << ", " << box.y + box.h/2 << ")" << std::endl;
    }
  }

  printf("Total elapsed time : %.6f sec\n", latency);
  printf("Average inference latency : %.6f sec\n", latency / inference_num);
  
  return 0;
}