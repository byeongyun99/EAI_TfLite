#include <cstdio>
#include <vector>
#include <iostream>
#include <fstream>
#include <memory>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <string>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include "headers/edgetpu_c.h"
#include "opencv2/opencv.hpp"


using namespace std;
using namespace cv;

#define IMAGE_PATH "/home/byeongyun/EAI_TfLite/04_tpu_yolo/image/sample.jpg" 
#define OUTPUT_IMAGE_PATH "/home/byeongyun/EAI_TfLite/04_tpu_yolo/output_image/output_with_boxes.jpg" 
#define VOC_CLASSES {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"}

const int S = 7;  // Grid size
const int B = 2;  // Boxes per cell
const int C = 20; // Classes
const float CONF_THRESH = 0.3f;
const float IOU_THRESH = 0.2f;

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Box structure
struct Box {
    float x, y, w, h, conf;
    int class_id;
    float prob;
};

// IoU calculation
float iou(const Box& a, const Box& b) {
    float ax1 = a.x - a.w / 2;
    float ay1 = a.y - a.h / 2;
    float ax2 = a.x + a.w / 2;
    float ay2 = a.y + a.h / 2;

    float bx1 = b.x - b.w / 2;
    float by1 = b.y - b.h / 2;
    float bx2 = b.x + b.w / 2;
    float by2 = b.y + b.h / 2;

    float inter_x1 = std::max(ax1, bx1);
    float inter_y1 = std::max(ay1, by1);
    float inter_x2 = std::min(ax2, bx2);
    float inter_y2 = std::min(ay2, by2);

    float inter_w = std::max(0.f, inter_x2 - inter_x1);
    float inter_h = std::max(0.f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;

    float area_a = (ax2 - ax1) * (ay2 - ay1);
    float area_b = (bx2 - bx1) * (by2 - by1);

    return inter_area / (area_a + area_b - inter_area + 1e-6f);
}

std::vector<Box> nms(std::vector<Box> boxes, float iou_thresh) {
    std::sort(boxes.begin(), boxes.end(), [](const Box& a, const Box& b) {
        return a.prob > b.prob;
    });

    std::vector<Box> result;
    std::vector<bool> removed(boxes.size(), false);

    for (size_t i = 0; i < boxes.size(); ++i) {
        if (removed[i]) continue;

        const Box& current = boxes[i];
        result.push_back(current);

        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (removed[j]) continue;
            const Box& compare = boxes[j];

            if (current.class_id == compare.class_id) {
                float overlap = iou(current, compare);
                if (overlap > iou_thresh) {
                    removed[j] = true;
                }
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
  
  input = cvimg;
}


int main(int argc, char* argv[]) {

  if (argc != 4) {
    fprintf(stderr, "<model> <use tpu 0/1> <rep>\n");
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
  std::vector<Box> boxes;

  for(int seq=0; seq<inference_num; ++seq){

    TfLiteTensor* input_tensor_ptr = interpreter->input_tensor(0);
    TfLiteQuantizationParams input_params = input_tensor_ptr->params;
    float input_scale = input_params.scale;
    int32_t input_zero_point = input_params.zero_point;

    // Assume input is int8_t, and image is CV_8UC3 [0,255]
    int8_t* input_tensor = interpreter->typed_input_tensor<int8_t>(0);

    const int8_t* src_data = input.ptr<int8_t>();

    for (size_t i = 0; i < 448 * 448 * 3; ++i) {
      input_tensor[i] = static_cast<int8_t>((static_cast<float>(src_data[i]) / input_scale) + input_zero_point);
    }

    // Get start time
    clock_gettime(CLOCK_MONOTONIC, &begin);
    // Run inference
    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

    // Get end time
    clock_gettime(CLOCK_MONOTONIC, &end);
    double temp = (end.tv_sec - begin.tv_sec) + ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
    latency += temp;

    TfLiteTensor* output_tensor_ptr = interpreter->output_tensor(0);
    TfLiteQuantizationParams output_params = output_tensor_ptr->params;
    float output_scale = output_params.scale;
    int32_t output_zero_point = output_params.zero_point;

    // Read output buffers
    int8_t* output_tensor = interpreter->typed_output_tensor<int8_t>(0);  // [1, 7, 7, 30]
    boxes.clear();
    // dequantization
    for (int y = 0; y < S; ++y) {
        for (int x = 0; x < S; ++x) {
            int cell_offset = (y * S + x) * (C + B * 5);
            // Boxes
            for (int b = 0; b < B; ++b) {
                int box_offset = cell_offset + C + b * 5;
                int8_t raw_conf = output_tensor[box_offset + 4];
                float dequant_conf = (static_cast<float>(raw_conf) - output_zero_point) * output_scale;
                float conf = sigmoid(dequant_conf);
                if (conf > CONF_THRESH) { 
                    // Find max class for this box
                    float max_prob = 0.f;
                    int max_class = -1;
                    for (int c = 0; c < C; ++c) {
                        int8_t raw = output_tensor[cell_offset + c];
                        float dequant = (static_cast<float>(raw) - output_zero_point) * output_scale;
                        float class_prob = sigmoid(dequant);   
                        if (class_prob > max_prob) {
                            max_prob = class_prob;
                            max_class = c;
                        }
                    }

                    float prob = conf * max_prob;
                    //printf("conf=%.3f, max_prob=%.3f, prob=%.3f\n", conf, max_prob, prob);

                    if (prob > CONF_THRESH) {
                        int8_t raw_cx = output_tensor[box_offset];
                        int8_t raw_cy = output_tensor[box_offset + 1];
                        int8_t raw_w = output_tensor[box_offset + 2];
                        int8_t raw_h = output_tensor[box_offset + 3];
                        float dequant_cx = (static_cast<float>(raw_cx) - output_zero_point) * output_scale;
                        float dequant_cy = (static_cast<float>(raw_cy) - output_zero_point) * output_scale;
                        float dequant_w = (static_cast<float>(raw_w) - output_zero_point) * output_scale;
                        float dequant_h = (static_cast<float>(raw_h) - output_zero_point) * output_scale;
                        float cx = (sigmoid(dequant_cx) + x) / S;
                        float cy = (sigmoid(dequant_cy) + y) / S;
                        float w = pow(dequant_w, 2); 
                        float h = pow(dequant_h, 2);
                        cx = std::max(0.0f, std::min(1.0f, cx));
                        cy = std::max(0.0f, std::min(1.0f, cy));
                        w = std::max(0.0f, std::min(1.0f, w));
                        h = std::max(0.0f, std::min(1.0f, h));
                        boxes.push_back({cx, cy, w, h, conf, max_class, prob});
                    }
                }
            }
        }
    }
  }
  std::sort(boxes.begin(), boxes.end(),
          [](const Box& a, const Box& b) { return a.prob > b.prob; });
  if (boxes.size() > 2)
      boxes.resize(2);
  auto final_boxes = nms(boxes, IOU_THRESH);

  // Print detections
  std::cout << "Detections:" << std::endl;
  for (const auto& box : final_boxes) {
      std::cout << "Class: " << voc_classes[box.class_id] << std::endl;
  }

  printf("Average inference latency : %.6f sec\n", latency / inference_num);

  // Draw bounding boxes on the resized image (input is RGB 448x448)
  cv::Mat output_img;
  cv::cvtColor(input, output_img, COLOR_RGB2BGR); 

  for (const auto& box : final_boxes) {
      int left = static_cast<int>((box.x - box.w / 2) * 448);
      int top = static_cast<int>((box.y - box.h / 2) * 448);
      int right = static_cast<int>((box.x + box.w / 2) * 448);
      int bottom = static_cast<int>((box.y + box.h / 2) * 448);

      left = std::max(0, left);
      top = std::max(0, top);
      right = std::min(447, right);
      bottom = std::min(447, bottom);

      cv::rectangle(output_img, Point(left, top), Point(right, bottom), Scalar(0, 255, 0), 2);
      cv::putText(output_img, voc_classes[box.class_id] + " " + std::to_string(box.prob).substr(0, 4),
                  Point(left, top - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
  }

  // Save the output image
  if (!cv::imwrite(OUTPUT_IMAGE_PATH, output_img)) {
      std::cerr << "Failed to save output image" << std::endl;
      return 1;
  }

  std::cout << "Output image with bounding boxes saved to: " << OUTPUT_IMAGE_PATH << std::endl;
  
  return 0;
}
