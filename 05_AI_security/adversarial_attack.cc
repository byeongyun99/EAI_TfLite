#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <memory>
#include <algorithm>
#include <chrono>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include "headers/edgetpu_c.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// ------------------------------------------------
const int   TARGET_SIZE = 320;          
const float CONF_THRESH_DEFAULT = 0.3f;
const float IOU_THRESH_DEFAULT  = 0.45f;
const int   PRE_NMS_TOPK        = 300;
const int   MAX_SHOW            = 100;
// ------------------------------------------------

static inline float sigmoidf(float x){ return 1.0f/(1.0f+expf(-x)); }

struct Detection {
    float x,y,w,h;
    float score;
    int cls;
};
static float iou_box(const Detection &a, const Detection &b){
    float ax1 = a.x - a.w/2.0f, ay1 = a.y - a.h/2.0f, ax2 = a.x + a.w/2.0f, ay2 = a.y + a.h/2.0f;
    float bx1 = b.x - b.w/2.0f, by1 = b.y - b.h/2.0f, bx2 = b.x + b.w/2.0f, by2 = b.y + b.h/2.0f;
    float ix1 = max(ax1,bx1), iy1 = max(ay1,by1), ix2 = min(ax2,bx2), iy2 = min(ay2,by2);
    float iw = max(0.0f, ix2 - ix1), ih = max(0.0f, iy2 - iy1);
    float inter = iw * ih;
    float areaA = max(0.0f, (ax2-ax1) * (ay2-ay1));
    float areaB = max(0.0f, (bx2-bx1) * (by2-by1));
    return inter / (areaA + areaB - inter + 1e-9f);
}

static vector<Detection> nms_classwise(vector<Detection> dets, float iou_th){
    sort(dets.begin(), dets.end(), [](const Detection &a, const Detection &b){ return a.score > b.score; });
    vector<Detection> out;
    vector<char> removed(dets.size(), 0);
    for (size_t i=0;i<dets.size();++i){
        if (removed[i]) continue;
        out.push_back(dets[i]);
        for (size_t j=i+1;j<dets.size();++j){
            if (removed[j]) continue;
            if (dets[i].cls != dets[j].cls) continue;
            if (iou_box(dets[i], dets[j]) > iou_th) removed[j]=1;
        }
    }
    return out;
}
static Scalar color_for_class(int cls){
    int hue = (cls * 37) % 180;
    Mat hsv(1,1,CV_8UC3, Scalar(hue, 200, 255));
    Mat bgr; cvtColor(hsv, bgr, COLOR_HSV2BGR);
    Vec3b v = bgr.at<Vec3b>(0,0);
    return Scalar(v[0], v[1], v[2]);
}

static vector<string> load_labels(const string &path){
    vector<string> labels;
    ifstream ifs(path);
    if(!ifs.is_open()){
        cerr<<"Warning: cannot open labels file: "<<path<<", proceeding without names.\n";
        return labels;
    }
    string line;
    while(getline(ifs, line)){
        if(line.size() && line.back()=='\r') line.pop_back();
        if (!line.empty()) labels.push_back(line);
    }
    cout<<"Loaded "<<labels.size()<<" labels\n";
    return labels;
}

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

static void letterbox_resize(
    const Mat& src_bgr,
    Mat& out_rgb,
    int target,
    float &scale,
    int &pad_x,
    int &pad_y
){
    // convert to RGB first
    Mat src_rgb;
    cvtColor(src_bgr, src_rgb, COLOR_BGR2RGB);

    int src_w = src_rgb.cols;
    int src_h = src_rgb.rows;

    float r = std::min( (float)target / (float)src_w,
                        (float)target / (float)src_h );
    int new_w = (int)round(src_w * r);
    int new_h = (int)round(src_h * r);

    Mat resized;
    resize(src_rgb, resized, Size(new_w, new_h));

    // make padded canvas
    out_rgb = Mat::zeros(Size(target, target), CV_8UC3);
    // we'll pad with 0 (black). that's fine for yolov8-style inference.
    pad_x = (target - new_w) / 2;
    pad_y = (target - new_h) / 2;

    // copy resized into center
    resized.copyTo(out_rgb(Rect(pad_x, pad_y, new_w, new_h)));

    // export info
    scale = r;
}
static Rect box_to_orig_rect(
    const Detection& d,
    int orig_w, int orig_h,
    float scale, int pad_x, int pad_y
){
    // d.x,d.y,d.w,d.h are normalized in [0..1] relative to TARGET_SIZE
    float cx_img = d.x * TARGET_SIZE;
    float cy_img = d.y * TARGET_SIZE;
    float w_img  = d.w * TARGET_SIZE;
    float h_img  = d.h * TARGET_SIZE;

    float x1_img = cx_img - w_img/2.0f;
    float y1_img = cy_img - h_img/2.0f;
    float x2_img = cx_img + w_img/2.0f;
    float y2_img = cy_img + h_img/2.0f;

    // undo padding: the model actually "saw" [pad_x:pad_x+new_w], [pad_y:pad_y+new_h]
    // so shift by -pad
    x1_img -= pad_x;
    y1_img -= pad_y;
    x2_img -= pad_x;
    y2_img -= pad_y;

    // undo scaling: scale = orig -> resized factor, so divide to get back to orig
    if (scale > 0){
        x1_img /= scale;
        y1_img /= scale;
        x2_img /= scale;
        y2_img /= scale;
    }

    // clamp to original image
    int ix1 = std::max(0, (int)round(x1_img));
    int iy1 = std::max(0, (int)round(y1_img));
    int ix2 = std::min(orig_w-1, (int)round(x2_img));
    int iy2 = std::min(orig_h-1, (int)round(y2_img));

    return Rect(Point(ix1, iy1), Point(ix2, iy2));
}

int main(int argc, char** argv){
    if (argc < 6){
        cerr<<"Usage: "<<argv[0]<<" <model.tflite> <labels.txt> <input.jpg> <output.jpg> [use_tpu 0/1]\n";
        return 1;
    }
    const char* filename = argv[1];
    string labels_path   = argv[2];
    string input_path    = argv[3];
    string output_path   = argv[4];
    bool use_tpu         = std::stoi(argv[5]);

    if(use_tpu){
        std::cout << "Use TPU acceleration" << "\n";
    } else {
        std::cout << "No TPU acceleration" << "\n";
    }

    vector<string> labels = load_labels(labels_path);

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

    // Setup for Edge TPU device
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
    // Read original image (BGR)
    Mat orig_bgr = imread(input_path);
    if (orig_bgr.empty()){
        cerr<<"Failed to read image: "<<input_path<<"\n";
        return 1;
    }

    // Letterbox resize to TARGET_SIZE x TARGET_SIZE RGB
    Mat input_rgb;
    float scale = 1.0f;
    int pad_x = 0, pad_y = 0;
    letterbox_resize(orig_bgr, input_rgb, TARGET_SIZE, scale, pad_x, pad_y);

    // Fill input tensor
    TfLiteTensor* in_t = interpreter->input_tensor(0);
    TfLiteType in_type = in_t->type;
    float in_scale = in_t->params.scale;
    int32_t in_zp = in_t->params.zero_point;

    if (in_type == kTfLiteFloat32){
        float* inptr = interpreter->typed_input_tensor<float>(0);
        Mat f;
        input_rgb.convertTo(f, CV_32FC3, 1.0f/255.0f);
        memcpy(inptr, f.data, sizeof(float)*TARGET_SIZE*TARGET_SIZE*3);
    } else if (in_type == kTfLiteUInt8 || in_type == kTfLiteInt8){
        Mat f;
        input_rgb.convertTo(f, CV_32FC3, 1.0f/255.0f);

        if (in_type == kTfLiteUInt8){
            uint8_t* inptr = interpreter->typed_input_tensor<uint8_t>(0);
            size_t idx = 0;
            for (int y=0;y<TARGET_SIZE;++y){
                for (int x=0;x<TARGET_SIZE;++x){
                    Vec3f px = f.at<Vec3f>(y,x);
                    for (int c=0;c<3;++c){
                        int q = (int)lround(px[c] / in_scale) + in_zp;
                        q = max(0, min(255, q));
                        inptr[idx++] = (uint8_t)q;
                    }
                }
            }
        } else { // kTfLiteInt8
            int8_t* inptr = interpreter->typed_input_tensor<int8_t>(0);
            size_t idx = 0;
            for (int y=0;y<TARGET_SIZE;++y){
                for (int x=0;x<TARGET_SIZE;++x){
                    Vec3f px = f.at<Vec3f>(y,x);
                    for (int c=0;c<3;++c){
                        int q = (int)lround(px[c] / in_scale) + in_zp;
                        q = max(-128, min(127, q));
                        inptr[idx++] = (int8_t)q;
                    }
                }
            }
        }
    } else {
        cerr<<"Unsupported input tensor type: "<<in_type<<"\n";
        return 1;
    }
    struct timespec begin, end;
    double latency = 0;
    // Run inference
    clock_gettime(CLOCK_MONOTONIC, &begin);
    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double temp = (end.tv_sec - begin.tv_sec) + ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
    latency += temp;
    printf("inference latency : %.6f sec\n", latency);

    // Read output
    TfLiteTensor* out_t = interpreter->output_tensor(0);
    if (!out_t){
        cerr<<"No output tensor\n";
        return 1;
    }
    int out_type = out_t->type;
    int dims = out_t->dims->size;
    if (dims != 3) {
        cerr<<"Unexpected output dims: "<<dims<<"\n";
    }
    int batch     = out_t->dims->data[0];
    int channels  = out_t->dims->data[1];
    int num_boxes = out_t->dims->data[2];

    float out_scale = out_t->params.scale;
    int32_t out_zp  = out_t->params.zero_point;

    auto read_val = [&](int ch, int box_idx)->float{
        if (out_type == kTfLiteFloat32){
            float* p = interpreter->typed_output_tensor<float>(0);
            return p[ch * (size_t)num_boxes + box_idx];
        } else if (out_type == kTfLiteUInt8){
            uint8_t* p = interpreter->typed_output_tensor<uint8_t>(0);
            return ((float)p[ch * (size_t)num_boxes + box_idx] - out_zp) * out_scale;
        } else {
            int8_t* p = interpreter->typed_output_tensor<int8_t>(0);
            return ((float)p[ch * (size_t)num_boxes + box_idx] - out_zp) * out_scale;
        }
    };

    int cls_count = channels - 4;
    bool has_objectness = false;
    vector<int> known = {80,20,1000,91,60};
    for (int k : known){
        if (channels == 4 + 1 + k){ has_objectness = true; cls_count = k; break; }
    }
    if (!has_objectness){
        cls_count = channels - 4;
    }
    bool output_is_sigmoid = false;
    {
        int sample_count = min(100, num_boxes);
        int in_range_count = 0;
        for (int j=0; j<sample_count; j++){
            int class_offset = has_objectness ? 5 : 4;
            float val = read_val(class_offset, j);
            if (val >= 0.0f && val <= 1.0f) in_range_count++;
        }
        if (in_range_count > sample_count * 0.9f){
            output_is_sigmoid = true;
        }
    }

    // collect candidates
    vector<Detection> candidates;
    for (int j=0;j<num_boxes;j++){
        float bx = read_val(0, j);
        float by = read_val(1, j);
        float bw = read_val(2, j);
        float bh = read_val(3, j);

        float objectness = 1.0f;
        int class_offset = 4;
        if (has_objectness){
            float obj_raw = read_val(4, j);
            objectness = output_is_sigmoid ? obj_raw : sigmoidf(obj_raw);
            class_offset = 5;
        }

        int best_cls = -1;
        float best_prob = -1e9f;
        for (int c=0;c<cls_count;c++){
            float val = read_val(class_offset + c, j);
            float prob = output_is_sigmoid ? val : sigmoidf(val);
            if (prob > best_prob){ best_prob = prob; best_cls = c; }
        }

        float final_score = has_objectness ? (objectness * best_prob) : best_prob;
        if (final_score <= CONF_THRESH_DEFAULT) continue;
        Detection d;
        d.x = bx; d.y = by; d.w = bw; d.h = bh;
        d.score = final_score; d.cls = best_cls;
        candidates.push_back(d);
    }

    // top-K
    sort(candidates.begin(), candidates.end(), [](const Detection&a,const Detection&b){
        return a.score > b.score;
    });
    if ((int)candidates.size() > PRE_NMS_TOPK) candidates.resize(PRE_NMS_TOPK);

    // NMS per class
    vector<Detection> final_dets = nms_classwise(candidates, IOU_THRESH_DEFAULT);

    if ((int)final_dets.size() > MAX_SHOW) final_dets.resize(MAX_SHOW);

    Mat output_bgr = orig_bgr.clone();
    for (size_t i=0;i<final_dets.size();++i){
        Detection &d = final_dets[i];

        Rect box_on_orig = box_to_orig_rect(
            d,
            orig_bgr.cols, orig_bgr.rows,
            scale, pad_x, pad_y
        );

        Scalar col = color_for_class(d.cls);
        rectangle(output_bgr, box_on_orig, col, 2);

        string label = (d.cls >=0 && d.cls < (int)labels.size()) ? labels[d.cls] : to_string(d.cls);
        char buf[64]; snprintf(buf, sizeof(buf), "%.2f", d.score);
        string text = label + " " + string(buf);

        int base_y = max(0, box_on_orig.y - 6);
        putText(output_bgr, text, Point(box_on_orig.x, base_y),
                FONT_HERSHEY_SIMPLEX, 0.5, col, 1);
    }

    if (!imwrite(output_path, output_bgr)){
        cerr<<"Failed to save output to "<<output_path<<"\n";
    } else {
        cout<<"Saved result to "<<output_path<<"\n";
    }

    return 0;
}
