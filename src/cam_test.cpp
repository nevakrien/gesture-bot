#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <iostream>
#include <cstring>

#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

static std::vector<uint8_t> load_file(const std::string& p){
  std::ifstream f(p, std::ios::binary); if(!f) throw std::runtime_error("open fail: "+p);
  f.seekg(0,std::ios::end); size_t n=(size_t)f.tellg(); f.seekg(0);
  std::vector<uint8_t> buf(n); f.read((char*)buf.data(), n); return buf;
}

static uint8_t tensor_arena[320*1024];

static bool open_camera(cv::VideoCapture& cap, int index) {
  // Try a few backends that can work on MinGW builds
  const int backends[] = {
    cv::CAP_DSHOW,     // DirectShow (often available)
    cv::CAP_MSMF,      // Media Foundation (may be OFF in MinGW builds)
    cv::CAP_ANY,       // Let OpenCV choose
    cv::CAP_FFMPEG,    // If enabled, can open some webcams
    cv::CAP_VFW,       // Legacy VFW
    cv::CAP_GSTREAMER  // If gstreamer is installed + enabled
  };
  for (int api : backends) {
    cap.release();
    cap.open(index, api);
    if (cap.isOpened()) {
      std::cout << "[ok] opened camera " << index << " with API=" << api << "\n";
      return true;
    } else {
      std::cout << "[fail] API=" << api << " didn’t open the camera\n";
    }
  }
  return false;
}

int main(int argc, char** argv){
  std::cout << "hello world\n";
  std::cerr << "hello world\n";

  std::string model_path = "models/person_detect.tflite";
  int cam_index = 0;
  if(argc>=2) model_path = argv[1];
  if(argc>=3) cam_index  = std::atoi(argv[2]);

  // Load model
  auto model_buf = load_file(model_path);
  const tflite::Model* model = tflite::GetModel(model_buf.data());
  if(model->version()!=TFLITE_SCHEMA_VERSION){
    std::cerr<<"schema mismatch\n"; return 1;
  }

  // Ops
  tflite::MicroMutableOpResolver<8> r;
  r.AddConv2D(); r.AddDepthwiseConv2D(); r.AddFullyConnected();
  r.AddMaxPool2D(); r.AddReshape(); r.AddSoftmax(); r.AddQuantize(); r.AddDequantize();

  // Interpreter
  tflite::MicroInterpreter I(model, r, tensor_arena, sizeof(tensor_arena));
  if(I.AllocateTensors()!=kTfLiteOk){ std::cerr<<"alloc fail\n"; return 1; }
  TfLiteTensor* in = I.input(0);
  const int H = in->dims->data[in->dims->size-3];
  const int W = in->dims->data[in->dims->size-2];

  // Camera
  cv::VideoCapture cap;
  if(!open_camera(cap, cam_index)){
    std::cerr<<"no webcam (none of the backends worked)\n"; return 2;
  }
  cap.set(cv::CAP_PROP_FRAME_WIDTH,640);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT,480);

  cv::Mat frame, gray, rz;
  std::cout<<"TFLM webcam (person-only)… press q/ESC to quit\n";

  // Make a window once (helps on some backends)
  cv::namedWindow("tflm cam demo", cv::WINDOW_AUTOSIZE);

  while (true){
    if(!cap.read(frame)){
      std::cerr<<"frame grab failed; retrying…\n";
      // brief retry rather than exiting immediately
      cv::waitKey(10);
      continue;
    }

    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, rz, {W,H}, 0,0, cv::INTER_AREA);

    if(in->type==kTfLiteInt8){
      for(int i=0;i<W*H;i++) in->data.int8[i] = int(rz.data[i]) - 128;
    } else if(in->type==kTfLiteUInt8){
      std::memcpy(in->data.uint8, rz.data, size_t(W)*H);
    } else {
      std::cerr<<"unexpected input type\n"; break;
    }

    bool person_yes=false;
    if(I.Invoke()==kTfLiteOk){
      const TfLiteTensor* out=I.output(0);
      if(out->type==kTfLiteInt8){
        int no=out->data.int8[0], yes=out->data.int8[1]; person_yes=(yes-no)>20;
      } else if(out->type==kTfLiteUInt8){
        int no=out->data.uint8[0], yes=out->data.uint8[1]; person_yes=(yes-no)>20;
      }
    } else {
      std::cerr<<"invoke failed\n";
    }

    cv::putText(frame, std::string("person: ")+(person_yes?"YES":"no"),
                {20,30}, cv::FONT_HERSHEY_SIMPLEX, 0.8,
                person_yes?cv::Scalar(0,255,0):cv::Scalar(0,0,255),2);
    cv::imshow("tflm cam demo", frame);

    // give GUI time; 30ms is more reliable than 1ms
    int k=cv::waitKey(30);
    if(k=='q'||k==27) break;
  }
  return 0;
}