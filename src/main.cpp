#include "detector.hpp"
#include "utils.hpp"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace fs=std::filesystem;
static void usage(const char* exe){
 std::cout<<"\nPPE / Glove Detection - C++\n\n"
 <<"Usage:\n  "<<exe<<" --input <file> --class <Gloves|NO-Gloves>\n  "<<exe<<" --list-classes\n\n"
 <<"Options:\n  --model <path>      models/glove_detector.onnx\n  --classes <path>    models/classes.txt\n  --output <dir>      output\n  --logs <dir>        logs\n  --conf <value>      0.25\n  --nms <value>       0.45\n  --imgsz <value>     640\n  --frame-skip <int>  1\n  --help              show help\n";
}
static std::string value(int& i,int argc,char** argv){if(i+1>=argc)throw std::runtime_error("Missing value after "+std::string(argv[i]));return argv[++i];}
int main(int argc,char** argv){
 std::string input, selected, model="models/glove_detector.onnx", classes="models/classes.txt", output="output", logs="logs";
 float conf=.25f,nms=.45f; int imgsz=640, frame_skip=1; bool list=false;
 try{
  for(int i=1;i<argc;++i){std::string a=argv[i];
   if(a=="--input")input=value(i,argc,argv); else if(a=="--class")selected=value(i,argc,argv);
   else if(a=="--model")model=value(i,argc,argv); else if(a=="--classes")classes=value(i,argc,argv);
   else if(a=="--output")output=value(i,argc,argv); else if(a=="--logs")logs=value(i,argc,argv);
   else if(a=="--conf")conf=std::stof(value(i,argc,argv)); else if(a=="--nms")nms=std::stof(value(i,argc,argv));
   else if(a=="--imgsz")imgsz=std::stoi(value(i,argc,argv)); else if(a=="--frame-skip")frame_skip=std::max(1,std::stoi(value(i,argc,argv)));
   else if(a=="--list-classes")list=true; else if(a=="--help"||a=="-h"){usage(argv[0]);return 0;} else throw std::runtime_error("Unknown argument: "+a);
  }
  if(!fs::exists(model)) throw std::runtime_error("ONNX model not found: "+model+"\nRun: python scripts/prepare_model.py --input best.pt");
  GloveDetector detector(model,classes,conf,nms,imgsz);
  if(list){auto cs=detector.classes();std::cout<<"Classes available:\n";for(size_t i=0;i<cs.size();++i)std::cout<<"  "<<i<<": "<<cs[i]<<"\n";return 0;}
  if(input.empty()||selected.empty()){usage(argv[0]);return 1;}
  if(!fs::exists(input))throw std::runtime_error("Input file not found: "+input);
  DetectionResult r;
  if(utils::isImage(input))r=detector.processImage(input,selected,output,logs);
  else if(utils::isVideo(input))r=detector.processVideo(input,selected,output,logs,frame_skip);
  else throw std::runtime_error("Unsupported input extension: "+utils::extensionLower(input));
  std::cout<<"\n========== RESULT ==========\n";
  std::cout<<"{\n  \"success\": true,\n  \"media_type\": \""<<utils::jsonEscape(r.media_type)<<"\",\n  \"filename\": \""<<utils::jsonEscape(r.filename)<<"\",\n  \"selected_class\": \""<<utils::jsonEscape(r.selected_class)<<"\",\n";
  if(!r.has_video_summary){
    std::cout<<"  \"detections\": [\n";
    for(size_t i=0;i<r.detections.size();++i){
      std::cout<<"    "<<utils::detectionJson(r.detections[i],false)<<(i+1<r.detections.size()?",":"")<<"\n";
    }
    std::cout<<"  ],\n";
  } else {
    std::cout<<"  \"overall_status\": \""<<utils::jsonEscape(r.overall_status)<<"\",\n";
    std::cout<<"  \"summary\": {\n"
      <<"    \"total_frames\": "<<r.video_summary.total_frames<<",\n"
      <<"    \"processed_frames\": "<<r.video_summary.processed_frames<<",\n"
      <<"    \"frame_skip\": "<<r.video_summary.frame_skip<<",\n"
      <<"    \"positive_frames\": "<<r.video_summary.positive_frames<<",\n"
      <<"    \"negative_frames\": "<<r.video_summary.negative_frames<<",\n"
      <<"    \"no_detection_frames\": "<<r.video_summary.no_detection_frames<<",\n"
      <<"    \"positive_frame_percentage\": "<<r.video_summary.positive_frame_percentage<<",\n"
      <<"    \"negative_frame_percentage\": "<<r.video_summary.negative_frame_percentage<<",\n"
      <<"    \"no_detection_frame_percentage\": "<<r.video_summary.no_detection_frame_percentage<<",\n"
      <<"    \"positive_object_detections\": "<<r.video_summary.positive_detection_count<<",\n"
      <<"    \"negative_object_detections\": "<<r.video_summary.negative_detection_count<<",\n"
      <<"    \"average_positive_confidence\": "<<r.video_summary.average_positive_confidence<<",\n"
      <<"    \"average_negative_confidence\": "<<r.video_summary.average_negative_confidence<<",\n"
      <<"    \"overall_status\": \""<<utils::jsonEscape(r.video_summary.overall_status)<<"\"\n  },\n";
    std::cout<<"  \"detections\": {} ,\n";
  }
  std::cout<<"  \"annotated_file\": \""<<utils::jsonEscape(r.annotated_file)<<"\",\n  \"json_file\": \""<<utils::jsonEscape(r.json_file)<<"\"\n}\n";
  std::cout<<"============================\n";
  return 0;
 }catch(const std::exception& e){std::cerr<<"\nERROR: "<<e.what()<<"\n";return 2;}
}
