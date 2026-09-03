#pragma once
#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <string>
#include <vector>
struct Detection { int class_id=-1; std::string label; float confidence=0.0f; cv::Rect bbox; std::string status; };
struct VideoSummary {
 int total_frames=0, processed_frames=0, frame_skip=1;
 int positive_frames=0, negative_frames=0, no_detection_frames=0;
 double positive_frame_percentage=0, negative_frame_percentage=0, no_detection_frame_percentage=0;
 int positive_detection_count=0, negative_detection_count=0;
 double average_positive_confidence=0, average_negative_confidence=0;
 std::string overall_status="no_detection";
};
struct DetectionResult { std::string media_type, filename, selected_class; std::vector<Detection> detections; std::string annotated_file,json_file,overall_status; VideoSummary video_summary; bool has_video_summary=false; };
class GloveDetector {
public:
 GloveDetector(const std::string& model_path,const std::string& classes_path,float confidence_threshold=.25f,float nms_threshold=.45f,int input_size=640);
 std::vector<std::string> classes() const; int findClassId(const std::string& name) const;
 DetectionResult processImage(const std::string& input_path,const std::string& selected_class,const std::string& output_dir,const std::string& log_dir);
 DetectionResult processVideo(const std::string& input_path,const std::string& selected_class,const std::string& output_dir,const std::string& log_dir,int frame_skip=1);
private:
 cv::dnn::Net net_; std::vector<std::string> class_names_; float conf_threshold_,nms_threshold_; int input_size_;
 struct ClassPair{int positive_id=-1,negative_id=-1;std::string positive_name,negative_name;};
 ClassPair findClassPair(const std::string& selected_class) const;
 std::vector<Detection> infer(const cv::Mat& frame,const ClassPair& pair);
 void drawDetections(cv::Mat& frame,const std::vector<Detection>& detections,const ClassPair& pair) const;
};
