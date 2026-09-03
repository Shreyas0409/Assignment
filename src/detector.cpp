#include "detector.hpp"
#include "utils.hpp"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace {

struct LetterboxInfo {
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
};

LetterboxInfo letterbox(const cv::Mat& src, cv::Mat& dst, int size) {
    if (src.empty()) throw std::runtime_error("Cannot letterbox an empty image.");
    const float scale = std::min(static_cast<float>(size) / src.cols,
                                 static_cast<float>(size) / src.rows);
    const int nw = std::max(1, static_cast<int>(std::round(src.cols * scale)));
    const int nh = std::max(1, static_cast<int>(std::round(src.rows * scale)));
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
    const int left = (size - nw) / 2;
    const int top = (size - nh) / 2;
    cv::copyMakeBorder(resized, dst, top, size - nh - top, left, size - nw - left,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return {scale, left, top};
}

cv::Rect xyxyToOriginal(float x1, float y1, float x2, float y2,
                        const LetterboxInfo& lb, const cv::Size& original) {
    x1 = (x1 - lb.pad_x) / lb.scale;
    y1 = (y1 - lb.pad_y) / lb.scale;
    x2 = (x2 - lb.pad_x) / lb.scale;
    y2 = (y2 - lb.pad_y) / lb.scale;

    int ix1 = std::clamp(static_cast<int>(std::floor(x1)), 0, original.width - 1);
    int iy1 = std::clamp(static_cast<int>(std::floor(y1)), 0, original.height - 1);
    int ix2 = std::clamp(static_cast<int>(std::ceil(x2)), 0, original.width - 1);
    int iy2 = std::clamp(static_cast<int>(std::ceil(y2)), 0, original.height - 1);
    if (ix2 < ix1) std::swap(ix1, ix2);
    if (iy2 < iy1) std::swap(iy1, iy2);
    return cv::Rect(cv::Point(ix1, iy1), cv::Point(ix2, iy2));
}

cv::Mat outputToRows(const cv::Mat& out, int class_count) {
    if (out.empty()) return {};
    if (out.dims == 2) return out;
    if (out.dims != 3) throw std::runtime_error("Unsupported ONNX output dimensions: " + std::to_string(out.dims));

    const int a = out.size[1];
    const int b = out.size[2];
    cv::Mat flat = out.reshape(1, a);
    if (a == 6 || a == 4 + class_count) return flat.t();
    if (b == 6 || b == 4 + class_count) return flat;
    if (a < b && a >= 6) return flat.t();
    return flat;
}

} 

GloveDetector::GloveDetector(const std::string& model_path,
                             const std::string& classes_path,
                             float confidence_threshold,
                             float nms_threshold,
                             int input_size)
    : conf_threshold_(confidence_threshold),
      nms_threshold_(nms_threshold),
      input_size_(input_size) {
    if (input_size_ <= 0) throw std::runtime_error("Input size must be > 0.");
    std::ifstream labels(classes_path);
    if (!labels) throw std::runtime_error("Cannot open classes file: " + classes_path);

    std::string line;
    while (std::getline(labels, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty() && line[0] != '#') class_names_.push_back(line);
    }
    if (class_names_.empty()) throw std::runtime_error("Classes file is empty: " + classes_path);

    try {
        net_ = cv::dnn::readNetFromONNX(model_path);
    } catch (const cv::Exception& e) {
        throw std::runtime_error("Could not load ONNX model '" + model_path + "': " + e.what());
    }
    if (net_.empty()) throw std::runtime_error("Could not load ONNX model: " + model_path);
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

std::vector<std::string> GloveDetector::classes() const { return class_names_; }

int GloveDetector::findClassId(const std::string& name) const {
    const std::string target = utils::normalizeClassName(name);
    for (size_t i = 0; i < class_names_.size(); ++i) {
        if (utils::normalizeClassName(class_names_[i]) == target) return static_cast<int>(i);
    }
    return -1;
}

GloveDetector::ClassPair GloveDetector::findClassPair(const std::string& selected_class) const {
    std::string selected = utils::normalizeClassName(selected_class);
    std::string positive = selected;
    if (positive.rfind("no ", 0) == 0) positive = positive.substr(3);

    const int positive_id = findClassId(positive);
    if (positive_id < 0) throw std::runtime_error("Selected class not found in model classes: " + selected_class);

    ClassPair pair;
    pair.positive_id = positive_id;
    pair.positive_name = class_names_[positive_id];

    const std::string negative = "no " + utils::normalizeClassName(pair.positive_name);
    pair.negative_id = findClassId(negative);
    if (pair.negative_id >= 0) pair.negative_name = class_names_[pair.negative_id];
    else pair.negative_name = "NO-" + pair.positive_name;
    return pair;
}

std::vector<Detection> GloveDetector::infer(const cv::Mat& frame, const ClassPair& pair) {
    cv::Mat padded;
    const LetterboxInfo lb = letterbox(frame, padded, input_size_);
    cv::Mat blob = cv::dnn::blobFromImage(padded, 1.0 / 255.0,
                                          cv::Size(input_size_, input_size_),
                                          cv::Scalar(), true, false, CV_32F);
    net_.setInput(blob);

    std::vector<cv::Mat> outputs;
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());
    if (outputs.empty()) return {};

    const int class_count = static_cast<int>(class_names_.size());
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> ids;

    for (const auto& raw : outputs) {
        cv::Mat out = outputToRows(raw, class_count);
        if (out.empty() || out.cols < 6) continue;

        if (out.cols == 6) {
            for (int i = 0; i < out.rows; ++i) {
                const float* p = out.ptr<float>(i);
                const float conf = p[4];
                const int cid = static_cast<int>(std::round(p[5]));
                if (conf < conf_threshold_) continue;
                if (cid != pair.positive_id && cid != pair.negative_id) continue;
                cv::Rect r = xyxyToOriginal(p[0], p[1], p[2], p[3], lb, frame.size());
                if (r.width > 0 && r.height > 0) {
                    boxes.push_back(r); scores.push_back(conf); ids.push_back(cid);
                }
            }
            continue;
        }

        const int expected = 4 + class_count;
        if (out.cols < expected) continue;
        for (int i = 0; i < out.rows; ++i) {
            const float* p = out.ptr<float>(i);
            float best_score = -std::numeric_limits<float>::infinity();
            int best_class = -1;
            for (int c = 0; c < class_count; ++c) {
                const float score = p[4 + c];
                if (score > best_score) { best_score = score; best_class = c; }
            }
            if (best_class < 0 || best_score < conf_threshold_) continue;
            if (best_class != pair.positive_id && best_class != pair.negative_id) continue;

            const float cx = p[0], cy = p[1], w = p[2], h = p[3];
            cv::Rect r = xyxyToOriginal(cx - w * 0.5f, cy - h * 0.5f,
                                        cx + w * 0.5f, cy + h * 0.5f, lb, frame.size());
            if (r.width > 0 && r.height > 0) {
                boxes.push_back(r); scores.push_back(best_score); ids.push_back(best_class);
            }
        }
    }

    std::vector<int> keep_all;
    for (int wanted_class : {pair.positive_id, pair.negative_id}) {
        if (wanted_class < 0) continue;
        std::vector<cv::Rect> class_boxes;
        std::vector<float> class_scores;
        std::vector<int> original_indices;
        for (size_t i = 0; i < boxes.size(); ++i) {
            if (ids[i] == wanted_class) {
                class_boxes.push_back(boxes[i]);
                class_scores.push_back(scores[i]);
                original_indices.push_back(static_cast<int>(i));
            }
        }
        std::vector<int> keep;
        cv::dnn::NMSBoxes(class_boxes, class_scores, conf_threshold_, nms_threshold_, keep);
        for (int k : keep) keep_all.push_back(original_indices[k]);
    }

    std::sort(keep_all.begin(), keep_all.end(), [&](int a, int b) { return scores[a] > scores[b]; });
    std::vector<Detection> detections;
    detections.reserve(keep_all.size());
    for (int idx : keep_all) {
        Detection d;
        d.class_id = ids[idx];
        d.label = class_names_.at(static_cast<size_t>(d.class_id));
        d.confidence = scores[idx];
        d.bbox = boxes[idx];
        d.status = (d.class_id == pair.positive_id) ? "wearing" : "not_wearing";
        detections.push_back(d);
    }
    return detections;
}

void GloveDetector::drawDetections(cv::Mat& frame, const std::vector<Detection>& detections,
                                   const ClassPair& pair) const {
    for (const auto& d : detections) {
        const cv::Scalar color = (d.class_id == pair.positive_id)
            ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255);
        cv::rectangle(frame, d.bbox, color, 2, cv::LINE_AA);
        std::ostringstream label;
        label.setf(std::ios::fixed); label.precision(2);
        label << d.label << " " << d.confidence;
        const std::string text = label.str();
        int baseline = 0;
        const cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.55, 2, &baseline);
        const int x = std::max(0, d.bbox.x);
        const int y = std::max(ts.height + baseline + 4, d.bbox.y);
        cv::rectangle(frame, cv::Rect(x, y-ts.height-baseline-4, ts.width+6, ts.height+baseline+4),
                      color, cv::FILLED);
        cv::putText(frame, text, cv::Point(x+3, y-baseline-2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0,0,0), 2, cv::LINE_AA);
    }
}

DetectionResult GloveDetector::processImage(const std::string& input_path,
                                            const std::string& selected_class,
                                            const std::string& output_dir,
                                            const std::string& log_dir) {
    const auto pair = findClassPair(selected_class);
    cv::Mat image = cv::imread(input_path);
    if (image.empty()) throw std::runtime_error("Unable to read image: " + input_path);
    const auto detections = infer(image, pair);
    drawDetections(image, detections, pair);

    utils::ensureDirectory(output_dir); utils::ensureDirectory(log_dir);
    const std::string output = (fs::path(output_dir) / (utils::baseName(input_path) + "_annotated.jpg")).string();
    const std::string log = (fs::path(log_dir) / (utils::baseName(input_path) + ".json")).string();
    if (!cv::imwrite(output, image)) throw std::runtime_error("Failed to save annotated image: " + output);
    utils::saveImageLog(log, fs::path(input_path).filename().string(), selected_class, detections);
    utils::appendGlobalLog((fs::path(log_dir) / "detections.json").string(), utils::readWholeFile(log));

    DetectionResult result;
    result.media_type = "image"; result.filename = fs::path(input_path).filename().string();
    result.selected_class = selected_class; result.detections = detections;
    result.annotated_file = output; result.json_file = log;
    return result;
}

DetectionResult GloveDetector::processVideo(const std::string& input_path,
                                            const std::string& selected_class,
                                            const std::string& output_dir,
                                            const std::string& log_dir,
                                            int frame_skip) {
    if (frame_skip < 1) frame_skip = 1;
    const auto pair = findClassPair(selected_class);
    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) throw std::runtime_error("Unable to open video: " + input_path);

    const int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (!(fps > 0.0) || !std::isfinite(fps)) fps = 25.0;
    const int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    if (width <= 0 || height <= 0) { cap.release(); throw std::runtime_error("Invalid video dimensions."); }

    utils::ensureDirectory(output_dir); utils::ensureDirectory(log_dir);
    const std::string output = (fs::path(output_dir) / (utils::baseName(input_path) + "_annotated.mp4")).string();
    const std::string log = (fs::path(log_dir) / (utils::baseName(input_path) + ".json")).string();

    cv::VideoWriter writer(output, cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(width,height));
    if (!writer.isOpened()) {
        writer = cv::VideoWriter(output, cv::VideoWriter::fourcc('a','v','c','1'), fps, cv::Size(width,height));
    }
    if (!writer.isOpened()) throw std::runtime_error("Could not create output video: " + output);

    VideoSummary summary;
    summary.total_frames = total_frames; summary.processed_frames = 0; summary.frame_skip = frame_skip;
    summary.positive_frames = 0; summary.negative_frames = 0; summary.no_detection_frames = 0;
    summary.positive_frame_percentage = summary.negative_frame_percentage = summary.no_detection_frame_percentage = 0.0;
    summary.positive_detection_count = summary.negative_detection_count = 0;
    summary.average_positive_confidence = summary.average_negative_confidence = 0.0;
    summary.overall_status = "no_detection";

    double pos_sum = 0.0, neg_sum = 0.0;
    cv::Mat frame;
    int frame_number = 1;
    while (cap.read(frame)) {
        const bool process = ((frame_number - 1) % frame_skip == 0);
        if (process) {
            ++summary.processed_frames;
            const auto detections = infer(frame, pair);
            bool pos = false, neg = false;
            for (const auto& d : detections) {
                if (d.class_id == pair.positive_id) { pos = true; ++summary.positive_detection_count; pos_sum += d.confidence; }
                else if (d.class_id == pair.negative_id) { neg = true; ++summary.negative_detection_count; neg_sum += d.confidence; }
            }
            if (pos) ++summary.positive_frames;
            else if (neg) ++summary.negative_frames;
            else ++summary.no_detection_frames;
            drawDetections(frame, detections, pair);
        }
        writer.write(frame);
        if (summary.total_frames > 0 && frame_number % 100 == 0) {
            std::cout << "Progress: " << (100.0 * frame_number / summary.total_frames)
                      << "% (" << frame_number << "/" << summary.total_frames << ")\n";
        }
        ++frame_number;
    }
    cap.release(); writer.release();
    if (!fs::exists(output) || fs::file_size(output) == 0) throw std::runtime_error("Annotated video was not created: " + output);

    if (summary.processed_frames > 0) {
        summary.positive_frame_percentage = 100.0 * summary.positive_frames / summary.processed_frames;
        summary.negative_frame_percentage = 100.0 * summary.negative_frames / summary.processed_frames;
        summary.no_detection_frame_percentage = 100.0 * summary.no_detection_frames / summary.processed_frames;
    }
    if (summary.positive_frames == 0 && summary.negative_frames == 0) summary.overall_status = "no_detection";
    else if (summary.positive_frames >= summary.negative_frames) summary.overall_status = "wearing";
    else summary.overall_status = "not_wearing";
    if (summary.positive_detection_count > 0) summary.average_positive_confidence = pos_sum / summary.positive_detection_count;
    if (summary.negative_detection_count > 0) summary.average_negative_confidence = neg_sum / summary.negative_detection_count;

    utils::saveVideoLog(log, fs::path(input_path).filename().string(), selected_class,
                        pair.positive_name, pair.negative_name, summary);
    utils::appendGlobalLog((fs::path(log_dir) / "detections.json").string(), utils::readWholeFile(log));

    DetectionResult result;
    result.media_type = "video"; result.filename = fs::path(input_path).filename().string();
    result.selected_class = selected_class; result.annotated_file = output; result.json_file = log;
    result.overall_status = summary.overall_status; result.video_summary = summary; result.has_video_summary = true;
    return result;
}
