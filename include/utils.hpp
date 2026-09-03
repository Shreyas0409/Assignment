#pragma once
#include "detector.hpp"
#include <string>
#include <vector>

namespace utils {
std::string normalizeClassName(const std::string& value);
std::string baseName(const std::string& path);
std::string extensionLower(const std::string& path);
bool isImage(const std::string& path);
bool isVideo(const std::string& path);
void ensureDirectory(const std::string& path);
std::string jsonEscape(const std::string& value);
std::string detectionJson(const Detection& d, bool include_class_id);
void saveImageLog(const std::string& json_path, const std::string& filename,
                  const std::string& selected_class, const std::vector<Detection>& detections);
void saveVideoLog(const std::string& json_path, const std::string& filename,
                  const std::string& selected_class, const std::string& positive_name,
                  const std::string& negative_name, const VideoSummary& summary);
void appendGlobalLog(const std::string& global_path, const std::string& entry_json);
std::string readWholeFile(const std::string& path);
}
