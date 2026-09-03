#include "utils.hpp"
#include <algorithm>
#include <cctype>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;
namespace utils {

std::string normalizeClassName(const std::string& value) {
    std::string out; bool last_space = false;
    for (char ch : value) {
        const unsigned char u = static_cast<unsigned char>(ch);
        if (ch == '_' || ch == '-') ch = ' ';
        if (std::isspace(static_cast<unsigned char>(ch))) {
            if (!last_space) out += ' ';
            last_space = true;
        } else { out += static_cast<char>(std::tolower(u)); last_space = false; }
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    while (!out.empty() && out.front() == ' ') out.erase(out.begin());
    return out;
}
std::string baseName(const std::string& path) { return fs::path(path).stem().string(); }
std::string extensionLower(const std::string& path) {
    std::string e = fs::path(path).extension().string();
    std::transform(e.begin(), e.end(), e.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    return e;
}
bool isImage(const std::string& path) { const auto e=extensionLower(path); return e==".jpg"||e==".jpeg"||e==".png"||e==".bmp"||e==".webp"||e==".tif"||e==".tiff"; }
bool isVideo(const std::string& path) { const auto e=extensionLower(path); return e==".mp4"||e==".avi"||e==".mov"||e==".mkv"||e==".wmv"||e==".m4v"; }
void ensureDirectory(const std::string& path) { if (!path.empty()) fs::create_directories(path); }
std::string jsonEscape(const std::string& value) {
    std::string out;
    for (char c : value) {
        switch(c){case '"':out+="\\\"";break;case '\\':out+="\\\\";break;case '\n':out+="\\n";break;case '\r':out+="\\r";break;case '\t':out+="\\t";break;default:out+=c;}
    }
    return out;
}
static std::string timestampNow() {
    std::time_t t=std::time(nullptr); std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm,&t);
#else
    localtime_r(&t,&tm);
#endif
    std::ostringstream s; s<<std::put_time(&tm,"%Y-%m-%dT%H:%M:%S"); return s.str();
}
std::string detectionJson(const Detection& d, bool include_class_id) {
    std::ostringstream s; s<<std::fixed<<std::setprecision(4)<<"{";
    if(include_class_id)s<<"\"class_id\":"<<d.class_id<<",";
    s<<"\"label\":\""<<jsonEscape(d.label)<<"\",\"confidence\":"<<d.confidence<<",";
    s<<"\"bbox\":["<<d.bbox.x<<","<<d.bbox.y<<","<<d.bbox.x+d.bbox.width<<","<<d.bbox.y+d.bbox.height<<"],";
    s<<"\"status\":\""<<jsonEscape(d.status)<<"\"}"; return s.str();
}
void saveImageLog(const std::string& path,const std::string& filename,const std::string& selected,const std::vector<Detection>& ds){
    std::ofstream f(path); if(!f) throw std::runtime_error("Cannot write log: "+path);
    f<<"{\n  \"timestamp\": \""<<timestampNow()<<"\",\n  \"filename\": \""<<jsonEscape(filename)<<"\",\n  \"media_type\": \"image\",\n  \"selected_class\": \""<<jsonEscape(selected)<<"\",\n  \"detections\": [\n";
    for(size_t i=0;i<ds.size();++i){f<<"    "<<detectionJson(ds[i],false)<<(i+1<ds.size()?",":"")<<"\n";}
    f<<"  ]\n}\n";
}
void saveVideoLog(const std::string& path,const std::string& filename,const std::string& selected,
                  const std::string& positive,const std::string& negative,const VideoSummary& s){
    std::ofstream f(path); if(!f) throw std::runtime_error("Cannot write log: "+path);
    f<<std::fixed<<std::setprecision(4);
    f<<"{\n  \"timestamp\": \""<<timestampNow()<<"\",\n  \"filename\": \""<<jsonEscape(filename)<<"\",\n  \"media_type\": \"video\",\n  \"selected_class\": \""<<jsonEscape(selected)<<"\",\n  \"detections\": {\n";
    f<<"    \""<<jsonEscape(positive)<<"\": {\n      \"frames_detected\": "<<s.positive_frames<<",\n      \"detection_percentage\": "<<std::setprecision(2)<<s.positive_frame_percentage<<",\n      \"object_detections\": "<<s.positive_detection_count<<",\n      \"average_confidence\": "<<std::setprecision(4)<<s.average_positive_confidence<<"\n    },\n";
    f<<"    \""<<jsonEscape(negative)<<"\": {\n      \"frames_detected\": "<<s.negative_frames<<",\n      \"detection_percentage\": "<<std::setprecision(2)<<s.negative_frame_percentage<<",\n      \"object_detections\": "<<s.negative_detection_count<<",\n      \"average_confidence\": "<<std::setprecision(4)<<s.average_negative_confidence<<"\n    }\n  },\n";
    f<<"  \"video_summary\": {\n";
    f<<"    \"total_frames\": "<<s.total_frames<<",\n    \"processed_frames\": "<<s.processed_frames<<",\n    \"frame_skip\": "<<s.frame_skip<<",\n";
    f<<"    \"positive_frames\": "<<s.positive_frames<<",\n    \"negative_frames\": "<<s.negative_frames<<",\n    \"no_detection_frames\": "<<s.no_detection_frames<<",\n";
    f<<"    \"positive_frame_percentage\": "<<std::setprecision(2)<<s.positive_frame_percentage<<",\n    \"negative_frame_percentage\": "<<s.negative_frame_percentage<<",\n    \"no_detection_frame_percentage\": "<<s.no_detection_frame_percentage<<",\n";
    f<<"    \"positive_object_detections\": "<<s.positive_detection_count<<",\n    \"negative_object_detections\": "<<s.negative_detection_count<<",\n";
    f<<"    \"average_positive_confidence\": "<<std::setprecision(4)<<s.average_positive_confidence<<",\n    \"average_negative_confidence\": "<<s.average_negative_confidence<<",\n    \"overall_status\": \""<<jsonEscape(s.overall_status)<<"\"\n  }\n}\n";
}
std::string readWholeFile(const std::string& path){std::ifstream f(path);if(!f)throw std::runtime_error("Cannot read file: "+path);std::stringstream b;b<<f.rdbuf();return b.str();}
void appendGlobalLog(const std::string& path,const std::string& entry){
    ensureDirectory(fs::path(path).parent_path().string());
    std::string body="";
    if(fs::exists(path)){
        std::ifstream in(path); std::stringstream b; b<<in.rdbuf(); std::string old=b.str();
        const auto first=old.find('['), last=old.rfind(']');
        if(first!=std::string::npos && last!=std::string::npos && last>first) body=old.substr(first+1,last-first-1);
    }
    while(!body.empty() && std::isspace(static_cast<unsigned char>(body.back()))) body.pop_back();
    std::ofstream out(path); if(!out)throw std::runtime_error("Cannot write global log: "+path);
    out<<"[\n"; if(!body.empty())out<<body<<",\n"; out<<entry<<"\n]\n";
}
} 
