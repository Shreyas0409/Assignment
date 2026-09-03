import argparse
import json
import os
import sys
from datetime import datetime

import cv2
from ultralytics import YOLO


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best2002.pt")
CLASS_FILE = os.path.join(BASE_DIR, "Data.txt")
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR = os.path.join(BASE_DIR, "logs")
GLOBAL_LOG_FILE = os.path.join(LOG_DIR, "detections.json")

CONFIDENCE_THRESHOLD = 0.18
FRAME_SKIP = 1

DEFAULT_VIDEO_FPS = 25.0

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def ensure_directories():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


def load_model(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"YOLO model was not found.\n\n"
            f"Expected model location:\n{model_path}\n\n"
            f"Put best2002.pt in the same folder as this script."
)
    return YOLO(model_path)


def load_model_classes(model):
    names = model.names

    if isinstance(names, dict):
        return {int(class_id): str(name) for class_id, name in names.items()}

    return {class_id: str(name) for class_id, name in enumerate(names)}


def build_class_lookup(model_classes):
    lookup = {}
    for class_id, class_name in model_classes.items():
        lookup[normalize_class_name(class_name)] = class_id
    return lookup


def load_data_classes(class_file=CLASS_FILE):
    classes = []
    if not os.path.exists(class_file):
        return classes
    try:
        with open(class_file, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                if ":" in line:
                    _, class_name = line.split(":", 1)
                    class_name = class_name.strip()
                else:
                    class_name = line
                if class_name:
                    classes.append(class_name)
    except OSError:
        pass
    return classes


def normalize_class_name(name):
    if name is None:
        return ""
    name = str(name).strip().lower()
    name = name.replace("-", " ").replace("_", " ")
    name = " ".join(name.split())
    return name


def find_model_class_id(class_name, class_lookup, model_classes):
    normalized_name = normalize_class_name(class_name)
    if normalized_name not in class_lookup:
        available = ", ".join(model_classes.values())
        raise ValueError(
            f"Class '{class_name}' was not found in the YOLO model.\n\n"
            f"Available classes:\n{available}"
        )
    return class_lookup[normalized_name]


def find_class_pair(selected_class, class_lookup, model_classes):

    normalized = normalize_class_name(selected_class)
    if not normalized:
        raise ValueError("Class name cannot be empty.")
    if normalized.startswith("no "):
        positive_normalized = normalized[3:].strip()
    else:
        positive_normalized = normalized
    negative_normalized = "no " + positive_normalized
    positive_id = find_model_class_id(positive_normalized, class_lookup, model_classes)
    negative_id = find_model_class_id(negative_normalized, class_lookup, model_classes)

    return {
        "positive_id": positive_id,
        "negative_id": negative_id,
        "positive_name": model_classes[positive_id],
        "negative_name": model_classes[negative_id],
        "allowed_ids": [positive_id, negative_id],
    }


def get_all_available_classes(model_classes):
    """Returns every class contained in the YOLO model."""
    return [{"id": class_id, "name": name} for class_id, name in model_classes.items()]


def get_box_color(class_id, pair):
    if class_id == pair["positive_id"]:
        return (0, 255, 0)
    if class_id == pair["negative_id"]:
        return (0, 0, 255)
    return (255, 255, 255)


def get_detection_status(class_id, pair):
    if class_id == pair["positive_id"]:
        return "wearing"
    if class_id == pair["negative_id"]:
        return "not_wearing"
    return "unknown"


def draw_detection_box(image, detection, pair):
    class_id = detection["class_id"]
    x1, y1, x2, y2 = detection["bbox"]
    confidence = detection["confidence"]
    label = detection["label"]
    color = get_box_color(class_id, pair)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    label_x = max(0, x1)
    label_y = max(text_height + baseline, y1)
    cv2.rectangle(image,(label_x, label_y - text_height - baseline),(label_x + text_width, label_y),color,-1,)
    cv2.putText(image, text, (label_x, label_y - baseline),font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA,)


def process_results(results, image_width, image_height, pair, model_classes):
    detections = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())

            x1 = max(0, min(x1, image_width - 1))
            y1 = max(0, min(y1, image_height - 1))
            x2 = max(0, min(x2, image_width - 1))
            y2 = max(0, min(y2, image_height - 1))

            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1

            label = model_classes.get(class_id, str(class_id))
            status = get_detection_status(class_id, pair)

            detections.append({
                "class_id": class_id,
                "label": label,
                "confidence": round(confidence, 4),
                "bbox": [x1, y1, x2, y2],
                "status": status,
            })
    return detections


def make_api_detection(detection):
    return {
        "label": detection["label"],
        "confidence": detection["confidence"],
        "bbox": detection["bbox"],
        "status": detection["status"],
    }

def save_detection_log(filename, selected_class, media_type, detections, video_summary=None):
    name, _ = os.path.splitext(filename)
    json_path = os.path.join(LOG_DIR, f"{name}.json")
    log_data = {"timestamp": datetime.now().isoformat(),"filename": filename,"media_type": media_type,"selected_class": selected_class,"detections": detections,}
    if video_summary is not None:
        log_data["video_summary"] = video_summary
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(log_data, file, indent=4)
    return json_path


def update_global_log(filename, selected_class, media_type, detections, video_summary=None):
    existing_logs = []
    if os.path.exists(GLOBAL_LOG_FILE):
        try:
            with open(GLOBAL_LOG_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)
                if isinstance(data, list):
                    existing_logs = data
        except (json.JSONDecodeError, OSError):
            existing_logs = []
    new_entry = {"timestamp": datetime.now().isoformat(),"filename": filename,"media_type": media_type,"selected_class": selected_class,"detections": detections,}
    if video_summary is not None:
        new_entry["video_summary"] = video_summary
    existing_logs.append(new_entry)
    with open(GLOBAL_LOG_FILE, "w", encoding="utf-8") as file:
        json.dump(existing_logs, file, indent=4)


def save_annotated_image(image, input_filename):
    name, _ = os.path.splitext(input_filename)
    output_filename = f"{name}_annotated.jpg"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    if not cv2.imwrite(output_path, image):
        raise RuntimeError(f"Failed to save annotated image:\n{output_path}")
    return output_path


def process_image(image_path, selected_class, model, class_lookup, model_classes):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found:\n{image_path}")
    pair = find_class_pair(selected_class, class_lookup, model_classes)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image:\n{image_path}")
    image_height, image_width = image.shape[:2]
    filename = os.path.basename(image_path)
    results = model.predict(source=image,conf=CONFIDENCE_THRESHOLD,classes=pair["allowed_ids"],verbose=False,)
    detections = process_results(results, image_width, image_height, pair, model_classes)

    for detection in detections:
        draw_detection_box(image, detection, pair)
    output_path = save_annotated_image(image, filename)
    api_detections = [make_api_detection(d) for d in detections]
    json_path = save_detection_log(filename, selected_class, "image", detections)
    update_global_log(filename, selected_class, "image", detections)
    return {
        "output_path": output_path,
        "json_path": json_path,
        "detections": api_detections,
    }


def create_video_writer(output_path, fps, width, height):
    for codec in ("mp4v", "avc1"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if writer.isOpened():
            return writer
    raise RuntimeError("Could not create video writer.")


def process_video(video_path, selected_class, model, class_lookup, model_classes):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found:\n{video_path}")
    pair = find_class_pair(selected_class, class_lookup, model_classes)
    filename = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video:\n{video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0 or original_fps != original_fps:
        original_fps = DEFAULT_VIDEO_FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Invalid video dimensions.")
    name, _ = os.path.splitext(filename)
    output_path = os.path.join(OUTPUT_DIR, f"{name}_annotated.mp4")
    writer = create_video_writer(output_path, original_fps, width, height)
    frame_number = 0
    processed_frames = 0
    positive_frames = 0
    negative_frames = 0
    no_detection_frames = 0
    positive_detection_count = 0
    negative_detection_count = 0
    positive_confidence_sum = 0.0
    negative_confidence_sum = 0.0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_number += 1
        if FRAME_SKIP > 1 and (frame_number - 1) % FRAME_SKIP != 0:
            writer.write(frame)
            continue
        processed_frames += 1
        results = model.predict(source=frame,conf=CONFIDENCE_THRESHOLD,classes=pair["allowed_ids"],verbose=False,)
        frame_detections = process_results(results, width, height, pair, model_classes)

        for detection in frame_detections:
            draw_detection_box(frame, detection, pair)
        frame_positive = [d for d in frame_detections if d["status"] == "wearing"]
        frame_negative = [d for d in frame_detections if d["status"] == "not_wearing"]
        positive_detection_count += len(frame_positive)
        negative_detection_count += len(frame_negative)
        positive_confidence_sum += sum(d["confidence"] for d in frame_positive)
        negative_confidence_sum += sum(d["confidence"] for d in frame_negative)

        if frame_positive and frame_negative:
            strongest_positive = max(frame_positive, key=lambda d: d["confidence"])
            strongest_negative = max(frame_negative, key=lambda d: d["confidence"])
            if strongest_positive["confidence"] >= strongest_negative["confidence"]:
                positive_frames += 1
            else:
                negative_frames += 1
        elif frame_positive:
            positive_frames += 1
        elif frame_negative:
            negative_frames += 1
        else:
            no_detection_frames += 1

        writer.write(frame)

    cap.release()
    writer.release()

    if not os.path.exists(output_path):
        raise RuntimeError("Annotated video was not created.")

    if processed_frames > 0:
        positive_percentage = (positive_frames / processed_frames) * 100
        negative_percentage = (negative_frames / processed_frames) * 100
        no_detection_percentage = (no_detection_frames / processed_frames) * 100
    else:
        positive_percentage = negative_percentage = no_detection_percentage = 0.0

    if positive_frames == 0 and negative_frames == 0:
        overall_status = "no_detection"
    elif positive_frames >= negative_frames:
        overall_status = "wearing"
    else:
        overall_status = "not_wearing"

    average_positive_confidence = (
        positive_confidence_sum / positive_detection_count if positive_detection_count > 0 else 0.0
    )
    average_negative_confidence = (
        negative_confidence_sum / negative_detection_count if negative_detection_count > 0 else 0.0
    )

    video_summary = {
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "frame_skip": FRAME_SKIP,
        "positive_frames": positive_frames,
        "negative_frames": negative_frames,
        "no_detection_frames": no_detection_frames,
        "positive_frame_percentage": round(positive_percentage, 2),
        "negative_frame_percentage": round(negative_percentage, 2),
        "no_detection_frame_percentage": round(no_detection_percentage, 2),
        "positive_object_detections": positive_detection_count,
        "negative_object_detections": negative_detection_count,
        "average_positive_confidence": round(average_positive_confidence, 4),
        "average_negative_confidence": round(average_negative_confidence, 4),
        "overall_status": overall_status,
    }

    video_detections = {
        pair["positive_name"]: {
            "frames_detected": positive_frames,
            "detection_percentage": round(positive_percentage, 2),
            "object_detections": positive_detection_count,
            "average_confidence": round(average_positive_confidence, 4),
        },
        pair["negative_name"]: {
            "frames_detected": negative_frames,
            "detection_percentage": round(negative_percentage, 2),
            "object_detections": negative_detection_count,
            "average_confidence": round(average_negative_confidence, 4),
        },
    }

    json_path = save_detection_log(
        filename, selected_class, "video", video_detections, video_summary
    )
    update_global_log(filename, selected_class, "video", video_detections, video_summary)

    return {
        "output_path": output_path,
        "json_path": json_path,
        "overall_status": overall_status,
        "video_summary": video_summary,
        "detections": video_detections,
    }


def get_media_type(filename):
    extension = os.path.splitext(filename)[1].lower()
    if extension in IMAGE_EXTENSIONS:
        return "image"
    if extension in VIDEO_EXTENSIONS:
        return "video"
    raise ValueError(
        f"Unsupported file type: {extension}\n\n"
        f"Supported image types: {', '.join(sorted(IMAGE_EXTENSIONS))}\n\n"
        f"Supported video types: {', '.join(sorted(VIDEO_EXTENSIONS))}"
    )


def run_detect(args, model, class_lookup, model_classes):
    filename = os.path.basename(args.filename.strip())
    selected_class = args.class_name.strip()
    if not filename:
        sys.exit("Error: filename cannot be empty.")
    if not selected_class:
        sys.exit("Error: class name cannot be empty.")

    media_type = get_media_type(filename)
    input_path = os.path.join(INPUT_DIR, filename)

    if not os.path.exists(input_path):
        sys.exit(f"Error: file not found:\n{input_path}")
    find_class_pair(selected_class, class_lookup, model_classes)

    if media_type == "image":
        result = process_image(input_path, selected_class, model, class_lookup, model_classes)
        output = {"success": True,"media_type": "image","filename": filename,"selected_class": selected_class,"detections": result["detections"],"annotated_file": result["output_path"],"json_file": result["json_path"],}
    else:
        result = process_video(input_path, selected_class, model, class_lookup, model_classes)
        output = {"success": True,"media_type": "video","filename": filename,"selected_class": selected_class,"overall_status": result["overall_status"],"summary": result["video_summary"],"detections": result["detections"],"annotated_file": result["output_path"],"json_file": result["json_path"],}

    print(json.dumps(output, indent=4))


def run_list_classes(model_classes):
    output = {"success": True, "classes": get_all_available_classes(model_classes)}
    print(json.dumps(output, indent=4))


def build_arg_parser():
    parser = argparse.ArgumentParser(description="PPE Detection CLI (YOLO-based)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect_parser = subparsers.add_parser("detect", help="Run detection on a file in the input/ folder")
    detect_parser.add_argument("--filename", required=True, help="Image or video filename inside input/")
    detect_parser.add_argument(
        "--class-name", dest="class_name", required=True, help="PPE class to check, e.g. Gloves"
    )

    subparsers.add_parser("list-classes", help="List all classes available in the YOLO model")
    return parser


def main():
    ensure_directories()
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        model = load_model()
    except FileNotFoundError as error:
        sys.exit(str(error))
    model_classes = load_model_classes(model)
    class_lookup = build_class_lookup(model_classes)
    try:
        if args.command == "detect":
            run_detect(args, model, class_lookup, model_classes)
        elif args.command == "list-classes":
            run_list_classes(model_classes)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        sys.exit(f"Error: {error}")


if __name__ == "__main__":
    main()