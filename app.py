from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
from ultralytics import YOLO


VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}


@dataclass
class TrafficDecision:
    vehicle_count: int
    traffic_level: str
    green_time_seconds: int


def decide_signal_timing(vehicle_count: int) -> TrafficDecision:
    if vehicle_count < 5:
        return TrafficDecision(vehicle_count, "LOW", 20)
    if vehicle_count < 15:
        return TrafficDecision(vehicle_count, "MEDIUM", 40)
    return TrafficDecision(vehicle_count, "HIGH", 60)


def count_vehicles(result, class_names: dict[int, str]) -> int:
    total = 0
    for box in result.boxes:
        cls_id = int(box.cls[0])
        if class_names.get(cls_id) in VEHICLE_CLASSES:
            total += 1
    return total


def draw_overlay(frame, decision: TrafficDecision) -> None:
    lines = [
        f"Vehicles: {decision.vehicle_count}",
        f"Traffic: {decision.traffic_level}",
        f"Green Time: {decision.green_time_seconds}s",
    ]

    y = 30
    for line in lines:
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 35


def analyze_frame(model: YOLO, frame):
    result = model(frame, verbose=False)[0]
    vehicle_count = count_vehicles(result, model.names)
    decision = decide_signal_timing(vehicle_count)
    annotated = result.plot()
    draw_overlay(annotated, decision)
    return annotated, decision


def process_image(model: YOLO, image_path: Path, output_path: Path | None) -> None:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    annotated, decision = analyze_frame(model, frame)

    print(f"Vehicles detected: {decision.vehicle_count}")
    print(f"Traffic level: {decision.traffic_level}")
    print(f"Green signal time: {decision.green_time_seconds} seconds")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)
        print(f"Saved output to: {output_path}")

    cv2.imshow("AI Traffic Signal Optimization", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def iter_video_frames(capture: cv2.VideoCapture) -> Iterable[tuple[bool, any]]:
    while True:
        ok, frame = capture.read()
        yield ok, frame
        if not ok:
            return


def process_video(model: YOLO, source: str, output_path: Path | None) -> None:
    capture = cv2.VideoCapture(0 if source == "webcam" else source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    writer = None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    try:
        for ok, frame in iter_video_frames(capture):
            if not ok:
                break

            annotated, decision = analyze_frame(model, frame)

            print(
                f"Vehicles: {decision.vehicle_count} | "
                f"Traffic: {decision.traffic_level} | "
                f"Green Time: {decision.green_time_seconds}s",
                end="\r",
            )

            cv2.imshow("AI Traffic Signal Optimization", annotated)
            if writer:
                writer.write(annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        capture.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    if output_path:
        print(f"\nSaved output video to: {output_path}")
    else:
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect vehicles with YOLOv8 and estimate traffic signal timing."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to an image, path to a video, or the word 'webcam'.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO model name or local model path.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the annotated image or video.",
    )
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    output_path = Path(args.output) if args.output else None

    if args.source.lower() == "webcam":
        process_video(model, "webcam", output_path)
        return

    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    if is_image_file(source_path):
        process_image(model, source_path, output_path)
    else:
        process_video(model, str(source_path), output_path)


if __name__ == "__main__":
    main()
