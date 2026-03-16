# AI Traffic Signal Optimization

This project uses a pretrained YOLOv8 model to detect vehicles from a traffic image, video, or webcam feed and then recommends a green signal duration based on the detected traffic density.

## Features

- Detects `car`, `motorcycle`, `bus`, and `truck`
- Estimates traffic level as `LOW`, `MEDIUM`, or `HIGH`
- Suggests green signal time automatically
- Works with images, recorded video, or webcam input
- Saves annotated output if needed
- Includes a simple Streamlit UI for demo presentations

## Project Flow

1. Capture traffic input from camera, image, or video.
2. Run YOLOv8 object detection.
3. Count detected vehicles.
4. Classify traffic density.
5. Recommend signal timing.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Streamlit UI

```bash
streamlit run ui.py
```

Then open the local URL shown in the terminal, upload an image, and click `Analyze Traffic`.

### Image input

```bash
python app.py --source traffic.jpg --output outputs/traffic_result.jpg
```

### Video input

```bash
python app.py --source traffic.mp4 --output outputs/traffic_result.mp4
```

### Webcam input

```bash
python app.py --source webcam
```

Press `q` to stop video or webcam mode.

## Example Logic

- Fewer than 5 vehicles: `LOW` traffic, `20` seconds green light
- 5 to 14 vehicles: `MEDIUM` traffic, `40` seconds green light
- 15 or more vehicles: `HIGH` traffic, `60` seconds green light

## Notes

- The default model is `yolov8n.pt`.
- The pretrained weights are downloaded automatically by Ultralytics on first run if not already present.
- `yolov8n.pt` is trained on the COCO dataset, which includes common road vehicles.
