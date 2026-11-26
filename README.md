# Windows NPU Webcam Monitor

A local webcam person detector using Ultralytics YOLO11n and OpenVINO on an Intel NPU. Inference and image handling run on the laptop. The script opens a preview when at least two people are detected in the camera view and saves a snapshot when that threshold is first reached.

## Requirements

- Windows 10 or 11, 64-bit
- Python 3.12, 64-bit
- An Intel NPU with its Windows driver installed
- A webcam

OpenVINO's NPU plugin is supplied by the OpenVINO package, but the Intel NPU driver is installed separately. Confirm Windows sees the NPU before troubleshooting the Python environment.

## Setup

Run these commands from the project directory in Command Prompt:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check that OpenVINO detects the NPU:

```bat
python -c "import openvino as ov; print(ov.Core().available_devices)"
```

The output should include `NPU`. If it does not, install or update the Intel NPU driver and check Device Manager for Intel AI Boost.

## Model and Run

The model files are local runtime artifacts and are intentionally not tracked by Git. From the project directory, export the model if `yolo11n_openvino_model` is not present:

```bat
yolo export model=yolo11n.pt format=openvino imgsz=640
```

Ultralytics downloads `yolo11n.pt` if needed. Then start the monitor:

```bat
python npu.py
```

The script requests `device="intel:npu"` for inference. The preview stays open until the count remains below two for the configured close delay. Press Esc while the preview is open, or Ctrl+C in the terminal, to stop it. The webcam remains active while the script runs, including when the preview window is hidden.

This is a person-count trigger, not a directional or proximity detector: it opens the preview when two or more people are visible, but does not determine who is behind you or whether someone is approaching.

## Local Data and Licensing

Snapshots are written to `snapshots/` when the count first reaches two or more people. JPEG snapshots older than three hours are deleted at startup and checked about once per minute while the monitor runs. The folder, model weights, exported model, and virtual environment are ignored by Git; they remain on this computer and are not included in commits by default. Review snapshots before sharing them.

The exported model metadata identifies YOLO11n and declares AGPL-3.0. The installed Ultralytics package also declares AGPL-3.0. Review the applicable license terms before redistributing this project or using it in a deployment.