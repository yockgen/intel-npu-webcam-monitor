# Windows NPU Webcam Monitor

Designed to monitor the space behind you while you face your screen. The webcam uses Ultralytics YOLO11n and OpenVINO on an Intel NPU to detect people in view; when it sees at least two people (you and someone else), it opens a preview and saves a snapshot. Detection and image handling run locally on your laptop. This is a person-count trigger, not directional tracking, so the camera must be positioned to see the area you want to monitor.

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

Ultralytics downloads `yolo11n.pt` if needed. Start the monitor with `run_monitor.bat`. It launches through the virtual environment's `pythonw.exe`, detached from the command prompt, so minimizing the terminal does not minimize the camera preview. Events and errors are recorded in `monitor.log`.

To stop the background monitor, run `stop_monitor.bat`; it exits after the next camera frame. Press Esc to stop while the preview is open. For console-based debugging, run `python npu.py` directly and use Ctrl+C to stop.

The script requests `device="intel:npu"` for inference. The preview stays open until the count remains below two for the configured close delay. It attempts to restore the preview if minimized while the alert is active and bring it forward when the alert first starts. Windows may restrict foreground activation. The webcam remains active while the script runs, including when the preview window is hidden.

This is a person-count trigger, not a directional or proximity detector: it opens the preview when two or more people are visible, but does not determine who is behind you or whether someone is approaching.

## Local Data and Licensing

Snapshots are written to `snapshots/` when the count first reaches two or more people. JPEG snapshots older than 15 minutes are deleted at startup and checked every 15 seconds while the monitor runs. The folder, model weights, exported model, and virtual environment are ignored by Git; they remain on this computer and are not included in commits by default. Review snapshots before sharing them.

The exported model metadata identifies YOLO11n and declares AGPL-3.0. The installed Ultralytics package also declares AGPL-3.0. Review the applicable license terms before redistributing this project or using it in a deployment.