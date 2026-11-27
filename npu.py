import ctypes
import logging
import time, os
import sys
from ctypes import wintypes
import cv2
from ultralytics import YOLO

MODEL_DIR   = "yolo11n_openvino_model"
CONF        = 0.5      # detection confidence
TARGET_FPS  = 8        # lower = less power
MIN_PEOPLE  = 2        # show window when count >= this
CLOSE_AFTER = 2        # seconds below MIN_PEOPLE before closing (avoids flicker; 0 = instant)
SNAP_DIR    = "snapshots"
RETENTION_SECONDS = 15 * 60
CLEANUP_INTERVAL = 15
WIN         = "NPU monitor"
SW_RESTORE  = 9
LOG_FILE    = "monitor.log"
STOP_FILE   = ".stop_monitor"

user32 = ctypes.WinDLL("user32", use_last_error=True)
user32.FindWindowW.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR)
user32.FindWindowW.restype = wintypes.HWND
user32.IsIconic.argtypes = (wintypes.HWND,)
user32.IsIconic.restype = wintypes.BOOL
user32.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)
user32.ShowWindow.restype = wintypes.BOOL
user32.SetForegroundWindow.argtypes = (wintypes.HWND,)
user32.SetForegroundWindow.restype = wintypes.BOOL

log_handlers = [logging.FileHandler(LOG_FILE, encoding="utf-8")]
if sys.stdout is not None:
    log_handlers.append(logging.StreamHandler(sys.stdout))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=log_handlers,
)

def log_uncaught_exception(exc_type, exc_value, exc_traceback):
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = log_uncaught_exception

def cleanup_old_snapshots():
    cutoff = time.time() - RETENTION_SECONDS
    removed = 0
    for entry in os.scandir(SNAP_DIR):
        if not entry.name.lower().endswith(".jpg"):
            continue
        try:
            if entry.is_file() and entry.stat().st_mtime < cutoff:
                os.remove(entry.path)
                removed += 1
        except FileNotFoundError:
            pass
    return removed

def restore_camera_window(activate=False):
    hwnd = user32.FindWindowW(None, WIN)
    if not hwnd:
        return

    minimized = user32.IsIconic(hwnd)
    if minimized:
        user32.ShowWindow(hwnd, SW_RESTORE)
    if activate or minimized:
        user32.SetForegroundWindow(hwnd)

os.makedirs(SNAP_DIR, exist_ok=True)
removed = cleanup_old_snapshots()
if removed:
    logging.info("Removed %s snapshot(s) older than 15 minutes", removed)

model = YOLO(MODEL_DIR, task="detect")
cap = cv2.VideoCapture(0)

window_open = False
crowd = False                 # True while >= MIN_PEOPLE are present
last_crowd_seen = 0.0
interval = 1.0 / TARGET_FPS
last_cleanup = time.monotonic()

try:
    while True:
        t0 = time.time()
        if os.path.exists(STOP_FILE):
            try:
                os.remove(STOP_FILE)
            except FileNotFoundError:
                pass
            logging.info("Stop requested")
            break

        if time.monotonic() - last_cleanup >= CLEANUP_INTERVAL:
            removed = cleanup_old_snapshots()
            if removed:
                logging.info("Removed %s snapshot(s) older than 15 minutes", removed)
            last_cleanup = time.monotonic()

        ok, frame = cap.read()
        if not ok:
            break

        res = model.predict(frame, imgsz=640, device="intel:npu",
                            classes=[0], conf=CONF, verbose=False)[0]
        count = len(res.boxes)

        if count >= MIN_PEOPLE:
            last_crowd_seen = t0
            if not crowd:                               # just reached 2+ people
                crowd = True
                fn = os.path.join(SNAP_DIR, time.strftime("%Y%m%d_%H%M%S") + ".jpg")
                cv2.imwrite(fn, res.plot())
                logging.info("%s people detected -> %s", count, fn)
        elif crowd and t0 - last_crowd_seen >= CLOSE_AFTER:
            crowd = False                               # back to 0 or 1 person
            logging.info("Back to %s person(s)", count)

        if crowd:
            cv2.imshow(WIN, res.plot())
            restore_camera_window(activate=not window_open)
            window_open = True
            if cv2.waitKey(1) == 27:                    # Esc quits while window is visible
                break
        elif window_open:
            cv2.destroyWindow(WIN)
            cv2.waitKey(1)
            window_open = False

        time.sleep(max(0, interval - (time.time() - t0)))
except KeyboardInterrupt:
    logging.info("Interrupted")
finally:
    cap.release()
    cv2.destroyAllWindows()
    logging.info("Monitor stopped")