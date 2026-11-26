import time, os
import cv2
from ultralytics import YOLO

MODEL_DIR   = "yolo11n_openvino_model"
CONF        = 0.5      # detection confidence
TARGET_FPS  = 8        # lower = less power
MIN_PEOPLE  = 2        # show window when count >= this
CLOSE_AFTER = 2        # seconds below MIN_PEOPLE before closing (avoids flicker; 0 = instant)
SNAP_DIR    = "snapshots"
RETENTION_SECONDS = 3 * 60 * 60
CLEANUP_INTERVAL = 60
WIN         = "NPU monitor"

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

os.makedirs(SNAP_DIR, exist_ok=True)
removed = cleanup_old_snapshots()
if removed:
    print(f"Removed {removed} snapshot(s) older than 3 hours")

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
        if time.monotonic() - last_cleanup >= CLEANUP_INTERVAL:
            removed = cleanup_old_snapshots()
            if removed:
                print(f"Removed {removed} snapshot(s) older than 3 hours")
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
                print(f"[{time.strftime('%H:%M:%S')}] {count} people -> {fn}")
        elif crowd and t0 - last_crowd_seen >= CLOSE_AFTER:
            crowd = False                               # back to 0 or 1 person
            print(f"[{time.strftime('%H:%M:%S')}] Back to {count} person(s)")

        if crowd:
            cv2.imshow(WIN, res.plot())
            window_open = True
            if cv2.waitKey(1) == 27:                    # Esc quits while window is visible
                break
        elif window_open:
            cv2.destroyWindow(WIN)
            cv2.waitKey(1)
            window_open = False

        time.sleep(max(0, interval - (time.time() - t0)))
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()