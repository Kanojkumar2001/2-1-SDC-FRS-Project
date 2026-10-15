# src/recognize.py
import cv2
import numpy as np
import os
import json

from config import CASCADE_PATH, TRAINER_FILE, CONFIDENCE_THRESH, FACE_SIZE, STUDENTS_JSON, CAMERA_INDEX

# ── Module-level caches ─────────────────────────────────────────────────────
_detector = None
_cached_recognizer = None
_last_trainer_mtime = 0
_cached_students = {}
_last_students_mtime = 0


def _get_detector():
    """Return a cached CascadeClassifier (load once per process)."""
    global _detector
    if _detector is None or _detector.empty():
        _detector = cv2.CascadeClassifier(CASCADE_PATH)
        if _detector.empty():
            raise RuntimeError(f"Could not load cascade classifier from {CASCADE_PATH}")
    return _detector


def _get_students_cached() -> dict:
    """Return student mapping, automatically reloading when students.json changes."""
    global _cached_students, _last_students_mtime
    if not os.path.exists(STUDENTS_JSON):
        return {}
    try:
        mtime = os.path.getmtime(STUDENTS_JSON)
        if mtime != _last_students_mtime:
            with open(STUDENTS_JSON, "r", encoding="utf-8") as f:
                _cached_students = json.load(f)
            _last_students_mtime = mtime
    except Exception:
        pass
    return _cached_students


def load_recognizer(force: bool = False):
    """
    Load (or reload) the LBPH recognizer from disk.
    Automatically reloads if trainer.yml file has been updated.
    """
    global _cached_recognizer, _last_trainer_mtime
    if not os.path.exists(TRAINER_FILE):
        _cached_recognizer = None
        return None

    try:
        mtime = os.path.getmtime(TRAINER_FILE)
        if _cached_recognizer is None or force or mtime != _last_trainer_mtime:
            rec = cv2.face.LBPHFaceRecognizer_create()
            rec.read(TRAINER_FILE)
            _cached_recognizer = rec
            _last_trainer_mtime = mtime
    except Exception as e:
        print(f"[ERROR] Could not load recognizer: {e}")
        return None

    return _cached_recognizer


def get_cached_recognizer():
    """Return the cached recognizer or reload if needed."""
    return load_recognizer()


def recognize_frame(
    frame: np.ndarray,
    recognizer=None,
    conf_thresh: float = None,
) -> tuple[np.ndarray, list[dict]]:
    """
    Run LBPH face recognition on a single BGR frame.
    - Uses standard grayscale for face detection (equalized fallback for dim light).
    - Crops, resizes to FACE_SIZE, and equalizes the face ROI to match training.
    - Compares distance with conf_thresh (default from config).
    Returns (annotated_frame, list_of_results).
    """
    if recognizer is None:
        recognizer = get_cached_recognizer()

    if recognizer is None:
        return frame, []

    if conf_thresh is None:
        conf_thresh = CONFIDENCE_THRESH

    detector = _get_detector()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces on natural grayscale
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(50, 50),
    )

    # Fallback to equalized frame for dark/underexposed camera feeds
    if len(faces) == 0:
        gray_eq = cv2.equalizeHist(gray)
        faces = detector.detectMultiScale(
            gray_eq,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=(50, 50),
        )

    results = []
    students = _get_students_cached()

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        if face_roi.size == 0:
            continue

        # Normalization: resize and equalize matching the training pipeline
        face_roi = cv2.resize(face_roi, FACE_SIZE)
        face_roi = cv2.equalizeHist(face_roi)

        roll, conf = recognizer.predict(face_roi)
        roll_str = str(roll)

        if conf <= conf_thresh:
            info = students.get(roll_str)
            if info:
                name = info.get("name", f"Student {roll}")
                label = f"{name} | {roll} (conf:{conf:.0f})"
                color = (0, 200, 0)          # Green
                status = "known"
            else:
                label = f"ID: {roll} (conf:{conf:.0f})"
                color = (0, 165, 255)        # Orange (in model, not in DB)
                status = "no_record"
        else:
            label = f"Unknown (conf:{conf:.0f})"
            color = (0, 0, 220)              # Red
            status = "unknown"

        # Draw bounding rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Draw filled banner for label above face (or below if too close to top edge)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        if y - th - 12 > 0:
            banner_top = y - th - 10
            banner_bot = y
            text_pos = (x + 4, y - 4)
        else:
            banner_top = y + h
            banner_bot = y + h + th + 10
            text_pos = (x + 4, y + h + th + 4)

        cv2.rectangle(frame, (x, banner_top), (x + tw + 8, banner_bot), color, -1)
        cv2.putText(
            frame,
            label,
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),    # High-contrast white text
            2,
            cv2.LINE_AA,
        )

        results.append({
            "roll": roll,
            "conf": conf,
            "status": status,
            "label": label,
            "bbox": (x, y, w, h),
        })

    return frame, results


def recognize():
    """CLI recognition loop using OpenCV VideoCapture for main.py."""
    recognizer = load_recognizer()
    if recognizer is None:
        print("[ERROR] No trained model found at data/trainer/trainer.yml. Run Train Model first.")
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera at index {CAMERA_INDEX}")
        return

    print("[INFO] Starting Face Recognition. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, _ = recognize_frame(frame, recognizer)
        cv2.imshow("FRS - Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()