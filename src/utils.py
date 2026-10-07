# src/utils.py
import os, urllib.request, shutil
import cv2
from config import CASCADE_PATH

CASCADE_URL = ("https://raw.githubusercontent.com/opencv/opencv/4.x/"
               "data/haarcascades/haarcascade_frontalface_default.xml")


def _is_valid_xml(path):
    """True if the file exists, is non-trivial in size, and OpenCV can load it."""
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) < 1000:        # real XML is ~930 KB
        return False
    return not cv2.CascadeClassifier(path).empty()


def ensure_cascade():
    """Ensure a valid haarcascade XML exists at CASCADE_PATH."""
    if _is_valid_xml(CASCADE_PATH):
        return CASCADE_PATH

    os.makedirs(os.path.dirname(CASCADE_PATH), exist_ok=True)

    # 1) Try copying from the local OpenCV install (fastest, offline)
    try:
        local = os.path.join(cv2.data.haarcascades,
                             "haarcascade_frontalface_default.xml")
        if os.path.exists(local):
            shutil.copy(local, CASCADE_PATH)
            if _is_valid_xml(CASCADE_PATH):
                print("[INFO] Cascade copied from local OpenCV install.")
                return CASCADE_PATH
    except Exception as e:
        print("[WARN] Local copy failed:", e)

    # 2) Fall back to downloading
    print("[INFO] Downloading cascade XML ...")
    urllib.request.urlretrieve(CASCADE_URL, CASCADE_PATH)

    if not _is_valid_xml(CASCADE_PATH):
        raise RuntimeError(
            f"Failed to obtain a valid cascade XML at {CASCADE_PATH}. "
            "Delete the file and re-run."
        )
    print("[DONE] Cascade ready.")
    return CASCADE_PATH


def get_detector():
    """Return a fresh CascadeClassifier with a guaranteed-valid XML."""
    ensure_cascade()
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        raise RuntimeError(f"cv2 could not load {CASCADE_PATH}")
    return detector


def draw_label(frame, x, y, w, h, text, color=(0, 255, 0)):
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)