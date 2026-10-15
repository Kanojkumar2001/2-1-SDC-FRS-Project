# src/capture_faces.py
import cv2
import os
import numpy as np
from config import DATASET_DIR, CASCADE_PATH, SAMPLES_PER_PERSON, FACE_SIZE, CAMERA_INDEX
from src.database import add_student
from src.utils import get_detector


def capture_from_frame(frame: np.ndarray, roll: str, name: str, gender: str,
                        count: int) -> tuple[np.ndarray, int]:
    """
    Process a single BGR frame for face sample capture:
    - Detects faces on clean grayscale (with equalized fallback for dim lighting).
    - Crops, resizes to FACE_SIZE, and equalizes the face sample.
    - Saves sample to dataset directory.
    - Draws bounding box and sample count counter on the frame.
    Returns (annotated_frame, new_count).
    """
    save_dir = os.path.join(DATASET_DIR, f"{roll}_{name.replace(' ', '_')}")
    os.makedirs(save_dir, exist_ok=True)

    detector = get_detector()
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces on natural grayscale first
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(50, 50),
    )

    # Fallback to equalized grayscale if no faces found (helps in dark rooms)
    if len(faces) == 0:
        gray_eq = cv2.equalizeHist(gray)
        faces = detector.detectMultiScale(
            gray_eq,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=(50, 50),
        )

    for (x, y, w, h) in faces:
        if count >= SAMPLES_PER_PERSON:
            break
        count += 1

        face_roi = gray[y:y + h, x:x + w]
        if face_roi.size == 0:
            continue
        crop = cv2.resize(face_roi, FACE_SIZE)
        crop = cv2.equalizeHist(crop)
        cv2.imwrite(os.path.join(save_dir, f"img_{count:03d}.jpg"), crop)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Samples: {count}/{SAMPLES_PER_PERSON}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )

    return frame, count


def finalize_capture(roll: str, name: str, gender: str, count: int) -> str:
    """Register student details in JSON database after samples are captured."""
    add_student(roll, name, gender)
    return f"✅ Saved {count} samples for {name} (Roll: {roll})."


def capture(roll: str, name: str, gender: str):
    """CLI capture method using OpenCV VideoCapture for main.py compatibility."""
    save_dir = os.path.join(DATASET_DIR, f"{roll}_{name.replace(' ', '_')}")
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot access camera at index {CAMERA_INDEX}")
        return

    count = 0
    print(f"[INFO] Capturing {SAMPLES_PER_PERSON} samples for {name}. Press 'q' to cancel.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, count = capture_from_frame(frame, roll, name, gender, count)
        cv2.imshow("Capturing Faces - FRS", frame)

        if count >= SAMPLES_PER_PERSON or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    finalize_capture(roll, name, gender, count)
    print(f"[SUCCESS] Registered {name} with {count} face samples.")