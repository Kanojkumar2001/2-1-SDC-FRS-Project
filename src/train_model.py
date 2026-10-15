# src/train_model.py
import cv2
import os
import numpy as np
from config import DATASET_DIR, TRAINER_FILE, TRAINER_DIR, FACE_SIZE


def train() -> str:
    """Train the LBPH face recognizer on the captured dataset.
    Normalizes every face sample with histogram equalization for robust matching.
    Returns a status message string."""
    if not os.path.exists(DATASET_DIR):
        return "❌ Dataset directory does not exist."

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces, ids = [], []
    for folder in os.listdir(DATASET_DIR):
        path = os.path.join(DATASET_DIR, folder)
        if not os.path.isdir(path):
            continue
        try:
            roll = int(folder.split("_")[0])
        except (ValueError, IndexError):
            continue

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            # Ensure consistent resolution and illumination across all samples
            face = cv2.resize(gray, FACE_SIZE)
            face = cv2.equalizeHist(face)
            faces.append(face)
            ids.append(roll)

    if not faces:
        return "❌ No face images found. Please capture faces first."

    os.makedirs(TRAINER_DIR, exist_ok=True)
    recognizer.train(faces, np.array(ids))
    recognizer.save(TRAINER_FILE)
    return f"✅ Model trained on {len(set(ids))} student(s), {len(faces)} image(s)."


if __name__ == "__main__":
    print(train())