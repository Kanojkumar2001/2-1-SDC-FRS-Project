import cv2, os, numpy as np
from config import DATASET_DIR, TRAINER_FILE, TRAINER_DIR, FACE_SIZE

def train():
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces, ids = [], []
    for folder in os.listdir(DATASET_DIR):
        path = os.path.join(DATASET_DIR, folder)
        if not os.path.isdir(path): continue
        roll = int(folder.split("_")[0])

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            faces.append(cv2.resize(gray, FACE_SIZE))
            ids.append(roll)

    if not faces:
        print("[ERROR] No faces found. Capture data first.")
        return

    os.makedirs(TRAINER_DIR, exist_ok=True)
    recognizer.train(faces, np.array(ids))
    recognizer.save(TRAINER_FILE)
    print(f"[DONE] Trained on {len(set(ids))} students, {len(faces)} images.")