import cv2, os
from config import DATASET_DIR, CASCADE_PATH, SAMPLES_PER_PERSON, CAMERA_INDEX
from src.database import add_student

def capture(roll, name, gender):
    save_dir = os.path.join(DATASET_DIR, f"{roll}_{name.replace(' ', '_')}")
    os.makedirs(save_dir, exist_ok=True)

    detector = cv2.CascadeClassifier(CASCADE_PATH)
    cam = cv2.VideoCapture(CAMERA_INDEX)
    count = 0

    print(f"[INFO] Capturing faces for {name} (Roll {roll}). Press 'q' to quit.")

    while True:
        ret, frame = cam.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            cv2.imwrite(f"{save_dir}/img_{count:03d}.jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Capture Faces - Press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= SAMPLES_PER_PERSON:
            break

    cam.release()
    cv2.destroyAllWindows()
    add_student(roll, name, gender)
    print(f"[DONE] Saved {count} samples for {name}.")