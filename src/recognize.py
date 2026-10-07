import cv2
from config import CASCADE_PATH, TRAINER_FILE, CONFIDENCE_THRESH, CAMERA_INDEX, FACE_SIZE
from src.database import get_student

def recognize():
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_FILE)

    cam = cv2.VideoCapture(CAMERA_INDEX)
    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = cam.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            face = cv2.resize(gray[y:y+h, x:x+w], FACE_SIZE)
            roll, conf = recognizer.predict(face)

            if conf < CONFIDENCE_THRESH:
                info = get_student(roll)
                if info:
                    label = f"{info['name']} | Roll:{roll} | {info['gender']}"
                    color = (0, 255, 0)
                else:
                    label = f"Unknown ID:{roll}"
                    color = (0, 165, 255)
            else:
                label = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("FRS - Face Recognition System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()