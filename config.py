import os

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR    = os.path.join(BASE_DIR, "data", "dataset")
TRAINER_DIR    = os.path.join(BASE_DIR, "data", "trainer")
TRAINER_FILE   = os.path.join(TRAINER_DIR, "trainer.yml")
STUDENTS_JSON  = os.path.join(BASE_DIR, "data", "students.json")
CASCADE_PATH   = os.path.join(BASE_DIR, "haarcascades",
                              "haarcascade_frontalface_default.xml")
OUTPUT_DIR     = os.path.join(BASE_DIR, "output")
LOG_FILE       = os.path.join(OUTPUT_DIR, "attendance_log.csv")

CAMERA_INDEX       = 0
CONFIDENCE_THRESH  = 110     # lower = stricter; 100-110 is optimal for webcam LBPH
FACE_SIZE          = (200, 200)
SAMPLES_PER_PERSON = 30