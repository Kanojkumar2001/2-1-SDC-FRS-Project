import json
import os
import shutil
from config import STUDENTS_JSON, DATASET_DIR


def load_students() -> dict:
    """Load the student records dictionary from JSON file."""
    if not os.path.exists(STUDENTS_JSON):
        return {}
    try:
        with open(STUDENTS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_students(data: dict):
    """Save the student records dictionary to JSON file."""
    os.makedirs(os.path.dirname(STUDENTS_JSON), exist_ok=True)
    with open(STUDENTS_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def add_student(roll, name, gender):
    """Add or update a student record."""
    data = load_students()
    data[str(roll)] = {"name": name.strip(), "gender": gender.strip()}
    save_students(data)


def get_student(roll):
    """Retrieve info dictionary for a given roll number."""
    return load_students().get(str(roll))


def delete_student(roll, delete_samples: bool = True) -> bool:
    """Delete a student from JSON and optionally remove their captured face samples."""
    roll_str = str(roll)
    data = load_students()
    if roll_str not in data:
        return False

    del data[roll_str]
    save_students(data)

    if delete_samples and os.path.exists(DATASET_DIR):
        for folder in os.listdir(DATASET_DIR):
            if folder.startswith(f"{roll_str}_") or folder == roll_str:
                folder_path = os.path.join(DATASET_DIR, folder)
                if os.path.isdir(folder_path):
                    shutil.rmtree(folder_path, ignore_errors=True)

    return True