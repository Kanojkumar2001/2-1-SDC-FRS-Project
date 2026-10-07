import json, os
from config import STUDENTS_JSON

def load_students():
    if not os.path.exists(STUDENTS_JSON):
        return {}
    with open(STUDENTS_JSON, "r") as f:
        return json.load(f)

def save_students(data):
    with open(STUDENTS_JSON, "w") as f:
        json.dump(data, f, indent=4)

def add_student(roll, name, gender):
    data = load_students()
    data[str(roll)] = {"name": name, "gender": gender}
    save_students(data)

def get_student(roll):
    return load_students().get(str(roll))