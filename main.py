import sys
from src.capture_faces import capture
from src.train_model import train
from src.recognize import recognize

def menu():
    print("\n===== FRS - Face Recognition System =====")
    print("1. Register New Person (Capture Faces)")
    print("2. Train Model")
    print("3. Start Recognition")
    print("4. Exit")
    return input("Enter choice: ").strip()

def main():
    while True:
        choice = menu()

        if choice == "1":
            roll   = input("Roll Number : ").strip()
            name   = input("Full Name   : ").strip()
            gender = input("Gender (Male/Female): ").strip().capitalize()
            if gender not in ("Male", "Female"):
                print("[ERROR] Gender must be Male or Female.")
                continue
            capture(roll, name, gender)

        elif choice == "2":
            train()

        elif choice == "3":
            recognize()

        elif choice == "4":
            print("Goodbye!")
            sys.exit(0)

        else:
            print("[ERROR] Invalid choice.")

if __name__ == "__main__":
    main()