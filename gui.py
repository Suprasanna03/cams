import tkinter as tk
from tkinter import messagebox, PhotoImage, Label
import cv2
import face_recognition
import numpy as np
import csv
from datetime import datetime
class AttendanceSystemGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")

        self.students = ["Ambhani", "Ratantata", "Kalam", "Elonmusk", "Suprasanna", "Devika", "Srihitha"]

        self.video_capture = cv2.VideoCapture(0)

        self.known_face_encoding, self.known_faces_names = self.load_known_faces()

        self.face_names = []
        self.initialize_gui()


    def initialize_gui(self):
        self.root.geometry("600x400")

        self.frame = tk.Frame(self.root)
        self.frame.pack(pady=20)

        self.label = tk.Label(self.frame, text="Attendance System", font=("Calibri", 20))
        self.label.grid(row=0, column=0, columnspan=2, pady=10)

        self.start_button = tk.Button(self.frame, text="Mark Attendance", command=self.start_attendance, width=15)
        self.start_button.grid(row=1, column=0, pady=10, padx=10)

        self.stop_button = tk.Button(self.frame, text="Exit Application", command=self.stop_attendance, width=15)
        self.stop_button.grid(row=1, column=1, pady=10, padx=10)

        self.face_label = tk.Label(self.frame, text="", font=("Helvetica", 14))
        self.face_label.grid(row=2, column=0, columnspan=2, pady=10)

    def load_known_faces(self):
        jobs_image = face_recognition.load_image_file("./photos/ambhani.jpeg")
        jobs_encoding = face_recognition.face_encodings(jobs_image)[0]

        ratan_tata_image = face_recognition.load_image_file("./photos/ratantata.jpeg")
        ratan_tata_encoding = face_recognition.face_encodings(ratan_tata_image)[0]

        sadmona_image = face_recognition.load_image_file("./photos/kalam.jpeg")
        sadmona_encoding = face_recognition.face_encodings(sadmona_image)[0]

        tesla_image = face_recognition.load_image_file("./photos/elonmusk.jpeg")
        tesla_encoding = face_recognition.face_encodings(tesla_image)[0]

        suprasanna = face_recognition.load_image_file("./photos/Suprasanna.jpeg")
        suprasanna_encoding = face_recognition.face_encodings(suprasanna)[0]

        devika_image = face_recognition.load_image_file("./photos/devika.jpeg")
        devika_encoding = face_recognition.face_encodings(devika_image)[0]

        srihitha_image = face_recognition.load_image_file("./photos/srihitha.jpeg")
        srihitha_encoding = face_recognition.face_encodings(srihitha_image)[0]

        known_face_encoding = [jobs_encoding, ratan_tata_encoding, sadmona_encoding, tesla_encoding, suprasanna_encoding, devika_encoding, srihitha_encoding]
        known_faces_names = ["ambhani", "ratantata", "kalam", "elonmusk", "suprasanna", "devika", "srihitha"]

        return known_face_encoding, known_faces_names

    def start_attendance(self):
        self.capture_attendance()

    def stop_attendance(self):
        self.video_capture.release()
        cv2.destroyAllWindows()
        self.root.destroy()

    def capture_attendance(self):
        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        filename = current_date + '.csv'

        with open(filename, 'w+', newline='') as f:
            lnwriter = csv.writer(f)

            while True:
                _, frame = self.video_capture.read()
                if frame is None:
                    break

                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]

                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                face_names = []

                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encoding, face_encoding)
                    name = ""
                    face_distance = face_recognition.face_distance(self.known_face_encoding, face_encoding)
                    best_match_index = np.argmin(face_distance)
                    if matches[best_match_index]:
                        name = self.known_faces_names[best_match_index]

                    face_names.append(name)
                    if name in self.known_faces_names:
                        # Convert name to lowercase for consistent comparison
                        name_lower = name.lower()

                        # Drawing text on the frame
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        bottomLeftCornerOfText = (8, 100)
                        fontScale = 1.5
                        fontColor = (100, 0, 0)
                        thickness = 3
                        lineType = 3

                        cv2.putText(frame, name + ' Present',
                                    bottomLeftCornerOfText,
                                    font,
                                    fontScale,
                                    fontColor,
                                    thickness,
                                    lineType)

                        # Update the GUI label
                        display_text = f"Recognized: {', '.join(face_names)}"
                        self.face_label.config(text=display_text)

                        # Perform actions if the recognized person is a student
                        if name_lower in self.students:
                            self.students.remove(name_lower)
                            print(self.students)
                            current_time = now.strftime("%H-%M-%S")
                            lnwriter.writerow([name_lower, current_time])

                cv2.imshow("Attendance System", frame)

                # Check for key events and break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.video_capture.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Attendance", "Attendance captured successfully.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceSystemGUI(root)
    root.mainloop()
