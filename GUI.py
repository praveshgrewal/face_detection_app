import tkinter as tk
from PIL import Image, ImageTk
import cv2
from tkinter import messagebox
import os
import numpy as np
import mysql.connector
import threading

# Function to resize the background image when the window is resized
def resize_bg_image(event):
    new_width = event.width
    new_height = event.height
    resized_image = original_bg_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    updated_bg_image = ImageTk.PhotoImage(resized_image)
    bg_label.config(image=updated_bg_image)
    bg_label.image = updated_bg_image  # Keep a reference to prevent garbage collection

def generate_dataset():
    if t1.get() == "" or t2.get() == "" or t3.get() == "":
        messagebox.showinfo("Result", "Please enter all fields")
    else:
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="",
            database="Authorized_users"
        )

        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM users")
        myresult = mycursor.fetchall()
        id = len(myresult) + 1
    
        sql = "INSERT INTO users (id, Name, Age, Department) VALUES (%s, %s, %s, %s)"
        val = (id, t1.get(), t2.get(), t3.get())
        mycursor.execute(sql, val)
        mydb.commit()

        face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        def face_cropped(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 0:
                return None
            for (x, y, w, h) in faces:
                return img[y:y+h, x:x+w]  # Return the cropped face

        cap = cv2.VideoCapture(0)
        img_id = 0

        while True:
            ret, frame = cap.read()
            if face_cropped(frame) is not None:
                img_id += 1
                face = cv2.resize(face_cropped(frame), (200, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                file_name_path = f"data/user.{id}.{img_id}.jpg"
                cv2.imwrite(file_name_path, face)
                cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Cropped face", face)
            if cv2.waitKey(1) == 13 or img_id == 200:  # Break on Enter key or after 200 images
                break

        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Result", "Generating datasets completed!")

def train_classifier():
    data_dir = "data"
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L')  # Convert to grayscale
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])

        faces.append(imageNp)
        ids.append(id)

    ids = np.array(ids)

    clf = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    clf.train(faces, ids)
    clf.write("classifier.xml")
    messagebox.showinfo("Result", "Training datasets completed!")

def detect_face():
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)

        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            id, pred = clf.predict(gray_image[y:y + h, x:x + w])
            confidence = int(100 * (1 - pred / 300))

            mydb = mysql.connector.connect(
                host="localhost",
                user="root",
                passwd="",
                database="Authorized_users"
            )
            mycursor = mydb.cursor()
            mycursor.execute("SELECT name FROM users WHERE id = %s", (id,))
            s = mycursor.fetchone()

            if s:
                s = s[0]
            else:
                s = "Unknown"

            mycursor.close()
            mydb.close()

            if confidence > 74:
                cv2.putText(img, s, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            else:
                cv2.putText(img, "UNKNOWN", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

    def recognize(img, clf, faceCascade):
        draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 255), "Face", clf)
        return img

    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    def video_loop():
        video_capture = cv2.VideoCapture(0)
        while True:
            ret, img = video_capture.read()
            img = recognize(img, clf, faceCascade)
            cv2.imshow("face detection", img)
            if cv2.waitKey(1) == 13:
                break
        video_capture.release()
        cv2.destroyAllWindows()

    # Run the video loop in a separate thread to prevent GUI freezing
    threading.Thread(target=video_loop).start()

# Tkinter GUI
window = tk.Tk()
window.title("Face Recognition System")
window.geometry("800x500")

original_bg_image = Image.open("untitled design.png")
bg_image = ImageTk.PhotoImage(original_bg_image)
bg_label = tk.Label(window, image=bg_image)
bg_label.grid(row=0, column=0, columnspan=3, rowspan=5, sticky="nsew")

window.grid_rowconfigure(0, weight=1)
window.grid_rowconfigure(1, weight=1)
window.grid_rowconfigure(2, weight=1)
window.grid_rowconfigure(3, weight=1)
window.grid_rowconfigure(4, weight=1)
window.grid_columnconfigure(0, weight=1)
window.grid_columnconfigure(1, weight=1)
window.grid_columnconfigure(2, weight=1)

window.bind("<Configure>", resize_bg_image)

header = tk.Label(window, text="Face Recognition System", font=("Helvetica", 24, "bold"), bg="#4A90E2", fg="white")
header.grid(row=0, column=0, columnspan=3, padx=10, pady=20, sticky="ew")

label_font = ("Helvetica", 16)

l1 = tk.Label(window, text="Name:", font=label_font, bg="#f0f0f0")
l1.grid(column=0, row=1, padx=10, pady=10, sticky="e")
t1 = tk.Entry(window, font=("Helvetica", 16), width=30, bd=3)
t1.grid(column=1, row=1, padx=10, pady=10, sticky="w")

l2 = tk.Label(window, text="Age:", font=label_font, bg="#f0f0f0")
l2.grid(column=0, row=2, padx=10, pady=10, sticky="e")
t2 = tk.Entry(window, font=("Helvetica", 16), width=30, bd=3)
t2.grid(column=1, row=2, padx=10, pady=10, sticky="w")

l3 = tk.Label(window, text="Department:", font=label_font, bg="#f0f0f0")
l3.grid(column=0, row=3, padx=10, pady=10, sticky="e")
t3 = tk.Entry(window, font=("Helvetica", 16), width=30, bd=3)
t3.grid(column=1, row=3, padx=10, pady=10, sticky="w")

button_font = ("Helvetica", 16, "bold")
button_padx = 10
button_pady = 20

b3 = tk.Button(window, text="1. Generate Dataset", font=button_font, bg="#FF69B4", fg="white", command=generate_dataset)
b3.grid(column=0, row=4, padx=button_padx, pady=button_pady)

b1 = tk.Button(window, text="2. Train Model", font=button_font, bg="#FFA500", fg="white", command=train_classifier)
b1.grid(column=1, row=4, padx=button_padx, pady=button_pady)

b2 = tk.Button(window, text="3. Detect Faces", font=button_font, bg="#32CD32", fg="white", command=detect_face)
b2.grid(column=2, row=4, padx=button_padx, pady=button_pady)

window.mainloop()
