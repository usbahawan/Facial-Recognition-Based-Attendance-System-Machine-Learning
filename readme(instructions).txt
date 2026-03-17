FACIAL RECOGNITION BASED ATTENDANCE SYSTEM
(Machine Learning Lab Project)

Developed by: Muhammad Huzaifa and Usbah Saleem
Course: Machine Learning Lab
Supervisor: Sir Umar Nauman

1. INTRODUCTION

The Facial Recognition Attendance System is an AI-based project designed to automate the attendance process using facial recognition technology.
It replaces traditional manual or biometric attendance systems with a fast, contactless, and reliable solution that detects and recognizes faces using computer vision.

The system uses Python (Flask) for the backend server and a mobile application for image capturing. When a student or employee takes a picture through the app, it sends the image to the Flask server, which recognizes the face and marks attendance automatically in a CSV file or database.

2. OBJECTIVE

To automate attendance marking using facial recognition.

To eliminate proxy attendance and manual data entry.

To create a real-time and contactless attendance system.

To design a system that can be deployed locally and later expanded for institutional use.

3. PROBLEM STATEMENT

Manual attendance marking is time-consuming and prone to human error. Biometric systems, while faster, still require physical contact, which may be unhygienic or slow during peak hours.
Hence, this project aims to develop a non-contact, automated attendance system using facial recognition that can record attendance in real-time with minimal effort.

4. LITERATURE REVIEW 

Facial recognition has been widely used for security and identification systems. Libraries like OpenCV and face_recognition (based on Dlib) have made it easier to implement recognition without building deep models from scratch.
Research shows that deep-learning-based embeddings achieve high accuracy in face matching. This project leverages such pre-trained models for faster and accurate implementation.

5. SYSTEM METHODOLOGY
5.1 Dataset Preparation

A dataset of known users was created by capturing several face images under different lighting conditions and angles. Each image was stored in the dataset/ folder.
The script encode_faces.py processes these images and converts them into encodings (numerical representations) that are later used for face comparison.

5.2 Model and Libraries

face_recognition: For detecting and encoding faces.

OpenCV (cv2): For image capturing and preprocessing.

Flask: For backend API and server communication.

NumPy, Pandas: For data handling and file operations.

5.3 System Architecture

The system consists of two main parts:

Android App (Frontend): Captures the user’s image and sends it to the Flask API using HTTP requests.

Flask Server (Backend): Receives the image, compares it with the stored encodings, and marks attendance in the CSV file.

5.4 Workflow

The app captures an image and sends it to the Flask endpoint (/recognize) as a POST request.

Flask loads pre-encoded known faces.

The received image is processed, and the face is detected.

The system generates an encoding for the detected face and compares it with the known encodings.

If a match is found:

Attendance is marked in attendance.csv with the person’s name, date, and time.

The response “Attendance marked successfully” is sent to the app.

If no match is found:

Response “Unknown face” is sent.

6. PROJECT STRUCTURE
FacialRecognitionAttendance/
│
├── dataset/               → Folder containing registered user face images
├── static/                → (Optional) Static web files (CSS, JS, images)
├── templates/             → (Optional) HTML templates if web used
├── app.py                 → Main Flask application (handles API and recognition)
├── encode_faces.py        → Script to encode known faces
├── attendance.csv         → Output file where attendance is stored
├── requirements.txt       → Python dependencies
└── README.txt             → Project documentation file

7. HOW TO RUN THE PROJECT
Step 1: Install Dependencies

Open Command Prompt or Terminal in the project folder and run:

pip install -r requirements.txt


If you don’t have a requirements.txt file, install manually:

pip install flask
pip install opencv-python
pip install face_recognition
pip install numpy
pip install pandas

Step 2: Encode Known Faces

Before running the main server, encode your dataset images by running:

python encode_faces.py


This will create a file containing all encoded facial data.

Step 3: Start Flask Server

Run the Flask app:

python app.py


You will see something like:

* Running on http://192.168.100.xx:5000/


Note: Our local IP address (e.g., 192.168.100.21).

Step 4: Connect Mobile App

In your mobile app’s code, replace the server IP with your computer’s local IP.
Both your PC and phone must be on the same Wi-Fi network.

Example:

API_URL = "http://192.168.100.21:5000/recognize"

Step 5: Test Attendance

Open the mobile app, capture your face, and send it.

The Flask terminal will print the recognized person’s name.

The attendance.csv file will be updated with the name, date, and time.

8. FLOW OF EXECUTION

Start Flask server → Load encodings

App sends image → Flask receives it

Flask detects and encodes face

Compare encoding with known dataset

If matched → Save attendance

Send response to app

Repeat for next user

9. RESULTS AND EVALUATION

The system achieved over 90% recognition accuracy in good lighting conditions.
Attendance was marked successfully within 2 seconds per user.
All marked entries were automatically saved in the CSV file with timestamps.

10. CHALLENGES FACED

Local network connectivity issues between Flask and the Android app.

Recognition errors under low-light conditions.

Dataset preparation and proper image alignment for encoding.

11. FUTURE IMPROVEMENTS

Deploy Flask server on a cloud platform for global access.

Use a custom-trained CNN model for improved recognition.

Add live video stream recognition instead of static image capture.

Integrate admin dashboard for attendance analytics.

12. CONCLUSION

The project successfully demonstrates how machine learning can be integrated into real-world applications like attendance management.
It provides a fast, contactless, and reliable way to mark attendance with minimal human effort. With further improvements and deployment, this system can be scaled for institutional and corporate use.