# 🎓 Face Recognition Attendance System

A real-time face recognition–based attendance system built using Flask, OpenCV, face_recognition (dlib), and MongoDB.
The system allows student registration via webcam image capture and automatically marks attendance using a live camera feed.

## 🚀 Features
- Student registration with face image capture
- Face encoding & recognition
- Live camera attendance window
- MongoDB storage for students & attendance
- Automatic CSV attendance export
- REST APIs for students & attendance data

## 🛠 Tech Stack
- Backend: Flask
- Face Recognition: OpenCV, dlib, face_recognition
- Database: MongoDB
- Frontend: HTML, CSS, JavaScript
- Language: Python

## 📂 Project Structure
'''
attendance-system/
  ├── attendance_sysytem.py
  ├── db_operations.py
  ├── requirements.txt
  ├── index.html
  ├── registration.html
  ├── student_images/
  ├── attendance.csv
'''

## ⚙️ Installation
pip install -r requirements.txt

## ▶️ Run
python attendance_system.py

Open http://127.0.0.1:5000/

## 📜 License
MIT License

## 👨‍💻 Author
Satej
