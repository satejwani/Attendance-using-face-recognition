from pymongo import MongoClient
from bson.binary import Binary
import numpy as np
import cv2
import face_recognition
from datetime import datetime
import base64
import os
import uuid
import csv

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['attendance_system']
students_collection = db['students']
attendance_collection = db['attendance']

# Directory for storing student images
STUDENT_IMAGES_DIR = 'student_images'

# Create the directory if it doesn't exist
os.makedirs(STUDENT_IMAGES_DIR, exist_ok=True)

# Path for attendance CSV file
ATTENDANCE_CSV_FILE = 'attendance.csv'

def get_db_collections():
    """Return the MongoDB collections for external use"""
    return students_collection, attendance_collection

def load_student_data():
    """Load images, names, and pre-computed encodings from MongoDB"""
    images = []
    roll_names = []
    encodings = []
    
    students = students_collection.find({})
    for student in students:
        if 'image_data' in student and 'face_encoding' in student:
            # Convert binary image data to numpy array
            img_data = student['image_data']
            img_array = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is not None:
                images.append(img)
                roll_names.append(f"{student['roll_no']}_{student['name']}")
                
                # Get pre-computed face encoding
                encoding = np.array(student['face_encoding'])
                encodings.append(encoding)
    
    return images, roll_names, encodings

def save_student_profile(roll_no, fullname, img_binary):
    """Save or update a student profile with image and face encoding"""
    try:
        # Convert to numpy array for face recognition
        img_array = np.frombuffer(img_binary, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # Convert to RGB for face_recognition library
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Find face encodings
        face_encodings = face_recognition.face_encodings(img_rgb)
        
        if not face_encodings:
            return {"success": False, "message": "No face detected in the image"}
        
        # Get the first face encoding
        face_encoding = face_encodings[0]
        
        # Generate a unique filename for local storage
        # Format: rollno_name_uuid.jpg
        sanitized_name = fullname.replace(" ", "_")
        unique_id = str(uuid.uuid4())[:8]  # Short UUID for filename
        filename = f"{roll_no}_{sanitized_name}_{unique_id}.jpg"
        file_path = os.path.join(STUDENT_IMAGES_DIR, filename)
        
        # Save image to file system
        with open(file_path, 'wb') as f:
            f.write(img_binary)
        
        # Save student profile to MongoDB with image and encoding
        student_profile = {
            "roll_no": roll_no,
            "name": fullname,
            "image_data": Binary(img_binary),  # Store image as binary data
            "face_encoding": face_encoding.tolist(),  # Store face encoding as list
            "registration_date": datetime.now(),
            "image_path": file_path  # Store the path to the local image
        }
        
        # Check if student already exists
        existing_student = students_collection.find_one({"roll_no": roll_no})
        if existing_student:
            # If updating, delete the old image file if it exists
            if "image_path" in existing_student and os.path.exists(existing_student["image_path"]):
                try:
                    os.remove(existing_student["image_path"])
                except:
                    pass  # Ignore errors during file removal
            
            students_collection.update_one(
                {"roll_no": roll_no},
                {"$set": student_profile}
            )
            return {"success": True, "message": "Profile updated successfully", "file_path": file_path}
        else:
            students_collection.insert_one(student_profile)
            return {"success": True, "message": "Profile saved successfully", "file_path": file_path}
            
    except Exception as e:
        return {"success": False, "message": str(e)}

def mark_student_attendance(roll_no, name):
    """Record a student's attendance in the database and CSV file"""
    try:
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        timestamp = now.strftime("%H:%M:%S")
        
        # Create attendance record for MongoDB
        attendance_record = {
            "roll_no": roll_no,
            "name": name,
            "date": date,
            "time": timestamp,
            "datetime": now  # Store actual datetime object for better querying
        }
        
        # Insert into MongoDB
        attendance_collection.insert_one(attendance_record)
        
        # Check if CSV file exists, create with headers if not
        file_exists = os.path.isfile(ATTENDANCE_CSV_FILE)
        
        # Write to CSV file
        with open(ATTENDANCE_CSV_FILE, mode='a', newline='') as file:
            fieldnames = ['roll_no', 'name', 'date', 'time']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            
            # Write header if file is being created
            if not file_exists:
                writer.writeheader()
            
            # Write attendance record
            writer.writerow({
                'roll_no': roll_no,
                'name': name,
                'date': date,
                'time': timestamp
            })
        
        return True
    except Exception as e:
        print(f"Error marking attendance: {str(e)}")
        return False

def get_all_attendance_records():
    """Retrieve all attendance records"""
    try:
        attendance_records = list(attendance_collection.find({}, {"_id": 0}))
        
        # Convert datetime objects to strings for JSON serialization
        for record in attendance_records:
            if "datetime" in record:
                record["datetime"] = record["datetime"].strftime("%Y-%m-%d %H:%M:%S")
                
        return {"success": True, "attendance": attendance_records}
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_all_students():
    """Retrieve all student profiles (excluding image data and encodings)"""
    try:
        students = list(students_collection.find({}, {"_id": 0, "image_data": 0, "face_encoding": 0}))
        
        # Convert datetime objects to strings for JSON serialization
        for student in students:
            if "registration_date" in student:
                student["registration_date"] = student["registration_date"].strftime("%Y-%m-%d %H:%M:%S")
                
        return {"success": True, "students": students}
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_student_image(roll_no):
    """Retrieve a student's image by roll number"""
    try:    
        student = students_collection.find_one({"roll_no": roll_no})
        if not student or 'image_data' not in student:
            return {"success": False, "error": "Student or image not found"}
        
        # Return image as base64 string
        image_base64 = base64.b64encode(student['image_data']).decode('utf-8')
        return {"success": True, "image": f"data:image/jpeg;base64,{image_base64}", "image_path": student.get("image_path", "")}
    except Exception as e:
        return {"success": False, "error": str(e)}