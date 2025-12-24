from flask import Flask, jsonify, request, render_template, send_from_directory, Response
from flask_cors import CORS
import cv2
import numpy as np
import face_recognition
import base64
from datetime import datetime
import db_operations
import os

app = Flask(__name__, static_url_path='')
CORS(app)

# Directory for storing student images
STUDENT_IMAGES_DIR = 'student_images'

# Create the directory if it doesn't exist
os.makedirs(STUDENT_IMAGES_DIR, exist_ok=True)

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/registration.html')
def registration():
    return send_from_directory('.', 'registration.html')

@app.route('/student_images/<path:filename>')
def serve_student_image(filename):
    return send_from_directory(STUDENT_IMAGES_DIR, filename)

@app.route('/attendance_window')
def attendance_window():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Attendance Window</title>
        <style>
            body { margin: 0; padding: 0; overflow: hidden; }
            #videoContainer { position: relative; width: 100vw; height: 100vh; }
            #videoFeed { width: 100%; height: 100%; object-fit: cover; }
            #closeBtn { 
                position: absolute; 
                top: 20px; 
                right: 20px; 
                padding: 10px 20px;
                background: #dc3545;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                z-index: 100;
                font-size: 16px;
                font-weight: bold;
            }
            #status {
                position: absolute;
                bottom: 20px;
                left: 20px;
                background: rgba(0,0,0,0.7);
                color: white;
                padding: 10px;
                border-radius: 5px;
                z-index: 100;
            }
        </style>
    </head>
    <body>
        <div id="videoContainer">
            <button id="closeBtn" onclick="window.close()">Close Window</button>
            <img id="videoFeed" src="/video_feed">
            <div id="status">Press 'q' or click Close Window to stop attendance</div>
        </div>
        <script>
            document.addEventListener('keydown', function(e) {
                if (e.key === 'q' || e.key === 'Q') {
                    window.close();
                }
            });
            
            // Handle window closing cleanly
            window.addEventListener('beforeunload', function() {
                fetch('/stop_attendance');
            });
        </script>
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    def generate():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            yield "Camera could not be opened"
            return
        
        images, roll_names, encodings_known = db_operations.load_student_data()
        recognized_students = set()
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            frame_small = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
            frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            
            faces_cur_frame = face_recognition.face_locations(frame_rgb)
            encodings_cur_frame = face_recognition.face_encodings(frame_rgb, faces_cur_frame)
            
            for (top, right, bottom, left), encoding in zip(faces_cur_frame, encodings_cur_frame):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                matches = face_recognition.compare_faces(encodings_known, encoding, tolerance=0.5)
                face_distances = face_recognition.face_distance(encodings_known, encoding)
                best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None
                
                if best_match_index is not None and matches[best_match_index]:
                    roll_name = roll_names[best_match_index]
                    roll_no, name = roll_name.split("_")
                    student_key = f"{roll_no}_{name}"
                    
                    if student_key not in recognized_students:
                        recognized_students.add(student_key)
                        db_operations.mark_student_attendance(roll_no, name)
                    
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, f"{name} ({roll_no})", (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
            
        cap.release()
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_attendance')
def stop_attendance():
    # Placeholder for any cleanup needed
    return jsonify({"status": "success"})

@app.route('/saveProfile', methods=['POST'])
def save_profile():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        roll_no = data['id']
        fullname = data['fullname']
        
        img_binary = base64.b64decode(image_data)
        result = db_operations.save_student_profile(roll_no, fullname, img_binary)
        
        if result["success"]:
            return jsonify({"message": result["message"], "file_path": result.get("file_path", "")})
        else:
            return jsonify({"error": result["message"]}), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/get_attendance', methods=['GET'])
def get_attendance():
    result = db_operations.get_all_attendance_records()
    if result["success"]:
        return jsonify(result)
    else:
        return jsonify({"error": result["error"]}), 500

@app.route('/get_students', methods=['GET'])
def get_students():
    result = db_operations.get_all_students()
    if result["success"]:
        return jsonify(result)
    else:
        return jsonify({"error": result["error"]}), 500

@app.route('/get_student_image/<roll_no>', methods=['GET'])
def get_student_image(roll_no):
    result = db_operations.get_student_image(roll_no)
    if result["success"]:
        return jsonify(result)
    else:
        return jsonify({"error": result["error"]}), 404

if __name__ == '__main__':
    app.run(debug=True)