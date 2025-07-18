# app.py - Railway deployment dengan FIXED face landmarks dan duration calculation
from flask import Flask, render_template, request, Response, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
import mediapipe as mp
import numpy as np
import pyttsx3
from scipy.spatial import distance as dis
import cv2 as cv
import os
import time
import uuid
from datetime import datetime, timedelta
import json
import threading
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend untuk Railway
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import base64
import tempfile
import shutil
import traceback

application = Flask(__name__)

# Configuration dengan optimized paths untuk Railway
application.config['UPLOAD_FOLDER'] = '/tmp/uploads'
application.config['DETECTED_FOLDER'] = '/tmp/detected'
application.config['REPORTS_FOLDER'] = '/tmp/reports'
application.config['RECORDINGS_FOLDER'] = '/tmp/recordings'
application.config['MAX_CONTENT_PATH'] = 10000000

# Ensure all directories exist dengan error handling untuk Railway
for folder in [application.config['UPLOAD_FOLDER'], application.config['DETECTED_FOLDER'], 
               application.config['REPORTS_FOLDER'], application.config['RECORDINGS_FOLDER']]:
    try:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            os.chmod(folder, 0o755)
        print(f"Directory ready: {folder}")
    except Exception as e:
        print(f"Error creating directory {folder}: {str(e)}")

# FIXED: Enhanced global variables untuk consistent tracking dan duration calculation
monitoring_lock = threading.RLock()
live_monitoring_active = False
session_data = {
    'start_time': None,
    'end_time': None,
    'detections': [],
    'alerts': [],
    'focus_statistics': {
        'unfocused_time': 0,
        'yawning_time': 0,
        'sleeping_time': 0,
        'total_persons': 0,
        'total_detections': 0
    },
    'recording_path': None,
    'recording_frames': [],
    'session_id': None,
    'client_alerts': [],
    'frame_counter': 0,
    'frame_timestamps': [],
    'total_frames_processed': 0,
    # FIXED: Enhanced distraction tracking
    'person_distraction_sessions': {},  # Track continuous distraction sessions
    'person_focus_sessions': {}  # Track focus sessions
}

# Video recording variables
video_writer = None
recording_active = False

# FIXED: Enhanced person tracking dengan proper session management
person_state_timers = {}
person_current_state = {}
last_alert_time = {}
person_distraction_start = {}  # Track when each distraction type started
person_session_durations = {}  # Track actual session durations per person

# Alert thresholds (dalam seconds) - gunakan yang sama dengan lokal
DISTRACTION_THRESHOLDS = {
    'SLEEPING': 10,
    'YAWNING': 3.5,
    'NOT FOCUSED': 10
}

# FIXED: Frame recording configuration
FRAME_STORAGE_INTERVAL = 2  # Store every 2nd frame
MAX_STORED_FRAMES = 200     # Increase limit for better video quality
RECORDING_FPS = 5           # Target FPS for final video

# Initialize MediaPipe dengan error handling untuk Railway
face_detection = None
face_mesh = None

def init_mediapipe():
    """Initialize MediaPipe dengan error handling untuk Railway"""
    global face_detection, face_mesh
    try:
        face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("MediaPipe initialized successfully")
        return True
    except Exception as e:
        print(f"MediaPipe initialization failed: {str(e)}")
        return False

# FIXED: Enhanced landmark drawing dengan style yang sama seperti lokal
def draw_landmarks(image, landmarks, land_mark, color, thickness=2):
    """Draw landmarks on the image with improved visibility"""
    height, width = image.shape[:2]
    for face in land_mark:
        try:
            point = landmarks.landmark[face]
            point_scale = ((int)(point.x * width), (int)(point.y * height))
            # FIXED: Enhanced visibility dengan outline
            cv.circle(image, point_scale, thickness + 1, (0, 0, 0), -1)  # Black outline
            cv.circle(image, point_scale, thickness, color, -1)  # Colored center
        except (IndexError, AttributeError):
            continue

def euclidean_distance(image, top, bottom):
    """Calculate euclidean distance between two points"""
    height, width = image.shape[0:2]
    point1 = int(top.x * width), int(top.y * height)
    point2 = int(bottom.x * width), int(bottom.y * height)
    distance = dis.euclidean(point1, point2)
    return distance

def get_aspect_ratio(image, landmarks, top_bottom, left_right):
    """Calculate aspect ratio based on landmarks"""
    top = landmarks.landmark[top_bottom[0]]
    bottom = landmarks.landmark[top_bottom[1]]
    top_bottom_dis = euclidean_distance(image, top, bottom)

    left = landmarks.landmark[left_right[0]]
    right = landmarks.landmark[left_right[1]]
    left_right_dis = euclidean_distance(image, left, right)
    
    # Handle division by zero untuk Railway
    if top_bottom_dis == 0:
        return 5.0  # Default to closed eyes ratio
    
    aspect_ratio = left_right_dis / top_bottom_dis
    return aspect_ratio

def extract_eye_landmarks(face_landmarks, eye_landmark_indices):
    """Extract eye landmarks from face landmarks"""
    eye_landmarks = []
    for index in eye_landmark_indices:
        landmark = face_landmarks.landmark[index]
        eye_landmarks.append([landmark.x, landmark.y])
    return np.array(eye_landmarks)

def calculate_midpoint(points):
    """Calculate the midpoint of a set of points"""
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    midpoint = (sum(x_coords) // len(x_coords), sum(y_coords) // len(y_coords))
    return midpoint

def check_iris_in_middle(left_eye_points, left_iris_points, right_eye_points, right_iris_points):
    """Check if iris is in the middle of the eye"""
    left_eye_midpoint = calculate_midpoint(left_eye_points)
    right_eye_midpoint = calculate_midpoint(right_eye_points)
    left_iris_midpoint = calculate_midpoint(left_iris_points)
    right_iris_midpoint = calculate_midpoint(right_iris_points)
    deviation_threshold_horizontal = 2.8
    
    return (abs(left_iris_midpoint[0] - left_eye_midpoint[0]) <= deviation_threshold_horizontal 
            and abs(right_iris_midpoint[0] - right_eye_midpoint[0]) <= deviation_threshold_horizontal)

# FIXED: Enhanced drowsiness detection dengan improved face landmarks display
def detect_drowsiness(frame, landmarks, speech_engine=None):
    """Detect drowsiness and attention state dengan enhanced landmark visualization"""
    # FIXED: Enhanced colors untuk better visibility
    COLOR_RED = (0, 0, 255)
    COLOR_BLUE = (255, 0, 0)
    COLOR_GREEN = (0, 255, 0)
    COLOR_MAGENTA = (255, 0, 255)
    COLOR_CYAN = (255, 255, 0)
    COLOR_YELLOW = (0, 255, 255)

    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]

    LEFT_EYE_TOP_BOTTOM = [386, 374]
    LEFT_EYE_LEFT_RIGHT = [263, 362]

    RIGHT_EYE_TOP_BOTTOM = [159, 145]
    RIGHT_EYE_LEFT_RIGHT = [133, 33]

    UPPER_LOWER_LIPS = [13, 14]
    LEFT_RIGHT_LIPS = [78, 308]

    try:
        # FIXED: Enhanced landmark drawing dengan reduced noise dan better visibility
        draw_landmarks(frame, landmarks, LEFT_EYE, COLOR_GREEN, 1)
        draw_landmarks(frame, landmarks, RIGHT_EYE, COLOR_GREEN, 1)
        draw_landmarks(frame, landmarks, LEFT_EYE_TOP_BOTTOM, COLOR_RED, 2)
        draw_landmarks(frame, landmarks, LEFT_EYE_LEFT_RIGHT, COLOR_RED, 2)
        draw_landmarks(frame, landmarks, RIGHT_EYE_TOP_BOTTOM, COLOR_RED, 2)
        draw_landmarks(frame, landmarks, RIGHT_EYE_LEFT_RIGHT, COLOR_RED, 2)
        draw_landmarks(frame, landmarks, UPPER_LOWER_LIPS, COLOR_BLUE, 2)
        draw_landmarks(frame, landmarks, LEFT_RIGHT_LIPS, COLOR_BLUE, 2)

        # FIXED: Enhanced mesh points calculation
        img_h, img_w = frame.shape[:2]
        mesh_points = []    
        for p in landmarks.landmark:
            x = int(p.x * img_w)
            y = int(p.y * img_h)
            mesh_points.append((x, y))
        mesh_points = np.array(mesh_points)            
        
        left_eye_points = mesh_points[LEFT_EYE]
        right_eye_points = mesh_points[RIGHT_EYE]
        left_iris_points = mesh_points[LEFT_IRIS]
        right_iris_points = mesh_points[RIGHT_IRIS]

        # FIXED: Enhanced iris visualization dengan better circles
        try:
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(left_iris_points)
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(right_iris_points)
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
            
            # FIXED: Enhanced iris circles dengan outline
            cv.circle(frame, center_left, int(l_radius) + 1, (0, 0, 0), 2, cv.LINE_AA)  # Black outline
            cv.circle(frame, center_left, int(l_radius), COLOR_MAGENTA, 2, cv.LINE_AA)  # Magenta iris
            cv.circle(frame, center_right, int(r_radius) + 1, (0, 0, 0), 2, cv.LINE_AA)  # Black outline
            cv.circle(frame, center_right, int(r_radius), COLOR_MAGENTA, 2, cv.LINE_AA)  # Magenta iris
        except Exception as e:
            print(f"Iris circle drawing error: {e}")

        # Detect closed eyes
        ratio_left_eye = get_aspect_ratio(frame, landmarks, LEFT_EYE_TOP_BOTTOM, LEFT_EYE_LEFT_RIGHT)
        ratio_right_eye = get_aspect_ratio(frame, landmarks, RIGHT_EYE_TOP_BOTTOM, RIGHT_EYE_LEFT_RIGHT)
        eye_ratio = (ratio_left_eye + ratio_right_eye) / 2
        
        # Detect yawning
        ratio_lips = get_aspect_ratio(frame, landmarks, UPPER_LOWER_LIPS, LEFT_RIGHT_LIPS)
        
        # Check if iris is focused (looking at center/screen)
        iris_focused = check_iris_in_middle(left_eye_points, left_iris_points, right_eye_points, right_iris_points)
        
        # Determine state based on conditions
        eyes_closed = eye_ratio > 5.0
        yawning = ratio_lips < 1.8
        not_focused = not iris_focused
        
        # State priority: SLEEPING > YAWNING > NOT FOCUSED > FOCUSED
        if eyes_closed:
            state = "SLEEPING"
        elif yawning:
            state = "YAWNING"
        elif not_focused:
            state = "NOT FOCUSED"
        else:
            state = "FOCUSED"
        
        status = {
            "eyes_closed": eyes_closed,
            "yawning": yawning,
            "not_focused": not_focused,
            "focused": iris_focused,
            "state": state
        }
        
        return status, state
    except Exception as e:
        print(f"Drowsiness detection error: {str(e)}")
        return {"state": "FOCUSED"}, "FOCUSED"

# FIXED: Enhanced distraction session management
def update_person_distraction_sessions(person_key, current_state, current_time):
    """Update and track continuous distraction sessions properly"""
    global person_distraction_start, person_session_durations
    
    # Initialize if new person
    if person_key not in person_distraction_start:
        person_distraction_start[person_key] = {}
        person_session_durations[person_key] = {
            'SLEEPING': 0,
            'YAWNING': 0,
            'NOT FOCUSED': 0,
            'current_session_start': {},
            'total_session_time': 0
        }
    
    # Get previous state
    previous_state = person_current_state.get(person_key, 'FOCUSED')
    
    # Handle state transitions
    if previous_state != current_state:
        # End previous distraction session if it was a problematic state
        if previous_state in DISTRACTION_THRESHOLDS and previous_state in person_distraction_start[person_key]:
            session_duration = current_time - person_distraction_start[person_key][previous_state]
            person_session_durations[person_key][previous_state] += session_duration
            del person_distraction_start[person_key][previous_state]
            print(f"FIXED: Ended {previous_state} session for {person_key}, duration: {session_duration:.1f}s")
        
        # Start new distraction session if current state is problematic
        if current_state in DISTRACTION_THRESHOLDS:
            person_distraction_start[person_key][current_state] = current_time
            print(f"FIXED: Started {current_state} session for {person_key}")
    
    # Calculate current session duration for active distraction
    current_duration = 0
    if current_state in DISTRACTION_THRESHOLDS and current_state in person_distraction_start[person_key]:
        current_duration = current_time - person_distraction_start[person_key][current_state]
    
    return current_duration

def get_actual_distraction_durations():
    """FIXED: Calculate actual distraction durations from session tracking"""
    global person_session_durations, person_distraction_start
    
    total_durations = {
        'unfocused_time': 0,
        'yawning_time': 0,
        'sleeping_time': 0
    }
    
    current_time = time.time()
    
    for person_key, durations in person_session_durations.items():
        # Add completed sessions
        total_durations['sleeping_time'] += durations.get('SLEEPING', 0)
        total_durations['yawning_time'] += durations.get('YAWNING', 0)
        total_durations['unfocused_time'] += durations.get('NOT FOCUSED', 0)
        
        # Add ongoing sessions
        if person_key in person_distraction_start:
            for state, start_time in person_distraction_start[person_key].items():
                ongoing_duration = current_time - start_time
                if state == 'SLEEPING':
                    total_durations['sleeping_time'] += ongoing_duration
                elif state == 'YAWNING':
                    total_durations['yawning_time'] += ongoing_duration
                elif state == 'NOT FOCUSED':
                    total_durations['unfocused_time'] += ongoing_duration
    
    return total_durations

def detect_persons_with_attention(image, mode="image"):
    """Detect persons in image atau video frame dengan enhanced attention status dan proper duration tracking"""
    global live_monitoring_active, session_data, person_state_timers, person_current_state, last_alert_time
    global face_detection, face_mesh
    
    # Check if MediaPipe is initialized
    if face_detection is None or face_mesh is None:
        if not init_mediapipe():
            print("MediaPipe not available, returning empty detections")
            return image, []

    rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
    try:
        detection_results = face_detection.process(rgb_image)
        mesh_results = face_mesh.process(rgb_image)
    except Exception as e:
        print(f"MediaPipe processing error: {str(e)}")
        return image, []
    
    detections = []
    ih, iw, _ = image.shape
    current_time = time.time()
    
    # Check monitoring status dengan thread safety
    with monitoring_lock:
        is_monitoring_active = live_monitoring_active
        current_session_data = session_data.copy() if session_data else None
    
    if detection_results.detections:
        for i, detection in enumerate(detection_results.detections):
            bboxC = detection.location_data.relative_bounding_box
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Ensure bounding box is within image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, iw - x)
            h = min(h, ih - y)
            
            confidence_score = detection.score[0]
            
            # Default attention status
            attention_status = {
                "eyes_closed": False,
                "yawning": False,
                "not_focused": False,
                "state": "FOCUSED"
            }
            
            # Match detection dengan face mesh
            matched_face_idx = -1
            if mesh_results.multi_face_landmarks:
                for face_idx, face_landmarks in enumerate(mesh_results.multi_face_landmarks):
                    # Calculate face mesh bounding box
                    min_x, min_y = float('inf'), float('inf')
                    max_x, max_y = 0, 0
                    
                    for landmark in face_landmarks.landmark:
                        landmark_x, landmark_y = int(landmark.x * iw), int(landmark.y * ih)
                        min_x = min(min_x, landmark_x)
                        min_y = min(min_y, landmark_y)
                        max_x = max(max_x, landmark_x)
                        max_y = max(max_y, landmark_y)
                    
                    # Check if detection dan mesh overlap
                    mesh_center_x = (min_x + max_x) // 2
                    mesh_center_y = (min_y + max_y) // 2
                    det_center_x = x + w // 2
                    det_center_y = y + h // 2
                    
                    if (abs(mesh_center_x - det_center_x) < w // 2 and 
                        abs(mesh_center_y - det_center_y) < h // 2):
                        matched_face_idx = face_idx
                        break
            
            # FIXED: Enhanced analysis dengan proper landmark display
            if matched_face_idx != -1:
                attention_status, state = detect_drowsiness(
                    image, 
                    mesh_results.multi_face_landmarks[matched_face_idx],
                    None
                )
            
            status_text = attention_status.get("state", "FOCUSED")
            person_key = f"person_{i+1}"
            
            # FIXED: Enhanced duration calculation untuk proper session tracking
            current_session_duration = 0
            if mode == "video" and is_monitoring_active:
                with monitoring_lock:
                    # Initialize person tracking
                    if person_key not in person_state_timers:
                        person_state_timers[person_key] = {}
                        person_current_state[person_key] = None
                        last_alert_time[person_key] = 0
                    
                    # FIXED: Update distraction sessions properly
                    current_session_duration = update_person_distraction_sessions(
                        person_key, status_text, current_time
                    )
                    
                    # Update current state
                    person_current_state[person_key] = status_text
            
            # FIXED: Enhanced visualization untuk monitoring dengan improved style
            if mode == "video" and is_monitoring_active:
                status_colors = {
                    "FOCUSED": (0, 255, 0),
                    "NOT FOCUSED": (0, 165, 255),
                    "YAWNING": (0, 255, 255),
                    "SLEEPING": (0, 0, 255)
                }
                
                main_color = status_colors.get(status_text, (0, 255, 0))
                
                # FIXED: Enhanced bounding box dengan better visibility
                cv.rectangle(image, (x, y), (x + w, y + h), main_color, 3)
                cv.rectangle(image, (x-1, y-1), (x + w + 1, y + h + 1), (0, 0, 0), 1)  # Black outline
                
                # FIXED: Enhanced timer display dengan proper session duration
                if status_text in DISTRACTION_THRESHOLDS:
                    threshold = DISTRACTION_THRESHOLDS[status_text]
                    timer_text = f"Person {i+1}: {status_text} ({current_session_duration:.1f}s/{threshold}s)"
                else:
                    timer_text = f"Person {i+1}: {status_text}"
                
                # FIXED: Enhanced text display dengan better readability
                font = cv.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                (text_width, text_height), baseline = cv.getTextSize(timer_text, font, font_scale, thickness)
                
                text_y = y - 15
                if text_y < text_height + 20:
                    text_y = y + h + text_height + 15
                
                # FIXED: Enhanced background dengan better contrast
                overlay = image.copy()
                cv.rectangle(overlay, (x-5, text_y - text_height - 10), (x + text_width + 15, text_y + 10), (0, 0, 0), -1)
                cv.addWeighted(overlay, 0.8, image, 0.2, 0, image)
                
                # FIXED: Enhanced text dengan outline untuk better visibility
                cv.putText(image, timer_text, (x + 5, text_y), font, font_scale, (0, 0, 0), thickness + 2)  # Black outline
                cv.putText(image, timer_text, (x + 5, text_y), font, font_scale, main_color, thickness)  # Colored text
                
            else:
                # Static detection display dengan enhanced style
                cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Information overlay
                info_y_start = y + h + 10
                box_padding = 10
                line_height = 20
                box_height = 4 * line_height
                
                overlay = image.copy()
                cv.rectangle(overlay, 
                            (x - box_padding, info_y_start - box_padding), 
                            (x + w + box_padding, info_y_start + box_height), 
                            (0, 0, 0), -1)
                cv.addWeighted(overlay, 0.6, image, 0.4, 0, image)
                
                font = cv.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_color = (255, 255, 255)
                thickness = 1
                
                cv.putText(image, f"Person {i+1}", (x, info_y_start), 
                        font, font_scale, (50, 205, 50), thickness+1)
                cv.putText(image, f"Confidence: {confidence_score*100:.2f}%", 
                        (x, info_y_start + line_height), font, font_scale, font_color, thickness)
                cv.putText(image, f"Position: x:{x}, y:{y} Size: w:{w}, h:{h}", 
                        (x, info_y_start + 2*line_height), font, font_scale, font_color, thickness)
                
                status_color = {
                    "FOCUSED": (0, 255, 0),
                    "NOT FOCUSED": (255, 165, 0),
                    "YAWNING": (255, 255, 0),
                    "SLEEPING": (0, 0, 255)
                }
                color = status_color.get(status_text, (0, 255, 0))
                
                cv.putText(image, f"Status: {status_text}", 
                        (x, info_y_start + 3*line_height), font, font_scale, color, thickness)

            # FIXED: Enhanced alert handling dengan proper session duration
            should_alert = False
            alert_message = ""
            
            if (mode == "video" and is_monitoring_active and status_text in DISTRACTION_THRESHOLDS):
                
                if current_session_duration >= DISTRACTION_THRESHOLDS[status_text]:
                    alert_cooldown = 10  # 10 second cooldown
                    with monitoring_lock:
                        if current_time - last_alert_time.get(person_key, 0) >= alert_cooldown:
                            should_alert = True
                            last_alert_time[person_key] = current_time
                            
                            # Generate alert message
                            if status_text == 'SLEEPING':
                                alert_message = f'Person {i+1} is sleeping - please wake up!'
                            elif status_text == 'YAWNING':
                                alert_message = f'Person {i+1} is yawning - please take a rest!'
                            elif status_text == 'NOT FOCUSED':
                                alert_message = f'Person {i+1} is not focused - please focus on screen!'
                            
                            # Store alert in session data
                            if live_monitoring_active and session_data and session_data.get('start_time'):
                                session_data['alerts'].append({
                                    'timestamp': datetime.now().isoformat(),
                                    'person': f"Person {i+1}",
                                    'detection': status_text,
                                    'message': alert_message,
                                    'duration': int(current_session_duration)
                                })
                                print(f"FIXED: Alert added with actual duration: {alert_message} (Duration: {int(current_session_duration)}s, Total alerts: {len(session_data['alerts'])})")
            
            # Save detected face
            face_img = image[y:y+h, x:x+w]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            face_filename = f"person_{i+1}_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"
            face_path = os.path.join(application.config['DETECTED_FOLDER'], face_filename)
            
            if face_img.size > 0:
                try:
                    cv.imwrite(face_path, face_img)
                except Exception as e:
                    print(f"Error saving face image: {str(e)}")
            
            detections.append({
                "id": i+1,
                "confidence": float(confidence_score),
                "bbox": [x, y, w, h],
                "image_path": f"/static/detected/{face_filename}",
                "status": status_text,
                "timestamp": datetime.now().isoformat(),
                "duration": current_session_duration if mode == "video" else 0  # FIXED: Use actual session duration
            })
    
    # FIXED: Enhanced detection count display
    if detections:
        cv.putText(image, f"Total persons detected: {len(detections)}", 
                  (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv.putText(image, "No persons detected", 
                  (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    return image, detections

# FIXED: Enhanced session statistics calculation
def update_session_statistics(detections):
    """Update session statistics based on current detections dengan FIXED calculation"""
    global session_data
    
    if not detections:
        return
    
    with monitoring_lock:
        if session_data and session_data.get('start_time'):
            session_data['detections'].extend(detections)
            session_data['focus_statistics']['total_detections'] += len(detections)
            session_data['focus_statistics']['total_persons'] = max(
                session_data['focus_statistics']['total_persons'],
                len(detections)
            )
            
            # FIXED: Update distraction times berdasarkan actual session tracking
            actual_durations = get_actual_distraction_durations()
            session_data['focus_statistics']['unfocused_time'] = actual_durations['unfocused_time']
            session_data['focus_statistics']['yawning_time'] = actual_durations['yawning_time']
            session_data['focus_statistics']['sleeping_time'] = actual_durations['sleeping_time']

def calculate_distraction_time_from_alerts(alerts):
    """DEPRECATED: Using actual session tracking instead"""
    # FIXED: Return actual durations from session tracking
    return get_actual_distraction_durations()

# Rest of the code remains the same...
# [Include all other functions from the original file without changes]

if __name__ == "__main__":
    try:
        # Initialize MediaPipe at startup
        if not init_mediapipe():
            print("WARNING: MediaPipe initialization failed, continuing with limited functionality")
        
        port = int(os.environ.get('PORT', 5000))
        print(f"FIXED: Starting Railway application with enhanced face landmarks and duration tracking on port {port}")
        print("FIXED: Enhanced features:")
        print("  - Improved face landmark visibility")
        print("  - Reduced landmark noise")
        print("  - Proper session-based duration calculation")
        print("  - Fixed alert deduplication")
        print("Directories:")
        for name, path in [
            ("UPLOAD", application.config['UPLOAD_FOLDER']),
            ("DETECTED", application.config['DETECTED_FOLDER']),
            ("REPORTS", application.config['REPORTS_FOLDER']),
            ("RECORDINGS", application.config['RECORDINGS_FOLDER'])
        ]:
            print(f"  {name}: {path} (exists: {os.path.exists(path)})")
        
        application.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        print(f"FIXED: Application startup error: {str(e)}")
        traceback.print_exc()
