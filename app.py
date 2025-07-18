# app.py - Synchronized Duration Tracking System
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import base64
import tempfile
import shutil
import traceback

application = Flask(__name__)

# Configuration
application.config['UPLOAD_FOLDER'] = '/tmp/uploads'
application.config['DETECTED_FOLDER'] = '/tmp/detected'
application.config['REPORTS_FOLDER'] = '/tmp/reports'
application.config['RECORDINGS_FOLDER'] = '/tmp/recordings'
application.config['MAX_CONTENT_PATH'] = 10000000

# Ensure all directories exist
for folder in [application.config['UPLOAD_FOLDER'], application.config['DETECTED_FOLDER'], 
               application.config['REPORTS_FOLDER'], application.config['RECORDINGS_FOLDER']]:
    try:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            os.chmod(folder, 0o755)
        print(f"Directory ready: {folder}")
    except Exception as e:
        print(f"Error creating directory {folder}: {str(e)}")

# Global variables for synchronized tracking
monitoring_lock = threading.RLock()
live_monitoring_active = False

# Synchronized session data structure
session_data = {
    'start_time': None,
    'end_time': None,
    'detections': [],
    'alerts': [],
    'focus_statistics': {
        'total_focused_time': 0,
        'total_unfocused_time': 0,
        'total_yawning_time': 0,
        'total_sleeping_time': 0,
        'total_persons': 0,
        'total_detections': 0
    },
    'recording_path': None,
    'recording_frames': [],
    'session_id': None,
    'client_alerts': [],
    'frame_counter': 0,
    'frame_timestamps': [],
    'total_frames_processed': 0
}

# Video recording variables
video_writer = None
recording_active = False

# Synchronized person tracking with accurate duration calculation
person_distraction_sessions = {}  # Track actual continuous distraction durations
person_current_states = {}       # Current state of each person
person_state_start_times = {}    # When current state started (timestamp)
last_alert_times = {}            # Last time alert was triggered for each person
session_start_time = None        # Global session start time

# Alert thresholds and improved cooldown system
DISTRACTION_THRESHOLDS = {
    'SLEEPING': 10,      # 10 seconds
    'YAWNING': 3.5,      # 3.5 seconds  
    'NOT FOCUSED': 10    # 10 seconds
}

# Reduced cooldown for better user experience (2 seconds as requested)
ALERT_COOLDOWN = 5.0  # 2 seconds between repeated alerts for same distraction

# Frame recording configuration
FRAME_STORAGE_INTERVAL = 2
MAX_STORED_FRAMES = 200
RECORDING_FPS = 5

# Initialize MediaPipe
face_detection = None
face_mesh = None

def init_mediapipe():
    """Initialize MediaPipe with error handling"""
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

def draw_landmarks(image, landmarks, land_mark, color):
    """Draw landmarks with reduced noise"""
    height, width = image.shape[:2]
    for face in land_mark:
        point = landmarks.landmark[face]
        point_scale = ((int)(point.x * width), (int)(point.y * height))     
        cv.circle(image, point_scale, 1, color, 1)

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
    
    if top_bottom_dis == 0:
        return 5.0
    
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

def detect_drowsiness(frame, landmarks, speech_engine=None):
    """Detect drowsiness with improved landmark visualization"""
    COLOR_RED = (0, 0, 255)
    COLOR_BLUE = (255, 0, 0)
    COLOR_GREEN = (0, 255, 0)
    COLOR_MAGENTA = (255, 0, 255)

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

    FACE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
            377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

    try:
        draw_landmarks(frame, landmarks, FACE, COLOR_GREEN)
        draw_landmarks(frame, landmarks, LEFT_EYE_TOP_BOTTOM, COLOR_RED)
        draw_landmarks(frame, landmarks, LEFT_EYE_LEFT_RIGHT, COLOR_RED)
        draw_landmarks(frame, landmarks, RIGHT_EYE_TOP_BOTTOM, COLOR_RED)
        draw_landmarks(frame, landmarks, RIGHT_EYE_LEFT_RIGHT, COLOR_RED)
        draw_landmarks(frame, landmarks, UPPER_LOWER_LIPS, COLOR_BLUE)
        draw_landmarks(frame, landmarks, LEFT_RIGHT_LIPS, COLOR_BLUE)

        # Create mesh points for iris detection
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

        # Draw iris circles
        try:
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(left_iris_points)
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(right_iris_points)
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
            
            cv.circle(frame, center_left, int(l_radius), COLOR_MAGENTA, 1, cv.LINE_AA)
            cv.circle(frame, center_right, int(r_radius), COLOR_MAGENTA, 1, cv.LINE_AA)
        except:
            pass

        # Detect closed eyes
        ratio_left_eye = get_aspect_ratio(frame, landmarks, LEFT_EYE_TOP_BOTTOM, LEFT_EYE_LEFT_RIGHT)
        ratio_right_eye = get_aspect_ratio(frame, landmarks, RIGHT_EYE_TOP_BOTTOM, RIGHT_EYE_LEFT_RIGHT)
        eye_ratio = (ratio_left_eye + ratio_right_eye) / 2
        
        # Detect yawning
        ratio_lips = get_aspect_ratio(frame, landmarks, UPPER_LOWER_LIPS, LEFT_RIGHT_LIPS)
        
        # Check if iris is focused
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

# Synchronized distraction session tracking
def update_distraction_sessions_synchronized(person_id, current_state, current_time):
    """FIXED: Synchronized distraction session tracking with real-time display"""
    global person_distraction_sessions, person_current_states, person_state_start_times
    global session_data, session_start_time
    
    person_key = f"person_{person_id}"
    
    # Initialize tracking for new person
    if person_key not in person_current_states:
        person_current_states[person_key] = None
        person_state_start_times[person_key] = current_time
        person_distraction_sessions[person_key] = {}
    
    # Check if state changed
    previous_state = person_current_states[person_key]
    
    if previous_state != current_state:
        print(f"Person {person_id} state changed: {previous_state} -> {current_state}")
        
        # Close previous session if it was a distraction with proper duration calculation
        if previous_state and previous_state in DISTRACTION_THRESHOLDS:
            session_duration = current_time - person_state_start_times[person_key]
            
            # Add to distraction sessions with exact duration
            if previous_state not in person_distraction_sessions[person_key]:
                person_distraction_sessions[person_key][previous_state] = []
            
            person_distraction_sessions[person_key][previous_state].append({
                'start_time': person_state_start_times[person_key],
                'end_time': current_time,
                'duration': session_duration
            })
            
            print(f"Closed {previous_state} session for person {person_id}: {session_duration:.2f}s")
        
        # Update state and start time
        person_current_states[person_key] = current_state
        person_state_start_times[person_key] = current_time
    
    # Return current session duration for synchronized display
    if current_state in DISTRACTION_THRESHOLDS:
        current_duration = current_time - person_state_start_times[person_key]
        return current_duration
    
    return 0

# Distraction time calculation to match alert history data
def calculate_synchronized_distraction_times():
    """Calculate total distraction times by summing all alert durations from history"""
    global session_data, person_distraction_sessions, person_current_states, person_state_start_times
    
    totals = {
        'total_unfocused_time': 0,
        'total_yawning_time': 0,
        'total_sleeping_time': 0,
        'total_focused_time': 0
    }
    
    current_time = time.time()
    
    # Calculate from actual alert history instead of sessions
    if session_data and session_data.get('alerts'):
        for alert in session_data['alerts']:
            alert_type = alert.get('detection', '')
            # Use the real_time_duration which matches the alert history display
            duration = alert.get('real_time_duration', alert.get('duration', 0))
            
            if alert_type == 'NOT FOCUSED':
                totals['total_unfocused_time'] += duration
            elif alert_type == 'YAWNING':
                totals['total_yawning_time'] += duration
            elif alert_type == 'SLEEPING':
                totals['total_sleeping_time'] += duration
    
    # Calculate total focused time
    if session_data and session_data.get('start_time'):
        if session_data.get('end_time'):
            total_session_time = (session_data['end_time'] - session_data['start_time']).total_seconds()
        else:
            total_session_time = current_time - time.mktime(session_data['start_time'].timetuple())
        
        total_distraction_time = (totals['total_unfocused_time'] + 
                                totals['total_yawning_time'] + 
                                totals['total_sleeping_time'])
        totals['total_focused_time'] = max(0, total_session_time - total_distraction_time)
    
    return totals

def should_trigger_alert_improved(person_id, current_state, current_duration):
    """Improved alert triggering with 2-second cooldown for repeated alerts"""
    global last_alert_times
    
    if current_state not in DISTRACTION_THRESHOLDS:
        return False
    
    threshold = DISTRACTION_THRESHOLDS[current_state]
    person_key = f"person_{person_id}"
    current_time = time.time()
    
    # Check if duration exceeds threshold
    if current_duration < threshold:
        return False
    
    # Improved cooldown system - 2 seconds for repeated alerts
    if person_key in last_alert_times:
        if current_time - last_alert_times[person_key] < ALERT_COOLDOWN:
            return False
    
    last_alert_times[person_key] = current_time
    return True

def detect_persons_with_attention(image, mode="image"):
    """Detect persons with synchronized session tracking"""
    global live_monitoring_active, session_data, face_detection, face_mesh
    global person_distraction_sessions, person_current_states, person_state_start_times
    
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
    
    # Check monitoring status
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
            
            # Match detection with face mesh
            matched_face_idx = -1
            if mesh_results.multi_face_landmarks:
                for face_idx, face_landmarks in enumerate(mesh_results.multi_face_landmarks):
                    min_x, min_y = float('inf'), float('inf')
                    max_x, max_y = 0, 0
                    
                    for landmark in face_landmarks.landmark:
                        landmark_x, landmark_y = int(landmark.x * iw), int(landmark.y * ih)
                        min_x = min(min_x, landmark_x)
                        min_y = min(min_y, landmark_y)
                        max_x = max(max_x, landmark_x)
                        max_y = max(max_y, landmark_y)
                    
                    mesh_center_x = (min_x + max_x) // 2
                    mesh_center_y = (min_y + max_y) // 2
                    det_center_x = x + w // 2
                    det_center_y = y + h // 2
                    
                    if (abs(mesh_center_x - det_center_x) < w // 2 and 
                        abs(mesh_center_y - det_center_y) < h // 2):
                        matched_face_idx = face_idx
                        break
            
            # Analyze attention
            if matched_face_idx != -1:
                attention_status, state = detect_drowsiness(
                    image, 
                    mesh_results.multi_face_landmarks[matched_face_idx],
                    None
                )
            
            status_text = attention_status.get("state", "FOCUSED")
            person_id = i + 1
            
            # Update synchronized distraction sessions
            session_duration = 0
            if mode == "video" and is_monitoring_active:
                session_duration = update_distraction_sessions_synchronized(person_id, status_text, current_time)
                
                # Check if alert should be triggered with improved cooldown
                if should_trigger_alert_improved(person_id, status_text, session_duration):
                    trigger_alert_synchronized(person_id, status_text, session_duration)
            
            # Visualization with synchronized timers
            if mode == "video" and is_monitoring_active:
                status_colors = {
                    "FOCUSED": (0, 255, 0),
                    "NOT FOCUSED": (0, 165, 255),
                    "YAWNING": (0, 255, 255),
                    "SLEEPING": (0, 0, 255)
                }
                
                main_color = status_colors.get(status_text, (0, 255, 0))
                cv.rectangle(image, (x, y), (x + w, y + h), main_color, 3)
                
                # Synchronized timer display matching PDF data
                if status_text in DISTRACTION_THRESHOLDS:
                    threshold = DISTRACTION_THRESHOLDS[status_text]
                    timer_text = f"Person {person_id}: {status_text} ({session_duration:.1f}s/{threshold}s)"
                else:
                    timer_text = f"Person {person_id}: {status_text}"
                
                # Draw text with background
                font = cv.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                (text_width, text_height), baseline = cv.getTextSize(timer_text, font, font_scale, thickness)
                
                text_y = y - 10
                if text_y < text_height + 10:
                    text_y = y + h + text_height + 10
                
                # Semi-transparent background
                overlay = image.copy()
                cv.rectangle(overlay, (x, text_y - text_height - 5), (x + text_width + 10, text_y + 5), (0, 0, 0), -1)
                cv.addWeighted(overlay, 0.7, image, 0.3, 0, image)
                
                cv.putText(image, timer_text, (x + 5, text_y), font, font_scale, main_color, thickness)
            else:
                # Static detection display
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
                
                cv.putText(image, f"Person {person_id}", (x, info_y_start), 
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

            # Save detected face
            face_img = image[y:y+h, x:x+w]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            face_filename = f"person_{person_id}_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"
            face_path = os.path.join(application.config['DETECTED_FOLDER'], face_filename)
            
            if face_img.size > 0:
                try:
                    cv.imwrite(face_path, face_img)
                except Exception as e:
                    print(f"Error saving face image: {str(e)}")
            
            detections.append({
                "id": person_id,
                "confidence": float(confidence_score),
                "bbox": [x, y, w, h],
                "image_path": f"/static/detected/{face_filename}",
                "status": status_text,
                "timestamp": datetime.now().isoformat(),
                "duration": session_duration if mode == "video" else 0
            })
    
    # Display detection count
    if detections:
        cv.putText(image, f"Total persons detected: {len(detections)}", 
                  (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv.putText(image, "No persons detected", 
                  (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return image, detections

def trigger_alert_synchronized(person_id, alert_type, duration):
    """Store alert with exact duration matching display"""
    global session_data
    
    alert_time = datetime.now().strftime("%H:%M:%S")
    
    # Generate alert message
    if alert_type == 'SLEEPING':
        alert_message = f'Person {person_id} is sleeping - please wake up!'
    elif alert_type == 'YAWNING':
        alert_message = f'Person {person_id} is yawning - please take a rest!'
    elif alert_type == 'NOT FOCUSED':
        alert_message = f'Person {person_id} is not focused - please focus on screen!'
    else:
        return
    
    # Store alert with duration that matches alert history display
    with monitoring_lock:
        if live_monitoring_active and session_data and session_data.get('start_time'):
            alert_entry = {
                'timestamp': datetime.now().isoformat(),
                'person': f"Person {person_id}",
                'detection': alert_type,
                'message': alert_message,
                'duration': int(duration),  # Duration in seconds as shown in alert history
                'alert_time': alert_time,
                'real_time_duration': duration  # Exact duration for calculation
            }
            session_data['alerts'].append(alert_entry)
            print(f"Alert stored - {alert_message} (Duration: {duration:.1f}s)")

def update_session_statistics_synchronized(detections):
    """Update session statistics with synchronized tracking"""
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
            
            # Update focus statistics with synchronized calculation
            totals = calculate_synchronized_distraction_times()
            session_data['focus_statistics']['total_focused_time'] = totals['total_focused_time']
            session_data['focus_statistics']['total_unfocused_time'] = totals['total_unfocused_time']
            session_data['focus_statistics']['total_yawning_time'] = totals['total_yawning_time']
            session_data['focus_statistics']['total_sleeping_time'] = totals['total_sleeping_time']

def create_session_recording_from_frames(recording_frames, output_path, session_start_time, session_end_time):
    """Create video recording with proper frame timing"""
    try:
        if not recording_frames:
            print("No frames to create video")
            return None

        actual_duration = session_end_time - session_start_time
        actual_duration_seconds = actual_duration.total_seconds()
        
        if actual_duration_seconds <= 0:
            print("Invalid session duration")
            return None

        fps = RECORDING_FPS
        total_frames_needed = int(fps * actual_duration_seconds)
        if total_frames_needed <= 0:
            total_frames_needed = len(recording_frames) * 5
        
        frame_repeat_count = max(1, total_frames_needed // len(recording_frames))
        
        height, width = recording_frames[0].shape[:2]
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            print(f"Could not open video writer for {output_path}")
            return None

        frames_written = 0
        for i, frame in enumerate(recording_frames):
            if frame is not None and frame.size > 0:
                for repeat in range(frame_repeat_count):
                    out.write(frame)
                    frames_written += 1

        out.release()
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            print(f"Recording created: {output_path}")
            return output_path
        else:
            print("Failed to create session recording")
            return None

    except Exception as e:
        print(f"Error creating session recording: {str(e)}")
        return None

def generate_live_pdf_report(session_data, output_path):
    """Generate PDF report with accurate distraction time calculation"""
    try:
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#3B82F6')
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.HexColor('#1F2937')
        )
        
        # Title
        story.append(Paragraph("Smart Focus Alert - Live Report", title_style))
        story.append(Spacer(1, 5))
        
        # Calculate session duration
        if session_data['start_time'] and session_data['end_time']:
            duration = session_data['end_time'] - session_data['start_time']
            total_session_seconds = duration.total_seconds()
            duration_str = str(duration).split('.')[0]
        else:
            total_session_seconds = 0
            duration_str = "N/A"
        
        # Calculate distraction times from alert history
        unfocused_time = 0
        yawning_time = 0
        sleeping_time = 0
        
        # Sum up all alert durations by type
        alert_durations_by_type = {}
        for alert in session_data.get('alerts', []):
            alert_type = alert.get('detection', '')
            duration = alert.get('real_time_duration', alert.get('duration', 0))
            
            if alert_type not in alert_durations_by_type:
                alert_durations_by_type[alert_type] = []
            alert_durations_by_type[alert_type].append(duration)
        
        # Calculate total time for each distraction type
        unfocused_time = sum(alert_durations_by_type.get('NOT FOCUSED', []))
        yawning_time = sum(alert_durations_by_type.get('YAWNING', []))
        sleeping_time = sum(alert_durations_by_type.get('SLEEPING', []))
        
        # Calculate total distraction time and focused time
        total_distraction_time = unfocused_time + yawning_time + sleeping_time
        focused_time = max(0, total_session_seconds - total_distraction_time)
        
        # Calculate focus accuracy percentage
        if total_session_seconds > 0:
            focus_accuracy = (focused_time / total_session_seconds) * 100
            distraction_percentage = (total_distraction_time / total_session_seconds) * 100
        else:
            focus_accuracy = 0
            distraction_percentage = 0
        
        # Determine focus quality rating
        if focus_accuracy >= 90:
            focus_rating = "Excellent"
            rating_color = colors.HexColor('#10B981')
        elif focus_accuracy >= 75:
            focus_rating = "Good"
            rating_color = colors.HexColor('#3B82F6')
        elif focus_accuracy >= 60:
            focus_rating = "Fair"
            rating_color = colors.HexColor('#F59E0B')
        elif focus_accuracy >= 40:
            focus_rating = "Poor"
            rating_color = colors.HexColor('#EF4444')
        else:
            focus_rating = "Very Poor"
            rating_color = colors.HexColor('#DC2626')
        
        def format_time(seconds):
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        
        # Session Information
        story.append(Paragraph("Session Information", heading_style))
        
        session_info = [
            ['Session Start Time', session_data.get('start_time', datetime.now()).strftime('%m/%d/%Y, %I:%M:%S %p')],
            ['Session Duration', duration_str],
            ['Total Detections', str(session_data['focus_statistics']['total_detections'])],
            ['Total Persons Detected', str(session_data['focus_statistics']['total_persons'])],
            ['Total Alerts Generated', str(len(session_data['alerts']))],
            ['Frames Recorded', str(len(session_data.get('recording_frames', [])))]
        ]
        
        session_table = Table(session_info, colWidths=[2*inch, 4*inch])
        session_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F3F4F6')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E5E7EB')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        story.append(session_table)
        story.append(Spacer(1, 20))
        
        # Focus Accuracy Summary
        story.append(Paragraph("Focus Accuracy Summary", heading_style))
        
        # Create highlighted focus accuracy display
        accuracy_text = f"<para align=center><font size=18 color='{rating_color.hexval()}'><b>{focus_accuracy:.1f}%</b></font></para>"
        story.append(Paragraph(accuracy_text, styles['Normal']))
        story.append(Spacer(1, 10))
        
        rating_text = f"<para align=center><font size=14 color='{rating_color.hexval()}'><b>Focus Quality: {focus_rating}</b></font></para>"
        story.append(Paragraph(rating_text, styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Detailed time breakdown with correct calculations
        focus_breakdown = [
            ['Metric', 'Time', 'Percentage'],
            ['Total Focused Time', format_time(focused_time), f"{(focused_time/total_session_seconds*100):.1f}%" if total_session_seconds > 0 else "0%"],
            ['Total Distraction Time', format_time(total_distraction_time), f"{distraction_percentage:.1f}%"],
            ['- Unfocused Time', format_time(unfocused_time), f"{(unfocused_time/total_session_seconds*100):.1f}%" if total_session_seconds > 0 else "0%"],
            ['- Yawning Time', format_time(yawning_time), f"{(yawning_time/total_session_seconds*100):.1f}%" if total_session_seconds > 0 else "0%"],
            ['- Sleeping Time', format_time(sleeping_time), f"{(sleeping_time/total_session_seconds*100):.1f}%" if total_session_seconds > 0 else "0%"]
        ]
        
        breakdown_table = Table(focus_breakdown, colWidths=[2*inch, 2*inch, 2*inch])
        breakdown_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3B82F6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E5E7EB')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F9FAFB')]),
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#ECFDF5')),
            ('TEXTCOLOR', (0, 1), (-1, 1), colors.HexColor('#065F46')),
            ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#FEF2F2')),
            ('TEXTCOLOR', (0, 2), (-1, 2), colors.HexColor('#991B1B')),
        ]))
        
        story.append(breakdown_table)
        story.append(Spacer(1, 15))
        
        # Focus Statistics
        story.append(Paragraph("Detailed Focus Statistics", heading_style))
        
        # Calculate meaningful statistics
        if total_session_seconds > 0:
            focused_minutes = focused_time / 60
            total_minutes = total_session_seconds / 60
            average_focus_metric = f"{focused_minutes:.1f} min focused out of {total_minutes:.1f} min total"
        else:
            average_focus_metric = "N/A"
        
        # Get most common distraction
        most_common_distraction = get_most_common_distraction_from_alerts(session_data.get('alerts', []))
        
        focus_stats = [
            ['Total Session Duration', format_time(total_session_seconds)],
            ['Focus Accuracy Score', f"{focus_accuracy:.2f}%"],
            ['Focus Quality Rating', focus_rating],
            ['Average Focus Metric', average_focus_metric],
            ['Distraction Frequency', f"{len(session_data['alerts'])} alerts in {format_time(total_session_seconds)}"],
            ['Most Common Distraction', most_common_distraction],
            ['Recording Quality', f"{len(session_data.get('recording_frames', []))} frames captured"]
        ]
        
        focus_table = Table(focus_stats, colWidths=[2*inch, 4*inch])
        focus_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F3F4F6')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E5E7EB')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        story.append(focus_table)
        story.append(Spacer(1, 15))
        
        # Alert History with correct durations
        if session_data['alerts']:
            story.append(Paragraph("Alert History", heading_style))
            
            alert_headers = ['Time', 'Person', 'Detection', 'Duration', 'Message']
            alert_data = [alert_headers]
            
            for alert in session_data['alerts'][-15:]:  # Show last 15 alerts
                try:
                    alert_time = datetime.fromisoformat(alert['timestamp']).strftime('%I:%M:%S %p')
                except:
                    alert_time = alert.get('alert_time', 'N/A')
                
                # Use the duration that matches alert history display
                duration = alert.get('real_time_duration', alert.get('duration', 0))
                duration_text = f"{duration:.0f}s" if duration > 0 else "N/A"
                
                alert_data.append([
                    alert_time,
                    alert['person'],
                    alert['detection'],
                    duration_text,
                    alert['message']
                ])
            
            alert_table = Table(alert_data, colWidths=[1*inch, 0.8*inch, 1*inch, 0.8*inch, 2.4*inch])
            alert_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3B82F6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E5E7EB')),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F9FAFB')])
            ]))
            
            story.append(alert_table)
        
        # Footer
        story.append(Spacer(1, 10))
        footer_text = f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>Smart Focus Alert System"
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#6B7280')
        )
        story.append(Paragraph(footer_text, footer_style))
        
        doc.build(story)
        print(f"PDF report generated with correct distraction time calculation: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error generating PDF report: {str(e)}")
        traceback.print_exc()
        return None

def get_most_common_distraction_from_alerts(alerts):
    """Get most common distraction from alert history"""
    if not alerts:
        return "None"
    
    distraction_counts = {}
    distraction_totals = {}
    
    for alert in alerts:
        alert_type = alert.get('detection', '')
        duration = alert.get('real_time_duration', alert.get('duration', 0))
        
        if alert_type not in distraction_counts:
            distraction_counts[alert_type] = 0
            distraction_totals[alert_type] = 0
        
        distraction_counts[alert_type] += 1
        distraction_totals[alert_type] += duration
    
    if not distraction_counts:
        return "None"
    
    # Find most common by total duration
    most_common = max(distraction_totals, key=distraction_totals.get)
    count = distraction_counts[most_common]
    total_duration = int(distraction_totals[most_common])
    
    return f"{most_common} ({count} alerts, {total_duration}s total)"

def process_video_file(video_path):
    """Process video file and detect persons in each frame"""
    cap = cv.VideoCapture(video_path)
    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"processed_{timestamp}_{uuid.uuid4().hex[:8]}.mp4"
    output_path = os.path.join(application.config['DETECTED_FOLDER'], output_filename)
    
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_path, fourcc, fps, (width, height))
    
    all_detections = []
    frame_count = 0
    process_every_n_frames = 5
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % process_every_n_frames == 0:
            processed_frame, detections = detect_persons_with_attention(frame, mode="video")
            all_detections.extend(detections)
        else:
            processed_frame = frame
            
        out.write(processed_frame)
    
    cap.release()
    out.release()
    
    return output_path, all_detections

def generate_upload_pdf_report(detections, file_info, output_path):
    """Generate PDF report for uploaded file analysis"""
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#3B82F6')
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20,
        textColor=colors.HexColor('#1F2937')
    )
    
    # Title
    story.append(Paragraph("Smart Focus Alert - Analysis Report", title_style))
    story.append(Spacer(1, 20))
    
    # File Information
    story.append(Paragraph("File Information", heading_style))
    
    file_info_data = [
        ['File Name', file_info.get('filename', 'Unknown')],
        ['File Type', file_info.get('type', 'Unknown')],
        ['Analysis Date', datetime.now().strftime('%m/%d/%Y, %I:%M:%S %p')],
        ['Total Persons Detected', str(len(detections))]
    ]
    
    file_table = Table(file_info_data, colWidths=[2*inch, 4*inch])
    file_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F3F4F6')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E5E7EB')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    story.append(file_table)
    story.append(Spacer(1, 20))
    
    # Analysis Statistics
    story.append(Paragraph("Analysis Statistics", heading_style))
    
    # Count statuses
    status_counts = {'FOCUSED': 0, 'NOT FOCUSED': 0, 'YAWNING': 0, 'SLEEPING': 0}
    for detection in detections:
        status = detection.get('status', 'FOCUSED')
        if status in status_counts:
            status_counts[status] += 1
    
    total_detections = len(detections)
    focus_accuracy = 0
    if total_detections > 0:
        focus_accuracy = (status_counts['FOCUSED'] / total_detections) * 100
    
    analysis_stats = [
        ['Focus Accuracy', f"{focus_accuracy:.1f}%"],
        ['Focused Persons', str(status_counts['FOCUSED'])],
        ['Unfocused Persons', str(status_counts['NOT FOCUSED'])],
        ['Yawning Persons', str(status_counts['YAWNING'])],
        ['Sleeping Persons', str(status_counts['SLEEPING'])]
    ]
    
    analysis_table = Table(analysis_stats, colWidths=[3*inch, 3*inch])
    analysis_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F3F4F6')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E5E7EB')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    story.append(analysis_table)
    story.append(Spacer(1, 20))
    
    # Individual Results
    if detections:
        story.append(Paragraph("Individual Detection Results", heading_style))
        
        detection_headers = ['Person ID', 'Status', 'Confidence', 'Position (X,Y)', 'Size (W,H)']
        detection_data = [detection_headers]
        
        for detection in detections:
            bbox = detection.get('bbox', [0, 0, 0, 0])
            detection_data.append([
                f"Person {detection.get('id', 'N/A')}",
                detection.get('status', 'Unknown'),
                f"{detection.get('confidence', 0)*100:.1f}%",
                f"({bbox[0]}, {bbox[1]})",
                f"({bbox[2]}, {bbox[3]})"
            ])
        
        detection_table = Table(detection_data, colWidths=[1*inch, 1*inch, 1*inch, 1.5*inch, 1.5*inch])
        detection_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3B82F6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E5E7EB')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F9FAFB')])
        ]))
        
        story.append(detection_table)
    
    # Footer
    story.append(Spacer(1, 30))
    footer_text = f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>Smart Focus Alert System - File Analysis Report"
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=10,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#6B7280')
    )
    story.append(Paragraph(footer_text, footer_style))
    
    doc.build(story)
    return output_path

# Flask Routes
@application.route('/')
def index():
    return render_template('index.html')

@application.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', error='No file part')
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('upload.html', error='No selected file')
        
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(application.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
            
            result = {
                "filename": filename,
                "file_path": f"/static/uploads/{filename}",
                "detections": []
            }
            
            if file_ext in ['jpg', 'jpeg', 'png', 'bmp']:
                # Process image
                image = cv.imread(file_path)
                processed_image, detections = detect_persons_with_attention(image)
                
                # Save processed image
                output_filename = f"processed_{filename}"
                output_path = os.path.join(application.config['DETECTED_FOLDER'], output_filename)
                cv.imwrite(output_path, processed_image)
                
                result["processed_image"] = f"/static/detected/{output_filename}"
                result["detections"] = detections
                result["type"] = "image"
                
                # Generate PDF report
                pdf_filename = f"report_{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                pdf_path = os.path.join(application.config['REPORTS_FOLDER'], pdf_filename)
                
                file_info = {
                    'filename': filename,
                    'type': file_ext.upper()
                }
                
                generate_upload_pdf_report(detections, file_info, pdf_path)
                result["pdf_report"] = f"/static/reports/{pdf_filename}"
                
            elif file_ext in ['mp4', 'avi', 'mov', 'mkv']:
                # Process video
                output_path, detections = process_video_file(file_path)
                
                result["processed_video"] = f"/static/detected/{os.path.basename(output_path)}"
                result["detections"] = detections
                result["type"] = "video"
                
                # Generate PDF report
                pdf_filename = f"report_{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                pdf_path = os.path.join(application.config['REPORTS_FOLDER'], pdf_filename)
                
                file_info = {
                    'filename': filename,
                    'type': file_ext.upper()
                }
                
                generate_upload_pdf_report(detections, file_info, pdf_path)
                result["pdf_report"] = f"/static/reports/{pdf_filename}"
            
            return render_template('result.html', result=result)
    
    return render_template('upload.html')

@application.route('/live')
def live():
    return render_template('live.html')

@application.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    """Start monitoring with synchronized session tracking"""
    global live_monitoring_active, session_data, recording_active
    global person_distraction_sessions, person_current_states, person_state_start_times
    global last_alert_times, session_start_time
    
    try:
        request_data = request.get_json() or {}
        client_session_id = request_data.get('sessionId')
        
        with monitoring_lock:
            if live_monitoring_active:
                return jsonify({"status": "error", "message": "Monitoring already active"})
            
            # Reset all tracking variables with synchronized initialization
            session_data = {
                'start_time': datetime.now(),
                'end_time': None,
                'detections': [],
                'alerts': [],
                'focus_statistics': {
                    'total_focused_time': 0,
                    'total_unfocused_time': 0,
                    'total_yawning_time': 0,
                    'total_sleeping_time': 0,
                    'total_persons': 0,
                    'total_detections': 0
                },
                'recording_path': None,
                'recording_frames': [],
                'session_id': client_session_id,
                'client_alerts': [],
                'frame_counter': 0,
                'frame_timestamps': [],
                'total_frames_processed': 0
            }
            
            # Reset synchronized session tracking
            person_distraction_sessions = {}
            person_current_states = {}
            person_state_start_times = {}
            last_alert_times = {}
            session_start_time = time.time()
            
            live_monitoring_active = True
            recording_active = True
            
            print(f"Synchronized session tracking started at {session_data['start_time']}")
            
            return jsonify({
                "status": "success", 
                "message": "synchronized session tracking started", 
                "session_id": client_session_id
            })
        
    except Exception as e:
        print(f"Error starting monitoring: {str(e)}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Failed to start monitoring: {str(e)}"})

@application.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    """Stop monitoring with finalized synchronized session tracking"""
    global live_monitoring_active, session_data, recording_active
    global person_distraction_sessions, person_current_states, person_state_start_times
    
    try:
        request_data = request.get_json() or {}
        client_alerts = request_data.get('alerts', [])
        client_session_id = request_data.get('sessionId')
        
        with monitoring_lock:
            if not live_monitoring_active and (not session_data or not session_data.get('start_time')):
                return jsonify({"status": "error", "message": "Monitoring not active"})
            
            # Finalize all ongoing distraction sessions with synchronized timing
            current_time = time.time()
            for person_key, current_state in person_current_states.items():
                if current_state and current_state in DISTRACTION_THRESHOLDS:
                    if person_key in person_state_start_times:
                        session_duration = current_time - person_state_start_times[person_key]
                        
                        # Add final session to distraction sessions
                        if current_state not in person_distraction_sessions.get(person_key, {}):
                            if person_key not in person_distraction_sessions:
                                person_distraction_sessions[person_key] = {}
                            person_distraction_sessions[person_key][current_state] = []
                        
                        person_distraction_sessions[person_key][current_state].append({
                            'start_time': person_state_start_times[person_key],
                            'end_time': current_time,
                            'duration': session_duration
                        })
                        
                        print(f"Finalized {current_state} session for {person_key}: {session_duration:.2f}s")
            
            # Merge client alerts if provided
            if client_alerts:
                session_data['client_alerts'] = client_alerts
                print(f"Merged {len(client_alerts)} client alerts")
            
            # Stop monitoring
            live_monitoring_active = False
            recording_active = False
            session_data['end_time'] = datetime.now()
            
            print(f"Synchronized session tracking stopped at {session_data['end_time']}")
            
            response_data = {
                "status": "success", 
                "message": "synchronized session tracking stopped",
                "alerts_processed": len(session_data['alerts']),
                "frames_captured": len(session_data.get('recording_frames', [])),
                "distraction_sessions": len(person_distraction_sessions)
            }
            
            # Generate PDF report with synchronized data
            try:
                pdf_filename = f"session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.pdf"
                pdf_path = os.path.join(application.config['REPORTS_FOLDER'], pdf_filename)
                
                pdf_result = generate_live_pdf_report(session_data, pdf_path)
                
                if pdf_result and os.path.exists(pdf_path):
                    response_data["pdf_report"] = f"/static/reports/{pdf_filename}"
                    print(f"SYNC PDF SUCCESS: {pdf_filename}")
                else:
                    print("SYNC PDF FAILED: File not created")
                    
            except Exception as pdf_error:
                print(f"SYNC PDF ERROR: {str(pdf_error)}")
                traceback.print_exc()
            
            # Generate video recording
            try:
                recording_filename = f"session_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.mp4"
                recording_path = os.path.join(application.config['RECORDINGS_FOLDER'], recording_filename)
                
                if len(session_data.get('recording_frames', [])) > 0:
                    video_result = create_session_recording_from_frames(
                        session_data['recording_frames'],
                        recording_path,
                        session_data.get('start_time', datetime.now() - timedelta(seconds=10)),
                        session_data.get('end_time', datetime.now())
                    )
                    
                    if video_result and os.path.exists(recording_path):
                        response_data["video_file"] = f"/static/recordings/{os.path.basename(recording_path)}"
                        session_data['recording_path'] = recording_path
                        print(f"SYNC VIDEO SUCCESS: {os.path.basename(recording_path)}")
                    else:
                        print("SYNC VIDEO FAILED: Unable to create from frames")
                else:
                    print("SYNC VIDEO SKIPPED: No frames available")
                    
            except Exception as video_error:
                print(f"SYNC VIDEO ERROR: {str(video_error)}")
                traceback.print_exc()
            
            return jsonify(response_data)
        
    except Exception as e:
        print(f"Error stopping monitoring: {str(e)}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Failed to stop monitoring: {str(e)}"})

@application.route('/process_frame', methods=['POST'])
def process_frame():
    """Frame processing with synchronized session tracking"""
    global session_data
    
    try:
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({"error": "No frame data provided"}), 400
            
        frame_data = data['frame'].split(',')[1]
        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv.imdecode(nparr, cv.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Invalid frame data"}), 400
        
        # Process frame for detection
        processed_frame, detections = detect_persons_with_attention(frame, mode="video")
        
        # Frame storage with consistent logic
        with monitoring_lock:
            if live_monitoring_active and recording_active and session_data:
                session_data['frame_counter'] = session_data.get('frame_counter', 0) + 1
                session_data['total_frames_processed'] = session_data.get('total_frames_processed', 0) + 1
                current_timestamp = time.time()
                
                should_store_frame = (
                    session_data['frame_counter'] % FRAME_STORAGE_INTERVAL == 0 or
                    len(detections) > 0 or
                    len(session_data.get('recording_frames', [])) < 10
                )
                
                if should_store_frame:
                    frame_copy = processed_frame.copy()
                    session_data['recording_frames'].append(frame_copy)
                    session_data['frame_timestamps'].append(current_timestamp)
                    
                    if len(session_data['recording_frames']) > MAX_STORED_FRAMES:
                        frames_to_remove = len(session_data['recording_frames']) - MAX_STORED_FRAMES
                        session_data['recording_frames'] = session_data['recording_frames'][frames_to_remove:]
                        session_data['frame_timestamps'] = session_data['frame_timestamps'][frames_to_remove:]
        
        # Update session statistics with synchronized tracking
        if live_monitoring_active and detections:
            update_session_statistics_synchronized(detections)
        
        # Encode processed frame
        _, buffer = cv.imencode('.jpg', processed_frame, [cv.IMWRITE_JPEG_QUALITY, 85])
        processed_frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "success": True,
            "processed_frame": f"data:image/jpeg;base64,{processed_frame_b64}",
            "detections": detections,
            "frame_count": len(session_data.get('recording_frames', [])) if session_data else 0,
            "total_processed": session_data.get('total_frames_processed', 0) if session_data else 0,
            "frame_number": session_data.get('frame_counter', 0) if session_data else 0
        })
        
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Frame processing failed: {str(e)}"}), 500

@application.route('/sync_alerts', methods=['POST'])
def sync_alerts():
    """Sync client-side alerts with server for synchronized tracking"""
    try:
        request_data = request.get_json() or {}
        client_alerts = request_data.get('alerts', [])
        session_id = request_data.get('sessionId')
        
        with monitoring_lock:
            if session_data and session_data.get('session_id') == session_id:
                session_data['client_alerts'] = client_alerts
                print(f"SYNC: Synced {len(client_alerts)} client alerts for session {session_id}")
                return jsonify({"status": "success", "synced_count": len(client_alerts)})
            else:
                return jsonify({"status": "error", "message": "Session mismatch"})
                
    except Exception as e:
        print(f"Alert sync error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

@application.route('/get_monitoring_data')
def get_monitoring_data():
    """Monitoring data endpoint with synchronized information"""
    global session_data
    
    try:
        with monitoring_lock:
            if not live_monitoring_active:
                return jsonify({"error": "Monitoring not active"})
            
            current_alerts = session_data.get('alerts', []) if session_data else []
            recent_alerts = current_alerts[-5:] if current_alerts else []
            
            formatted_alerts = []
            for alert in recent_alerts:
                try:
                    alert_time = datetime.fromisoformat(alert['timestamp']).strftime('%H:%M:%S')
                except:
                    alert_time = alert.get('alert_time', 'N/A')
                
                # Show synchronized duration from real-time tracking
                duration = alert.get('real_time_duration', alert.get('duration', 0))
                duration_text = f" ({duration:.1f}s)" if duration > 0 else ""
                
                formatted_alerts.append({
                    'time': alert_time,
                    'message': alert['message'] + duration_text,
                    'type': 'warning' if alert['detection'] in ['YAWNING', 'NOT FOCUSED'] else 'error',
                    'duration': duration
                })
            
            current_detections = session_data.get('detections', []) if session_data else []
            recent_detections = current_detections[-10:] if current_detections else []
            current_status = 'READY'
            focused_count = 0
            total_persons = 0
            
            if recent_detections:
                latest_states = {}
                for detection in reversed(recent_detections):
                    person_id = detection['id']
                    if person_id not in latest_states:
                        latest_states[person_id] = detection['status']
                
                total_persons = len(latest_states)
                focused_count = sum(1 for state in latest_states.values() if state == 'FOCUSED')
                
                if all(state == 'FOCUSED' for state in latest_states.values()):
                    current_status = 'FOCUSED'
                elif any(state == 'SLEEPING' for state in latest_states.values()):
                    current_status = 'SLEEPING'
                elif any(state == 'YAWNING' for state in latest_states.values()):
                    current_status = 'YAWNING'
                elif any(state == 'NOT FOCUSED' for state in latest_states.values()):
                    current_status = 'NOT FOCUSED'
            
            return jsonify({
                'total_persons': total_persons,
                'focused_count': focused_count,
                'alert_count': len(current_alerts),
                'current_status': current_status,
                'latest_alerts': formatted_alerts,
                'frame_count': len(session_data.get('recording_frames', [])) if session_data else 0,
                'total_processed': session_data.get('total_frames_processed', 0) if session_data else 0
            })
        
    except Exception as e:
        print(f"Error getting monitoring data: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Failed to get monitoring data: {str(e)}"})

@application.route('/monitoring_status')
def monitoring_status():
    """Get current monitoring status with synchronized information"""
    try:
        with monitoring_lock:
            return jsonify({
                "is_active": live_monitoring_active,
                "session_id": session_data.get('session_id') if session_data else None,
                "alerts_count": len(session_data.get('alerts', [])) if session_data else 0,
                "frames_stored": len(session_data.get('recording_frames', [])) if session_data else 0,
                "frames_processed": session_data.get('total_frames_processed', 0) if session_data else 0,
                "distraction_sessions": len(person_distraction_sessions),
                "synchronized_tracking": True
            })
    except Exception as e:
        print(f"Error getting monitoring status: {str(e)}")
        return jsonify({"is_active": False})

@application.route('/check_camera')
def check_camera():
    """Check camera availability"""
    try:
        return jsonify({"camera_available": False})  # Force client-side camera
    except Exception as e:
        print(f"Error checking camera: {str(e)}")
        return jsonify({"camera_available": False})

@application.route('/health')
def health_check():
    """Health check endpoint with synchronized information"""
    try:
        with monitoring_lock:
            return jsonify({
                "status": "healthy", 
                "timestamp": datetime.now().isoformat(),
                "directories": {
                    "uploads": os.path.exists(application.config['UPLOAD_FOLDER']),
                    "detected": os.path.exists(application.config['DETECTED_FOLDER']),
                    "reports": os.path.exists(application.config['REPORTS_FOLDER']),
                    "recordings": os.path.exists(application.config['RECORDINGS_FOLDER'])
                },
                "monitoring_active": live_monitoring_active,
                "session_alerts": len(session_data.get('alerts', [])) if session_data else 0,
                "recording_frames": len(session_data.get('recording_frames', [])) if session_data else 0,
                "total_frames_processed": session_data.get('total_frames_processed', 0) if session_data else 0,
                "frame_storage_ratio": len(session_data.get('recording_frames', [])) / max(1, session_data.get('total_frames_processed', 1)) * 100 if session_data else 0,
                "mediapipe_status": "initialized" if face_detection and face_mesh else "error",
                "distraction_sessions": len(person_distraction_sessions),
                "synchronized_tracking": True,
                "alert_cooldown": ALERT_COOLDOWN,
                "version": "synchronized_duration_tracking_v1.0"
            })
    except Exception as e:
        print(f"Health check error: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@application.route('/api/detect', methods=['POST'])
def api_detect():
    """API endpoint for file detection"""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(application.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    
    if file_ext in ['jpg', 'jpeg', 'png', 'bmp']:
        image = cv.imread(file_path)
        processed_image, detections = detect_persons_with_attention(image)
        
        output_filename = f"processed_{filename}"
        output_path = os.path.join(application.config['DETECTED_FOLDER'], output_filename)
        cv.imwrite(output_path, processed_image)
        
        return jsonify({
            "type": "image",
            "processed_image": f"/static/detected/{output_filename}",
            "detections": detections
        })
        
    elif file_ext in ['mp4', 'avi', 'mov', 'mkv']:
        output_path, detections = process_video_file(file_path)
        
        return jsonify({
            "type": "video",
            "processed_video": f"/static/detected/{os.path.basename(output_path)}",
            "detections": detections
        })
    
    return jsonify({"error": "Unsupported file format"}), 400

# Static file routes
@application.route('/static/reports/<filename>')
def report_file(filename):
    """Serve PDF report files"""
    try:
        file_path = os.path.join(application.config['REPORTS_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_from_directory(
                application.config['REPORTS_FOLDER'], 
                filename,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=filename
            )
        else:
            return jsonify({"error": "Report file not found"}), 404
    except Exception as e:
        print(f"Error serving report file: {str(e)}")
        return jsonify({"error": "Error accessing report file"}), 500

@application.route('/static/recordings/<filename>')
def recording_file(filename):
    """Serve video recording files"""
    try:
        file_path = os.path.join(application.config['RECORDINGS_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_from_directory(
                application.config['RECORDINGS_FOLDER'], 
                filename,
                mimetype='video/mp4',
                as_attachment=True,
                download_name=filename
            )
        else:
            return jsonify({"error": "Recording file not found"}), 404
    except Exception as e:
        print(f"Error serving recording file: {str(e)}")
        return jsonify({"error": "Error accessing recording file"}), 500

@application.route('/static/detected/<filename>')
def detected_file(filename):
    """Serve detected image files"""
    try:
        return send_from_directory(application.config['DETECTED_FOLDER'], filename)
    except Exception as e:
        print(f"Error serving detected file: {str(e)}")
        return jsonify({"error": "Error accessing detected file"}), 500

@application.route('/static/uploads/<filename>')
def upload_file(filename):
    """Serve uploaded files"""
    try:
        return send_from_directory(application.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        print(f"Error serving upload file: {str(e)}")
        return jsonify({"error": "Error accessing upload file"}), 500

if __name__ == "__main__":
    try:
        # Initialize MediaPipe at startup
        if not init_mediapipe():
            print("WARNING: MediaPipe initialization failed, continuing with limited functionality")
        
        port = int(os.environ.get('PORT', 5000))
        print(f"Starting Smart Focus Alert with SYNCHRONIZED Duration Tracking on port {port}")
        print("Frame storage configuration:")
        print(f"  - Storage interval: every {FRAME_STORAGE_INTERVAL} frames")
        print(f"  - Max stored frames: {MAX_STORED_FRAMES}")
        print(f"  - Recording FPS: {RECORDING_FPS}")
        print(f"Alert configuration:")
        print(f"  - Alert cooldown: {ALERT_COOLDOWN} seconds")
        print(f"  - Thresholds: {DISTRACTION_THRESHOLDS}")
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
        print(f"Application startup error: {str(e)}")
        traceback.print_exc()
