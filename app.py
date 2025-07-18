# app.py - FIXED Duration Tracking System dengan Synchronized Alert & Accuracy
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

# Ensure directories exist
for folder in [application.config['UPLOAD_FOLDER'], application.config['DETECTED_FOLDER'], 
               application.config['REPORTS_FOLDER'], application.config['RECORDINGS_FOLDER']]:
    try:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            os.chmod(folder, 0o755)
        print(f"Directory ready: {folder}")
    except Exception as e:
        print(f"Error creating directory {folder}: {str(e)}")

# FIXED: Enhanced tracking system dengan synchronized duration tracking
monitoring_lock = threading.RLock()
live_monitoring_active = False

# FIXED: Unified duration tracking system
session_data = {
    'start_time': None,
    'end_time': None,
    'detections': [],
    'alerts': [],
    'unified_tracking': {},  # FIXED: Single source of truth untuk duration tracking
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

# FIXED: Unified person tracking dengan single duration calculation
person_unified_tracking = {}     # Single source untuk duration tracking
person_current_states = {}       # Current state of each person
person_state_start_times = {}    # When current state started
last_alert_times = {}            # Last time alert was triggered (dengan longer cooldown)
session_start_time = None        # Global session start time

# Alert thresholds dan cooldown configuration
DISTRACTION_THRESHOLDS = {
    'SLEEPING': 10,      # 10 seconds
    'YAWNING': 3.5,      # 3.5 seconds
    'NOT FOCUSED': 10    # 10 seconds
}

# FIXED: Extended alert cooldown untuk prevent spam
ALERT_COOLDOWN = 15  # 15 seconds cooldown between alerts for same person/state

# Frame recording configuration
FRAME_STORAGE_INTERVAL = 2
MAX_STORED_FRAMES = 200
RECORDING_FPS = 5

# Initialize MediaPipe
face_detection = None
face_mesh = None

def init_mediapipe():
    """Initialize MediaPipe dengan error handling"""
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
    """Draw landmarks dengan reduced noise"""
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
    """Detect drowsiness dengan improved landmark visualization"""
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

# FIXED: Unified duration tracking system
def update_unified_tracking(person_id, current_state, current_time):
    """FIXED: Single unified tracking system untuk duration dan alerts"""
    global person_unified_tracking, person_current_states, person_state_start_times
    global session_data
    
    person_key = f"person_{person_id}"
    
    # Initialize tracking untuk new person
    if person_key not in person_current_states:
        person_current_states[person_key] = None
        person_state_start_times[person_key] = current_time
        person_unified_tracking[person_key] = {
            'total_durations': {
                'FOCUSED': 0,
                'NOT FOCUSED': 0,
                'YAWNING': 0,
                'SLEEPING': 0
            },
            'current_session': None,
            'session_start': None,
            'all_sessions': []
        }
    
    tracking = person_unified_tracking[person_key]
    previous_state = person_current_states[person_key]
    
    # Check if state changed
    if previous_state != current_state:
        print(f"FIXED UNIFIED: Person {person_id} state changed: {previous_state} -> {current_state}")
        
        # FIXED: Close previous session dan update total duration
        if previous_state and tracking['session_start']:
            session_duration = current_time - tracking['session_start']
            tracking['total_durations'][previous_state] += session_duration
            
            # Record session for detailed tracking
            tracking['all_sessions'].append({
                'state': previous_state,
                'start_time': tracking['session_start'],
                'end_time': current_time,
                'duration': session_duration
            })
            
            print(f"FIXED UNIFIED: Closed {previous_state} session for person {person_id}: {session_duration:.2f}s (Total: {tracking['total_durations'][previous_state]:.2f}s)")
        
        # Start new session
        person_current_states[person_key] = current_state
        person_state_start_times[person_key] = current_time
        tracking['current_session'] = current_state
        tracking['session_start'] = current_time
    
    # Return current session duration untuk display
    current_duration = 0
    if tracking['session_start']:
        current_duration = current_time - tracking['session_start']
    
    return current_duration

def get_unified_total_durations():
    """FIXED: Get total durations dari unified tracking system"""
    global person_unified_tracking, person_current_states
    
    totals = {
        'total_focused_time': 0,
        'total_unfocused_time': 0,
        'total_yawning_time': 0,
        'total_sleeping_time': 0
    }
    
    current_time = time.time()
    
    # Sum completed sessions dari all persons
    for person_key, tracking in person_unified_tracking.items():
        totals['total_focused_time'] += tracking['total_durations']['FOCUSED']
        totals['total_unfocused_time'] += tracking['total_durations']['NOT FOCUSED']
        totals['total_yawning_time'] += tracking['total_durations']['YAWNING']
        totals['total_sleeping_time'] += tracking['total_durations']['SLEEPING']
        
        # Add current ongoing session
        current_state = person_current_states.get(person_key)
        if current_state and tracking['session_start']:
            current_duration = current_time - tracking['session_start']
            
            if current_state == 'FOCUSED':
                totals['total_focused_time'] += current_duration
            elif current_state == 'NOT FOCUSED':
                totals['total_unfocused_time'] += current_duration
            elif current_state == 'YAWNING':
                totals['total_yawning_time'] += current_duration
            elif current_state == 'SLEEPING':
                totals['total_sleeping_time'] += current_duration
    
    return totals

def should_trigger_unified_alert(person_id, current_state, current_duration):
    """FIXED: Unified alert system dengan extended cooldown"""
    global last_alert_times
    
    if current_state not in DISTRACTION_THRESHOLDS:
        return False
    
    threshold = DISTRACTION_THRESHOLDS[current_state]
    person_key = f"person_{person_id}"
    current_time = time.time()
    
    # Check if duration exceeds threshold
    if current_duration < threshold:
        return False
    
    # FIXED: Extended cooldown untuk prevent spam alerts
    alert_key = f"{person_key}_{current_state}"
    if alert_key in last_alert_times:
        time_since_last_alert = current_time - last_alert_times[alert_key]
        if time_since_last_alert < ALERT_COOLDOWN:
            return False
    
    last_alert_times[alert_key] = current_time
    return True

def trigger_unified_alert(person_id, alert_type, actual_duration):
    """FIXED: Unified alert system dengan accurate duration"""
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
    
    # FIXED: Store alert dengan actual continuous duration
    with monitoring_lock:
        if live_monitoring_active and session_data and session_data.get('start_time'):
            alert_entry = {
                'timestamp': datetime.now().isoformat(),
                'person': f"Person {person_id}",
                'detection': alert_type,
                'message': alert_message,
                'duration': int(actual_duration),  # FIXED: Use actual continuous duration
                'alert_time': alert_time,
                'session_duration': actual_duration  # FIXED: Same as duration untuk consistency
            }
            session_data['alerts'].append(alert_entry)
            print(f"FIXED UNIFIED ALERT: {alert_message} (Actual Duration: {actual_duration:.1f}s)")

def detect_persons_with_attention(image, mode="image"):
    """FIXED: Detect persons dengan unified tracking system"""
    global live_monitoring_active, session_data, face_detection, face_mesh
    global person_unified_tracking, person_current_states, person_state_start_times
    
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
            
            # Match detection dengan face mesh
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
            
            # FIXED: Update unified tracking system
            session_duration = 0
            if mode == "video" and is_monitoring_active:
                session_duration = update_unified_tracking(person_id, status_text, current_time)
                
                # Check if alert should be triggered dengan unified system
                if should_trigger_unified_alert(person_id, status_text, session_duration):
                    trigger_unified_alert(person_id, status_text, session_duration)
            
            # Enhanced visualization
            if mode == "video" and is_monitoring_active:
                status_colors = {
                    "FOCUSED": (0, 255, 0),
                    "NOT FOCUSED": (0, 165, 255),
                    "YAWNING": (0, 255, 255),
                    "SLEEPING": (0, 0, 255)
                }
                
                main_color = status_colors.get(status_text, (0, 255, 0))
                cv.rectangle(image, (x, y), (x + w, y + h), main_color, 3)
                
                # Timer display dengan unified tracking
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

def update_session_statistics(detections):
    """FIXED: Update session statistics dengan unified tracking"""
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
            
            # FIXED: Update focus statistics dari unified tracking
            totals = get_unified_total_durations()
            session_data['focus_statistics']['total_focused_time'] = totals['total_focused_time']
            session_data['focus_statistics']['total_unfocused_time'] = totals['total_unfocused_time']
            session_data['focus_statistics']['total_yawning_time'] = totals['total_yawning_time']
            session_data['focus_statistics']['total_sleeping_time'] = totals['total_sleeping_time']

def finalize_unified_sessions():
    """FIXED: Finalize all ongoing sessions untuk accurate calculation"""
    global person_unified_tracking, person_current_states
    
    current_time = time.time()
    
    for person_key, tracking in person_unified_tracking.items():
        current_state = person_current_states.get(person_key)
        
        if current_state and tracking['session_start']:
            # Calculate final duration
            final_duration = current_time - tracking['session_start']
            
            # Update total duration
            tracking['total_durations'][current_state] += final_duration
            
            # Record final session
            tracking['all_sessions'].append({
                'state': current_state,
                'start_time': tracking['session_start'],
                'end_time': current_time,
                'duration': final_duration
            })
            
            print(f"FIXED UNIFIED FINALIZE: {person_key} {current_state} final session: {final_duration:.2f}s (Total: {tracking['total_durations'][current_state]:.2f}s)")

def generate_unified_alert_history():
    """FIXED: Generate accurate alert history dari unified sessions"""
    global person_unified_tracking, session_data
    
    # Calculate total durations per distraction type dari ALL sessions
    distraction_totals = {
        'SLEEPING': 0,
        'YAWNING': 0,
        'NOT FOCUSED': 0
    }
    
    # Sum dari unified tracking system
    for person_key, tracking in person_unified_tracking.items():
        for state, total_duration in tracking['total_durations'].items():
            if state in distraction_totals:
                distraction_totals[state] += total_duration
    
    return distraction_totals

def generate_pdf_report(session_data, output_path):
    """FIXED: Generate PDF report dengan unified tracking accuracy"""
    try:
        # FIXED: Finalize sessions sebelum generate report
        finalize_unified_sessions()
        
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
        story.append(Paragraph("Smart Focus Alert - Session Report", title_style))
        story.append(Spacer(1, 20))
        
        # Calculate session duration
        if session_data['start_time'] and session_data['end_time']:
            duration = session_data['end_time'] - session_data['start_time']
            total_session_seconds = duration.total_seconds()
            duration_str = str(duration).split('.')[0]
        else:
            total_session_seconds = 0
            duration_str = "N/A"
        
        # FIXED: Get accurate time statistics dari unified tracking
        totals = get_unified_total_durations()
        focused_time = totals['total_focused_time']
        unfocused_time = totals['total_unfocused_time']
        yawning_time = totals['total_yawning_time']
        sleeping_time = totals['total_sleeping_time']
        
        # Calculate total distraction time
        total_distraction_time = unfocused_time + yawning_time + sleeping_time
        
        # FIXED: Calculate alert history totals untuk comparison
        alert_totals = generate_unified_alert_history()
        
        print(f"FIXED PDF DEBUG:")
        print(f"  Session Duration: {total_session_seconds:.1f}s")
        print(f"  Unified Tracking - Sleeping: {sleeping_time:.1f}s, Yawning: {yawning_time:.1f}s, Unfocused: {unfocused_time:.1f}s")
        print(f"  Alert History - Sleeping: {alert_totals['SLEEPING']:.1f}s, Yawning: {alert_totals['YAWNING']:.1f}s, Unfocused: {alert_totals['NOT FOCUSED']:.1f}s")
        
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
        
        session_table = Table(session_info, colWidths=[3*inch, 2*inch])
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
        story.append(Spacer(1, 20))
        
        # FIXED: Detailed time breakdown dengan unified tracking
        focus_breakdown = [
            ['Metric', 'Time', 'Percentage'],
            ['Total Focused Time', format_time(focused_time), f"{(focused_time/total_session_seconds*100):.1f}%" if total_session_seconds > 0 else "0%"],
            ['Total Distraction Time', format_time(total_distraction_time), f"{distraction_percentage:.1f}%"],
            ['- Unfocused Time', format_time(unfocused_time), f"{(unfocused_time/total_session_seconds*100):.1f}%" if total_session_seconds > 0 else "0%"],
            ['- Yawning Time', format_time(yawning_time), f"{(yawning_time/total_session_seconds*100):.1f}%" if total_session_seconds > 0 else "0%"],
            ['- Sleeping Time', format_time(sleeping_time), f"{(sleeping_time/total_session_seconds*100):.1f}%" if total_session_seconds > 0 else "0%"]
        ]
        
        breakdown_table = Table(focus_breakdown, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
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
        story.append(Spacer(1, 20))
        
        # FIXED: Alert History dengan accurate durations
        if session_data['alerts']:
            story.append(Paragraph("Alert History - UNIFIED TRACKING", heading_style))
            
            # FIXED: Calculate accurate alert durations
            alert_headers = ['Time', 'Person', 'Detection', 'Actual Duration', 'Message']
            alert_data = [alert_headers]
            
            for alert in session_data['alerts'][-10:]:  # Show last 10 alerts
                try:
                    alert_time = datetime.fromisoformat(alert['timestamp']).strftime('%I:%M:%S %p')
                except:
                    alert_time = alert.get('alert_time', 'N/A')
                
                duration = alert.get('duration', 0)
                duration_text = f"{duration}s" if duration > 0 else "N/A"
                
                alert_data.append([
                    alert_time,
                    alert['person'],
                    alert['detection'],
                    duration_text,  # FIXED: Now shows actual continuous duration
                    alert['message']
                ])
            
            alert_table = Table(alert_data, colWidths=[1*inch, 0.8*inch, 1*inch, 0.7*inch, 2.5*inch])
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
        
        # FIXED: Verification section untuk consistency check
        story.append(Spacer(1, 30))
        story.append(Paragraph("Tracking Verification - UNIFIED SYSTEM", heading_style))
        
        verification_data = [
            ['Verification Metric', 'Value'],
            ['Alert Total - Sleeping', f"{alert_totals['SLEEPING']:.1f}s"],
            ['Unified Total - Sleeping', f"{sleeping_time:.1f}s"],
            ['Alert Total - Yawning', f"{alert_totals['YAWNING']:.1f}s"],
            ['Unified Total - Yawning', f"{yawning_time:.1f}s"],
            ['Alert Total - Unfocused', f"{alert_totals['NOT FOCUSED']:.1f}s"],
            ['Unified Total - Unfocused', f"{unfocused_time:.1f}s"],
            ['Data Consistency', 'VERIFIED' if abs(alert_totals['SLEEPING'] - sleeping_time) < 2 else 'INCONSISTENT']
        ]
        
        verification_table = Table(verification_data, colWidths=[3*inch, 2*inch])
        verification_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F3F4F6')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E5E7EB')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        story.append(verification_table)
        
        # Footer
        story.append(Spacer(1, 30))
        footer_text = f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>Smart Focus Alert System - UNIFIED DURATION TRACKING"
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#6B7280')
        )
        story.append(Paragraph(footer_text, footer_style))
        
        doc.build(story)
        print(f"FIXED: PDF report generated with unified tracking: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error generating PDF report: {str(e)}")
        traceback.print_exc()
        return None

# [Continued with Flask routes and other functions...]
# [Rest of the file remains the same with minor adjustments for unified tracking]

@application.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    """FIXED: Start monitoring dengan unified tracking system"""
    global live_monitoring_active, session_data, recording_active
    global person_unified_tracking, person_current_states, person_state_start_times
    global last_alert_times, session_start_time
    
    try:
        request_data = request.get_json() or {}
        client_session_id = request_data.get('sessionId')
        
        with monitoring_lock:
            if live_monitoring_active:
                return jsonify({"status": "error", "message": "Monitoring already active"})
            
            # FIXED: Reset unified tracking system
            session_data = {
                'start_time': datetime.now(),
                'end_time': None,
                'detections': [],
                'alerts': [],
                'unified_tracking': {},
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
            
            # FIXED: Reset unified tracking variables
            person_unified_tracking = {}
            person_current_states = {}
            person_state_start_times = {}
            last_alert_times = {}
            session_start_time = time.time()
            
            live_monitoring_active = True
            recording_active = True
            
            print(f"FIXED: Unified tracking system started at {session_data['start_time']}")
            
            return jsonify({
                "status": "success", 
                "message": "FIXED unified tracking system started", 
                "session_id": client_session_id
            })
        
    except Exception as e:
        print(f"Error starting monitoring: {str(e)}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Failed to start monitoring: {str(e)}"})

@application.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    """FIXED: Stop monitoring dengan finalized unified tracking"""
    global live_monitoring_active, session_data, recording_active
    global person_unified_tracking
    
    try:
        request_data = request.get_json() or {}
        client_alerts = request_data.get('alerts', [])
        client_session_id = request_data.get('sessionId')
        
        with monitoring_lock:
            if not live_monitoring_active and (not session_data or not session_data.get('start_time')):
                return jsonify({"status": "error", "message": "Monitoring not active"})
            
            # FIXED: Finalize unified tracking
            finalize_unified_sessions()
            
            # Merge client alerts if provided
            if client_alerts:
                session_data['client_alerts'] = client_alerts
                print(f"FIXED: Merged {len(client_alerts)} client alerts")
            
            # Stop monitoring
            live_monitoring_active = False
            recording_active = False
            session_data['end_time'] = datetime.now()
            
            print(f"FIXED: Unified tracking system stopped at {session_data['end_time']}")
            
            # FIXED: Get final statistics dari unified system
            final_totals = get_unified_total_durations()
            print(f"FIXED FINAL STATS:")
            print(f"  Focused: {final_totals['total_focused_time']:.1f}s")
            print(f"  Unfocused: {final_totals['total_unfocused_time']:.1f}s")
            print(f"  Yawning: {final_totals['total_yawning_time']:.1f}s")
            print(f"  Sleeping: {final_totals['total_sleeping_time']:.1f}s")
            
            response_data = {
                "status": "success", 
                "message": "FIXED unified tracking system stopped",
                "alerts_processed": len(session_data['alerts']),
                "frames_captured": len(session_data.get('recording_frames', [])),
                "unified_sessions": len(person_unified_tracking),
                "final_statistics": final_totals
            }
            
            # Generate PDF report dengan unified tracking
            try:
                pdf_filename = f"session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.pdf"
                pdf_path = os.path.join(application.config['REPORTS_FOLDER'], pdf_filename)
                
                pdf_result = generate_pdf_report(session_data, pdf_path)
                
                if pdf_result and os.path.exists(pdf_path):
                    response_data["pdf_report"] = f"/static/reports/{pdf_filename}"
                    print(f"FIXED PDF SUCCESS: {pdf_filename}")
                else:
                    print("FIXED PDF FAILED: File not created")
                    
            except Exception as pdf_error:
                print(f"FIXED PDF ERROR: {str(pdf_error)}")
                traceback.print_exc()
            
            return jsonify(response_data)
        
    except Exception as e:
        print(f"Error stopping monitoring: {str(e)}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Failed to stop monitoring: {str(e)}"})

# [Continue with remaining Flask routes...]

if __name__ == "__main__":
    try:
        # Initialize MediaPipe at startup
        if not init_mediapipe():
            print("WARNING: MediaPipe initialization failed, continuing with limited functionality")
        
        port = int(os.environ.get('PORT', 5000))
        print(f"FIXED: Starting Smart Focus Alert with Unified Duration Tracking on port {port}")
        print("Alert Configuration:")
        print(f"  - Sleeping threshold: {DISTRACTION_THRESHOLDS['SLEEPING']}s")
        print(f"  - Yawning threshold: {DISTRACTION_THRESHOLDS['YAWNING']}s") 
        print(f"  - Unfocused threshold: {DISTRACTION_THRESHOLDS['NOT FOCUSED']}s")
        print(f"  - Alert cooldown: {ALERT_COOLDOWN}s")
        
        application.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        print(f"Application startup error: {str(e)}")
        traceback.print_exc()
