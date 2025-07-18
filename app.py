# app.py - Railway deployment dengan FIXED video recording consistency
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

# FIXED: Enhanced global variables untuk consistent video recording
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
    'frame_counter': 0,  # FIXED: Added consistent frame counter
    'frame_timestamps': [],  # FIXED: Track frame timing
    'total_frames_processed': 0  # FIXED: Track total processing
}

# Video recording variables
video_writer = None
recording_active = False

# Enhanced person tracking untuk Railway
person_state_timers = {}
person_current_state = {}
last_alert_time = {}

# Alert thresholds (dalam seconds) - gunakan yang sama dengan lokal
DISTRACTION_THRESHOLDS = {
    'SLEEPING': 10,
    'YAWNING': 3.5,
    'NOT FOCUSED': 10
}

# FIXED: Frame recording configuration
FRAME_STORAGE_INTERVAL = 2  # Store every 2nd frame instead of 5th
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

def draw_landmarks(image, landmarks, land_mark, color):
    """Draw landmarks on the image for a single face"""
    height, width = image.shape[:2]
    for face in land_mark:
        point = landmarks.landmark[face]
        point_scale = ((int)(point.x * width), (int)(point.y * height))     
        cv.circle(image, point_scale, 2, color, 1)

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

def detect_drowsiness(frame, landmarks, speech_engine=None):
    """Detect drowsiness and attention state based on eye aspect ratio and other metrics"""
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

    try:
        # Draw landmarks
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

        # Draw iris circles dengan error handling
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

def detect_persons_with_attention(image, mode="image"):
    """Detect persons in image atau video frame dengan attention status - Railway optimized"""
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
            
            # Analyze attention jika face mesh matched
            if matched_face_idx != -1:
                attention_status, state = detect_drowsiness(
                    image, 
                    mesh_results.multi_face_landmarks[matched_face_idx],
                    None
                )
            
            status_text = attention_status.get("state", "FOCUSED")
            person_key = f"person_{i+1}"
            
            duration = 0
            if mode == "video" and is_monitoring_active:
                with monitoring_lock:
                    # Initialize person tracking
                    if person_key not in person_state_timers:
                        person_state_timers[person_key] = {}
                        person_current_state[person_key] = None
                        last_alert_time[person_key] = 0
                    
                    # Update state timing
                    if person_current_state[person_key] != status_text:
                        # State changed - reset timers
                        person_state_timers[person_key] = {}
                        person_current_state[person_key] = status_text
                        person_state_timers[person_key][status_text] = current_time
                    else:
                        # Same state continues
                        if status_text not in person_state_timers[person_key]:
                            person_state_timers[person_key][status_text] = current_time
                    
                    # Calculate duration
                    if status_text in person_state_timers[person_key]:
                        duration = current_time - person_state_timers[person_key][status_text]
            
            # Enhanced visualization untuk monitoring
            if mode == "video" and is_monitoring_active:
                status_colors = {
                    "FOCUSED": (0, 255, 0),
                    "NOT FOCUSED": (0, 165, 255),
                    "YAWNING": (0, 255, 255),
                    "SLEEPING": (0, 0, 255)
                }
                
                main_color = status_colors.get(status_text, (0, 255, 0))
                cv.rectangle(image, (x, y), (x + w, y + h), main_color, 3)
                
                # Timer display
                if status_text in DISTRACTION_THRESHOLDS:
                    threshold = DISTRACTION_THRESHOLDS[status_text]
                    timer_text = f"Person {i+1}: {status_text} ({duration:.1f}s/{threshold}s)"
                else:
                    timer_text = f"Person {i+1}: {status_text}"
                
                # Draw text background
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
                
                # Information overlay - gunakan layout file lokal
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

            # Alert handling dengan timing yang benar
            should_alert = False
            alert_message = ""
            
            if (mode == "video" and is_monitoring_active and status_text in DISTRACTION_THRESHOLDS and 
                person_key in person_state_timers and status_text in person_state_timers[person_key]):
                
                if duration >= DISTRACTION_THRESHOLDS[status_text]:
                    alert_cooldown = 5  # 5 second cooldown
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
                                    'duration': int(duration)
                                })
                                print(f"Alert added: {alert_message} (Total alerts: {len(session_data['alerts'])})")
            
            # Save detected face - Railway optimized path
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
                "duration": duration if mode == "video" else 0
            })
    
    # Display detection count
    if detections:
        cv.putText(image, f"Total persons detected: {len(detections)}", 
                  (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv.putText(image, "No persons detected", 
                  (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return image, detections

def calculate_distraction_time_from_alerts(alerts):
    """Calculate actual distraction time based on alert history - FIXED VERSION dari file lokal"""
    distraction_times = {
        'unfocused_time': 0,
        'yawning_time': 0,
        'sleeping_time': 0
    }
    
    if not alerts:
        return distraction_times
    
    # Group alerts by person and distraction type untuk proper accumulation
    person_distractions = {}
    
    for alert in alerts:
        person = alert.get('person', 'Unknown')
        detection = alert.get('detection', 'Unknown')
        duration = alert.get('duration', 0)
        
        if person not in person_distractions:
            person_distractions[person] = {}
        
        if detection not in person_distractions[person]:
            person_distractions[person][detection] = []
        
        person_distractions[person][detection].append(duration)
    
    # Calculate total time untuk setiap distraction type by SUMMING all durations
    for person, distractions in person_distractions.items():
        for detection_type, durations in distractions.items():
            if detection_type == 'NOT FOCUSED':
                # Sum all unfocused durations untuk person ini
                distraction_times['unfocused_time'] += sum(durations)
            elif detection_type == 'YAWNING':
                # Sum all yawning durations untuk person ini
                distraction_times['yawning_time'] += sum(durations)
            elif detection_type == 'SLEEPING':
                # Sum all sleeping durations untuk person ini
                distraction_times['sleeping_time'] += sum(durations)
    
    return distraction_times

def update_session_statistics(detections):
    """Update session statistics based on current detections - dengan fixed calculation"""
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
            
            # Update distraction times berdasarkan actual alert history - FIXED
            distraction_times = calculate_distraction_time_from_alerts(session_data['alerts'])
            session_data['focus_statistics']['unfocused_time'] = distraction_times['unfocused_time']
            session_data['focus_statistics']['yawning_time'] = distraction_times['yawning_time']
            session_data['focus_statistics']['sleeping_time'] = distraction_times['sleeping_time']

def get_most_common_distraction(alerts):
    """Helper function untuk find the most common type of distraction dengan total duration"""
    if not alerts:
        return "None"
    
    distraction_counts = {}
    distraction_durations = {}
    
    for alert in alerts:
        detection = alert.get('detection', 'Unknown')
        duration = alert.get('duration', 0)
        
        # Count occurrences
        distraction_counts[detection] = distraction_counts.get(detection, 0) + 1
        
        # Sum durations
        distraction_durations[detection] = distraction_durations.get(detection, 0) + duration
    
    if not distraction_counts:
        return "None"
    
    # Find most common by count
    most_common = max(distraction_counts, key=distraction_counts.get)
    count = distraction_counts[most_common]
    total_duration = distraction_durations[most_common]
    
    return f"{most_common} ({count} times, {total_duration}s total)"

def calculate_average_focus_metric(focused_time, total_session_seconds):
    """Calculate meaningful average focus metric berdasarkan session duration"""
    if total_session_seconds <= 0:
        return "N/A"
    
    # Convert ke minutes untuk easier reading
    total_minutes = total_session_seconds / 60
    focused_minutes = focused_time / 60
    
    # Different metrics berdasarkan session duration
    if total_session_seconds < 60:  # Kurang dari 1 minute
        # Show focus percentage of session time
        focus_percentage = (focused_time / total_session_seconds) * 100
        return f"{focus_percentage:.1f}% of session time"
    
    elif total_session_seconds < 3600:  # Kurang dari 1 hour
        # Show focused minutes per session
        return f"{focused_minutes:.1f} min focused out of {total_minutes:.1f} min total"
    
    else:  # 1 hour atau lebih
        # Show focused minutes per hour (extrapolated)
        hours = total_session_seconds / 3600
        focused_per_hour = focused_minutes / hours
        return f"{focused_per_hour:.1f} min focused per hour"

# FIXED: Enhanced video creation dengan consistent frame rate and duration
def create_session_recording_from_frames(recording_frames, output_path, session_start_time, session_end_time):
    """FIXED: Create video recording dengan proper frame timing dan duration"""
    try:
        if not recording_frames:
            print("FIXED: No frames to create video")
            return None

        actual_duration = session_end_time - session_start_time
        actual_duration_seconds = actual_duration.total_seconds()
        
        if actual_duration_seconds <= 0:
            print("FIXED: Invalid session duration")
            return None

        # FIXED: Use consistent FPS instead of calculated
        fps = RECORDING_FPS  # Use constant FPS for predictable results
        print(f"FIXED: Using FPS: {fps} for duration {actual_duration_seconds:.2f}s with {len(recording_frames)} frames")

        # FIXED: Calculate how many times each frame should be repeated
        total_frames_needed = int(fps * actual_duration_seconds)
        if total_frames_needed <= 0:
            total_frames_needed = len(recording_frames) * 5  # Fallback
        
        frame_repeat_count = max(1, total_frames_needed // len(recording_frames))
        print(f"FIXED: Target {total_frames_needed} frames, repeating each frame {frame_repeat_count} times")

        height, width = recording_frames[0].shape[:2]
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            print(f"FIXED ERROR: Could not open video writer for {output_path}")
            return None

        # FIXED: Write frames dengan proper repetition
        frames_written = 0
        for i, frame in enumerate(recording_frames):
            if frame is not None and frame.size > 0:
                # Write each frame multiple times untuk maintain duration
                for repeat in range(frame_repeat_count):
                    out.write(frame)
                    frames_written += 1
                    
                print(f"FIXED: Processed frame {i+1}/{len(recording_frames)}, written {frames_written} video frames")
            else:
                print(f"FIXED WARNING: Skipping invalid frame {i+1}")

        out.release()
        
        print(f"FIXED: Total frames written: {frames_written}")

        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:  # Minimal size check
            file_size = os.path.getsize(output_path)
            print(f"FIXED SUCCESS: Recording created: {output_path} (size: {file_size} bytes)")
            
            # FIXED: Verify video duration using ffprobe if available
            try:
                import subprocess
                result = subprocess.run(['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', 
                                       '-of', 'csv=p=0', output_path], 
                                     capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    video_duration = float(result.stdout.strip())
                    print(f"FIXED: Verified video duration: {video_duration:.2f}s (expected: {actual_duration_seconds:.2f}s)")
            except:
                print("FIXED: ffprobe not available for duration verification")
            
            return output_path
        else:
            print("FIXED ERROR: Failed to create session recording - file not created atau too small")
            return None

    except Exception as e:
        print(f"FIXED ERROR: Exception creating session recording: {str(e)}")
        traceback.print_exc()
        return None

def generate_pdf_report(session_data, output_path):
    """Generate PDF report untuk session dengan corrected focus accuracy calculation"""
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
        story.append(Paragraph("Smart Focus Alert - Session Report", title_style))
        story.append(Spacer(1, 20))
        
        # Calculate session duration dan focus accuracy
        if session_data['start_time'] and session_data['end_time']:
            duration = session_data['end_time'] - session_data['start_time']
            total_session_seconds = duration.total_seconds()
            duration_str = str(duration).split('.')[0]  # Remove microseconds
        else:
            total_session_seconds = 0
            duration_str = "N/A"
        
        # Get corrected time statistics dari alert history
        distraction_times = calculate_distraction_time_from_alerts(session_data['alerts'])
        unfocused_time = distraction_times['unfocused_time']
        yawning_time = distraction_times['yawning_time']
        sleeping_time = distraction_times['sleeping_time']
        
        # Calculate total distraction time
        total_distraction_time = unfocused_time + yawning_time + sleeping_time
        
        # Calculate focused time (session time minus distraction time)
        if total_session_seconds > 0:
            focused_time = max(0, total_session_seconds - total_distraction_time)
        else:
            focused_time = 0
        
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
            ['Frames Recorded', str(len(session_data.get('recording_frames', [])))]  # FIXED: Added frame count
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
        
        # Detailed time breakdown
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
            # Highlight focused time row
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#ECFDF5')),
            ('TEXTCOLOR', (0, 1), (-1, 1), colors.HexColor('#065F46')),
            # Highlight total distraction row
            ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#FEF2F2')),
            ('TEXTCOLOR', (0, 2), (-1, 2), colors.HexColor('#991B1B')),
        ]))
        
        story.append(breakdown_table)
        story.append(Spacer(1, 20))
        
        # Focus Statistics - FIXED AVERAGE CALCULATION
        story.append(Paragraph("Detailed Focus Statistics", heading_style))
        
        # Calculate corrected average focus metric
        average_focus_metric = calculate_average_focus_metric(focused_time, total_session_seconds)
        
        focus_stats = [
            ['Total Session Duration', format_time(total_session_seconds)],
            ['Focus Accuracy Score', f"{focus_accuracy:.2f}%"],
            ['Focus Quality Rating', focus_rating],
            ['Average Focus Metric', average_focus_metric],  # FIXED: More meaningful metric
            ['Distraction Frequency', f"{len(session_data['alerts'])} alerts in {format_time(total_session_seconds)}"],
            ['Most Common Distraction', get_most_common_distraction(session_data['alerts'])],
            ['Recording Quality', f"{len(session_data.get('recording_frames', []))} frames captured"]  # FIXED: Added recording info
        ]
        
        focus_table = Table(focus_stats, colWidths=[3*inch, 2*inch])
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
        story.append(Spacer(1, 30))
        
        # Alert History
        if session_data['alerts']:
            story.append(Paragraph("Alert History", heading_style))
            
            alert_headers = ['Time', 'Person', 'Detection', 'Duration', 'Message']
            alert_data = [alert_headers]
            
            for alert in session_data['alerts'][-10:]:  # Show last 10 alerts
                try:
                    alert_time = datetime.fromisoformat(alert['timestamp']).strftime('%I:%M:%S %p')
                except:
                    alert_time = alert['timestamp']
                
                duration = alert.get('duration', 0)
                duration_text = f"{duration}s" if duration > 0 else "N/A"
                
                alert_data.append([
                    alert_time,
                    alert['person'],
                    alert['detection'],
                    duration_text,
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
        
        # Footer
        story.append(Spacer(1, 30))
        footer_text = f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>Smart Focus Alert System - FIXED Railway Deployment"
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#6B7280')
        )
        story.append(Paragraph(footer_text, footer_style))
        
        doc.build(story)
        print(f"PDF report generated successfully: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error generating PDF report: {str(e)}")
        traceback.print_exc()
        return None

def generate_upload_pdf_report(detections, file_info, output_path):
    """Generate PDF report untuk uploaded file analysis"""
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
        
        file_table = Table(file_info_data, colWidths=[3*inch, 2*inch])
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
        
        analysis_table = Table(analysis_stats, colWidths=[3*inch, 2*inch])
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
            
            detection_table = Table(detection_data, colWidths=[1*inch, 1.5*inch, 1*inch, 1.2*inch, 1.3*inch])
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
        footer_text = f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>Smart Focus Alert System - FIXED Railway Deployment"
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
    except Exception as e:
        print(f"Error generating upload PDF report: {str(e)}")
        traceback.print_exc()
        return None

def process_video_file(video_path):
    """Process video file dan detect persons in each frame"""
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
    process_every_n_frames = 5  # Process setiap 10 frame untuk Railway optimization
    
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

@application.route('/webcam')
def webcam():
    return render_template('webcam.html')

@application.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    global live_monitoring_active, session_data, recording_active, person_state_timers, person_current_state, last_alert_time
    
    try:
        request_data = request.get_json() or {}
        client_session_id = request_data.get('sessionId')
        
        with monitoring_lock:
            print(f"=== FIXED START MONITORING REQUEST ===")
            print(f"Current status: live_monitoring_active={live_monitoring_active}")
            print(f"Client session ID: {client_session_id}")
            
            if live_monitoring_active:
                print("WARNING: Monitoring already active, returning error")
                return jsonify({"status": "error", "message": "Monitoring already active"})
            
            # FIXED: Reset session data completely dengan enhanced frame tracking
            session_data = {
                'start_time': datetime.now(),
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
                'session_id': client_session_id,
                'client_alerts': [],
                'frame_counter': 0,  # FIXED: Consistent frame tracking
                'frame_timestamps': [],  # FIXED: Track timing
                'total_frames_processed': 0  # FIXED: Total processing count
            }
            
            person_state_timers = {}
            person_current_state = {}
            last_alert_time = {}
            
            live_monitoring_active = True
            recording_active = True
            
            print(f"FIXED: Railway monitoring started at {session_data['start_time']}")
            print(f"Status: live_monitoring_active={live_monitoring_active}")
            print(f"Session ID: {client_session_id}")
            print(f"=== FIXED START MONITORING SUCCESS ===")
            
            return jsonify({"status": "success", "message": "FIXED Railway monitoring started", "session_id": client_session_id})
        
    except Exception as e:
        print(f"Error starting monitoring: {str(e)}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Failed to start monitoring: {str(e)}"})

@application.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    global live_monitoring_active, session_data, recording_active
    
    try:
        request_data = request.get_json() or {}
        client_alerts = request_data.get('alerts', [])
        client_session_id = request_data.get('sessionId')
        total_client_alerts = request_data.get('totalAlerts', 0)
        
        with monitoring_lock:
            print(f"=== FIXED STOP MONITORING REQUEST ===")
            print(f"Current status: live_monitoring_active={live_monitoring_active}")
            print(f"Client session ID: {client_session_id}")
            print(f"Client alerts count: {len(client_alerts)}")
            print(f"Total client alerts: {total_client_alerts}")
            
            if session_data:
                print(f"Server session ID: {session_data.get('session_id')}")
                print(f"Session start time: {session_data.get('start_time')}")
                print(f"Server alerts: {len(session_data.get('alerts', []))}")
                print(f"FIXED: Total frames captured: {len(session_data.get('recording_frames', []))}")
                print(f"FIXED: Frame counter: {session_data.get('frame_counter', 0)}")
                print(f"FIXED: Total frames processed: {session_data.get('total_frames_processed', 0)}")
            
            # Enhanced session validation
            if not live_monitoring_active and (not session_data or not session_data.get('start_time')):
                print("ERROR: No active monitoring session found")
                return jsonify({"status": "error", "message": "Monitoring not active"})
            
            # Ensure session_data exists dan merge client alerts
            if not session_data:
                print("WARNING: Creating minimal session data")
                session_data = {
                    'start_time': datetime.now() - timedelta(minutes=1),
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
                    'session_id': client_session_id,
                    'client_alerts': [],
                    'frame_counter': 0,
                    'frame_timestamps': [],
                    'total_frames_processed': 0
                }
            
            if not session_data.get('start_time'):
                session_data['start_time'] = datetime.now() - timedelta(minutes=1)
            
            # Merge client alerts dengan server alerts
            if client_alerts:
                print(f"FIXED: Merging {len(client_alerts)} client alerts with server data")
                session_data['client_alerts'] = client_alerts
                
                # Convert client alerts ke server format dan merge
                for client_alert in client_alerts:
                    server_alert = {
                        'timestamp': client_alert.get('time', datetime.now().isoformat()),
                        'person': client_alert.get('person', 'Unknown'),
                        'detection': client_alert.get('detection', 'Unknown'),
                        'message': client_alert.get('message', ''),
                        'duration': client_alert.get('duration', 0)
                    }
                    session_data['alerts'].append(server_alert)
                
                print(f"FIXED: Total alerts after merge: {len(session_data['alerts'])}")
            
            # Stop monitoring
            live_monitoring_active = False
            recording_active = False
            session_data['end_time'] = datetime.now()
            
            print(f"FIXED: Monitoring stopped at {session_data['end_time']}")
            
            response_data = {
                "status": "success", 
                "message": "FIXED Railway monitoring stopped",
                "alerts_processed": len(session_data['alerts']),
                "frames_captured": len(session_data.get('recording_frames', []))  # FIXED: Added frame info
            }
            
            # Generate PDF report
            print("=== FIXED GENERATING PDF REPORT ===")
            try:
                pdf_filename = f"session_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.pdf"
                pdf_path = os.path.join(application.config['REPORTS_FOLDER'], pdf_filename)
                
                print(f"FIXED: PDF path: {pdf_path}")
                print(f"FIXED: Session data alerts: {len(session_data['alerts'])}")
                print(f"FIXED: Session frames for report: {len(session_data.get('recording_frames', []))}")
                
                pdf_result = generate_pdf_report(session_data, pdf_path)
                
                if pdf_result and os.path.exists(pdf_path):
                    response_data["pdf_report"] = f"/static/reports/{pdf_filename}"
                    print(f"FIXED PDF SUCCESS: {pdf_filename} (size: {os.path.getsize(pdf_path)} bytes)")
                else:
                    print("FIXED PDF FAILED: File not created")
                    
            except Exception as pdf_error:
                print(f"FIXED PDF ERROR: {str(pdf_error)}")
                traceback.print_exc()
            
            # FIXED: Generate video recording dengan enhanced timing
            print("=== FIXED GENERATING VIDEO RECORDING ===")
            try:
                recording_filename = f"session_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.mp4"
                recording_path = os.path.join(application.config['RECORDINGS_FOLDER'], recording_filename)
                
                print(f"FIXED: Video path: {recording_path}")
                frame_count = len(session_data.get('recording_frames', []))
                print(f"FIXED: Available frames: {frame_count}")
                print(f"FIXED: Session duration: {(session_data.get('end_time', datetime.now()) - session_data.get('start_time', datetime.now())).total_seconds():.2f}s")
                
                if frame_count > 0:
                    print(f"FIXED: Creating video from {frame_count} recorded frames with enhanced timing")
                    video_result = create_session_recording_from_frames(
                        session_data['recording_frames'],
                        recording_path,
                        session_data.get('start_time', datetime.now() - timedelta(seconds=10)),
                        session_data.get('end_time', datetime.now())
                    )
                    
                    if video_result and os.path.exists(recording_path):
                        response_data["video_file"] = f"/static/recordings/{os.path.basename(recording_path)}"
                        session_data['recording_path'] = recording_path
                        file_size = os.path.getsize(recording_path)
                        print(f"FIXED VIDEO SUCCESS: {os.path.basename(recording_path)} (size: {file_size} bytes)")
                    else:
                        print("FIXED VIDEO FAILED: Unable to create from frames")
                else:
                    print("FIXED VIDEO SKIPPED: No frames available - check frame storage logic")
                    
            except Exception as video_error:
                print(f"FIXED VIDEO ERROR: {str(video_error)}")
                traceback.print_exc()
            
            print(f"=== FIXED STOP MONITORING COMPLETE ===")
            print(f"Response: {response_data}")
            return jsonify(response_data)
        
    except Exception as e:
        print(f"FIXED FATAL ERROR stopping monitoring: {str(e)}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Failed to stop monitoring: {str(e)}"})

@application.route('/process_frame', methods=['POST'])
def process_frame():
    """FIXED: Enhanced frame processing dengan consistent storage dan timing"""
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
        
        # Process frame untuk detection dengan enhanced analysis
        processed_frame, detections = detect_persons_with_attention(frame, mode="video")
        
        # FIXED: Enhanced frame storage dengan consistent logic
        with monitoring_lock:
            if live_monitoring_active and recording_active and session_data:
                # FIXED: Use consistent frame counter instead of array length
                session_data['frame_counter'] = session_data.get('frame_counter', 0) + 1
                session_data['total_frames_processed'] = session_data.get('total_frames_processed', 0) + 1
                current_timestamp = time.time()
                
                # FIXED: Store frames more frequently dengan better logic
                should_store_frame = (
                    session_data['frame_counter'] % FRAME_STORAGE_INTERVAL == 0 or  # Every Nth frame
                    len(detections) > 0 or  # Always store frames with detections
                    len(session_data.get('recording_frames', [])) < 10  # Always store first 10 frames
                )
                
                if should_store_frame:
                    # Create a copy to avoid reference issues
                    frame_copy = processed_frame.copy()
                    session_data['recording_frames'].append(frame_copy)
                    session_data['frame_timestamps'].append(current_timestamp)
                    
                    print(f"FIXED: Frame {session_data['frame_counter']} stored (total stored: {len(session_data['recording_frames'])})")
                    
                    # FIXED: Better memory management - keep more frames for better video quality
                    if len(session_data['recording_frames']) > MAX_STORED_FRAMES:
                        # Remove oldest frames but keep reasonable amount
                        frames_to_remove = len(session_data['recording_frames']) - MAX_STORED_FRAMES
                        session_data['recording_frames'] = session_data['recording_frames'][frames_to_remove:]
                        session_data['frame_timestamps'] = session_data['frame_timestamps'][frames_to_remove:]
                        print(f"FIXED: Memory management - removed {frames_to_remove} oldest frames")
                
                # FIXED: Debug logging every 10th frame
                if session_data['frame_counter'] % 10 == 0:
                    print(f"FIXED FRAME STORAGE STATUS:")
                    print(f"  - Frames processed: {session_data['total_frames_processed']}")
                    print(f"  - Frames stored: {len(session_data.get('recording_frames', []))}")
                    print(f"  - Current detections: {len(detections)}")
                    print(f"  - Storage ratio: {len(session_data.get('recording_frames', [])) / max(1, session_data['total_frames_processed']) * 100:.1f}%")
        
        # Update session statistics jika monitoring active
        if live_monitoring_active and detections:
            update_session_statistics(detections)
        
        # Encode processed frame back to base64 dengan quality optimization untuk Railway
        _, buffer = cv.imencode('.jpg', processed_frame, [cv.IMWRITE_JPEG_QUALITY, 85])  # Slightly higher quality
        processed_frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "success": True,
            "processed_frame": f"data:image/jpeg;base64,{processed_frame_b64}",
            "detections": detections,
            "frame_count": len(session_data.get('recording_frames', [])) if session_data else 0,
            "total_processed": session_data.get('total_frames_processed', 0) if session_data else 0,  # FIXED: Added processing count
            "frame_number": session_data.get('frame_counter', 0) if session_data else 0  # FIXED: Added frame number
        })
        
    except Exception as e:
        print(f"FIXED: Error processing frame: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Frame processing failed: {str(e)}"}), 500

# Additional utility routes
@application.route('/sync_alerts', methods=['POST'])
def sync_alerts():
    """Sync client-side alerts dengan server"""
    try:
        request_data = request.get_json() or {}
        client_alerts = request_data.get('alerts', [])
        session_id = request_data.get('sessionId')
        
        with monitoring_lock:
            if session_data and session_data.get('session_id') == session_id:
                session_data['client_alerts'] = client_alerts
                print(f"FIXED: Synced {len(client_alerts)} client alerts for session {session_id}")
                return jsonify({"status": "success", "synced_count": len(client_alerts)})
            else:
                return jsonify({"status": "error", "message": "Session mismatch"})
                
    except Exception as e:
        print(f"FIXED: Alert sync error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

@application.route('/get_monitoring_data')
def get_monitoring_data():
    """Enhanced monitoring data endpoint"""
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
                    alert_time = alert['timestamp']
                
                formatted_alerts.append({
                    'time': alert_time,
                    'message': alert['message'],
                    'type': 'warning' if alert['detection'] in ['YAWNING', 'NOT FOCUSED'] else 'error'
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
                'total_processed': session_data.get('total_frames_processed', 0) if session_data else 0  # FIXED: Added total processed
            })
        
    except Exception as e:
        print(f"FIXED: Error getting monitoring data: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Failed to get monitoring data: {str(e)}"})

@application.route('/monitoring_status')
def monitoring_status():
    """Get current monitoring status dengan enhanced info"""
    try:
        with monitoring_lock:
            return jsonify({
                "is_active": live_monitoring_active,
                "session_id": session_data.get('session_id') if session_data else None,
                "alerts_count": len(session_data.get('alerts', [])) if session_data else 0,
                "frames_stored": len(session_data.get('recording_frames', [])) if session_data else 0,  # FIXED: Added frame info
                "frames_processed": session_data.get('total_frames_processed', 0) if session_data else 0  # FIXED: Added processing info
            })
    except Exception as e:
        print(f"FIXED: Error getting monitoring status: {str(e)}")
        return jsonify({"is_active": False})

@application.route('/check_camera')
def check_camera():
    """Check camera availability - Railway optimized"""
    try:
        return jsonify({"camera_available": False})  # Force client-side camera untuk Railway
    except Exception as e:
        print(f"FIXED: Error checking camera: {str(e)}")
        return jsonify({"camera_available": False})

@application.route('/health')
def health_check():
    """Enhanced health check endpoint untuk Railway dengan frame info"""
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
                "total_frames_processed": session_data.get('total_frames_processed', 0) if session_data else 0,  # FIXED: Added processing count
                "frame_storage_ratio": len(session_data.get('recording_frames', [])) / max(1, session_data.get('total_frames_processed', 1)) * 100 if session_data else 0,  # FIXED: Storage ratio
                "mediapipe_status": "initialized" if face_detection and face_mesh else "error",
                "version": "railway_optimized_FIXED_v2.0"
            })
    except Exception as e:
        print(f"FIXED: Health check error: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@application.route('/api/detect', methods=['POST'])
def api_detect():
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

# Static file routes untuk Railway
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
        print(f"FIXED: Error serving report file: {str(e)}")
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
        print(f"FIXED: Error serving recording file: {str(e)}")
        return jsonify({"error": "Error accessing recording file"}), 500

if __name__ == "__main__":
    try:
        # Initialize MediaPipe at startup
        if not init_mediapipe():
            print("WARNING: MediaPipe initialization failed, continuing with limited functionality")
        
        port = int(os.environ.get('PORT', 5000))
        print(f"FIXED: Starting Railway Optimized Smart Focus Alert application on port {port}")
        print("FIXED: Frame storage configuration:")
        print(f"  - Storage interval: every {FRAME_STORAGE_INTERVAL} frames")
        print(f"  - Max stored frames: {MAX_STORED_FRAMES}")
        print(f"  - Recording FPS: {RECORDING_FPS}")
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
