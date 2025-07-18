# app.py - FIXED Complete Smart Focus Alert System
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

# FIXED: Enhanced global variables for synchronized tracking
monitoring_lock = threading.RLock()
live_monitoring_active = False

# FIXED: Synchronized session data structure
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

# FIXED: Synchronized person tracking with accurate duration calculation
person_distraction_sessions = {}  # Track actual continuous distraction durations
person_current_states = {}       # Current state of each person
person_state_start_times = {}    # When current state started (timestamp)
last_alert_times = {}            # Last time alert was triggered for each person
session_start_time = None        # Global session start time

# FIXED: Alert thresholds and improved cooldown system
DISTRACTION_THRESHOLDS = {
    'SLEEPING': 10,      # 10 seconds
    'YAWNING': 3.5,      # 3.5 seconds  
    'NOT FOCUSED': 10    # 10 seconds
}

# FIXED: Reduced cooldown for better user experience (2 seconds as requested)
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

def detect_persons_with_attention(image, mode="image"):
    """FIXED: Detect persons with synchronized session tracking"""
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
            
            # Enhanced visualization
            status_colors = {
                "FOCUSED": (0, 255, 0),
                "NOT FOCUSED": (0, 165, 255),
                "YAWNING": (0, 255, 255),
                "SLEEPING": (0, 0, 255)
            }
            
            main_color = status_colors.get(status_text, (0, 255, 0))
            cv.rectangle(image, (x, y), (x + w, y + h), main_color, 3)
            
            # Status text
            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            status_display = f"Person {person_id}: {status_text}"
            
            (text_width, text_height), baseline = cv.getTextSize(status_display, font, font_scale, thickness)
            
            text_y = y - 10
            if text_y < text_height + 10:
                text_y = y + h + text_height + 10
            
            # Semi-transparent background
            overlay = image.copy()
            cv.rectangle(overlay, (x, text_y - text_height - 5), (x + text_width + 10, text_y + 5), (0, 0, 0), -1)
            cv.addWeighted(overlay, 0.7, image, 0.3, 0, image)
            
            cv.putText(image, status_display, (x + 5, text_y), font, font_scale, main_color, thickness)

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
                "duration": 0  # For upload mode, duration is 0
            })
    
    # Display detection count
    if detections:
        cv.putText(image, f"Total persons detected: {len(detections)}", 
                  (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv.putText(image, "No persons detected", 
                  (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return image, detections

def generate_upload_pdf_report(detections, file_info, output_path):
    """Generate PDF report for uploaded file analysis"""
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
        return output_path
    except Exception as e:
        print(f"Error generating upload PDF report: {str(e)}")
        traceback.print_exc()
        return None

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
            try:
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
                    print(f"Processing image: {filename}")
                    image = cv.imread(file_path)
                    if image is None:
                        return render_template('upload.html', error='Invalid image file or corrupted')
                    
                    processed_image, detections = detect_persons_with_attention(image)
                    
                    # Save processed image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_filename = f"processed_{timestamp}_{uuid.uuid4().hex[:8]}_{filename}"
                    output_path = os.path.join(application.config['DETECTED_FOLDER'], output_filename)
                    cv.imwrite(output_path, processed_image)
                    
                    result["processed_image"] = f"/static/detected/{output_filename}"
                    result["detections"] = detections
                    result["type"] = "image"
                    
                    print(f"Image processed successfully. Found {len(detections)} persons")
                    
                elif file_ext in ['mp4', 'avi', 'mov', 'mkv']:
                    # Process video
                    print(f"Processing video: {filename}")
                    output_path, detections = process_video_file(file_path)
                    
                    result["processed_video"] = f"/static/detected/{os.path.basename(output_path)}"
                    result["detections"] = detections
                    result["type"] = "video"
                    
                    print(f"Video processed successfully. Found {len(detections)} total detections")
                else:
                    return render_template('upload.html', error='Unsupported file format')
                
                # Generate PDF report
                pdf_filename = f"report_{filename.split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                pdf_path = os.path.join(application.config['REPORTS_FOLDER'], pdf_filename)
                
                file_info = {
                    'filename': filename,
                    'type': file_ext.upper()
                }
                
                pdf_result = generate_upload_pdf_report(detections, file_info, pdf_path)
                if pdf_result:
                    result["pdf_report"] = f"/static/reports/{pdf_filename}"
                    print(f"PDF report generated: {pdf_filename}")
                
                print(f"Rendering result template with {len(detections)} detections")
                return render_template('result.html', result=result)
                
            except Exception as e:
                print(f"Error processing upload: {str(e)}")
                traceback.print_exc()
                return render_template('upload.html', error=f'Error processing file: {str(e)}')
    
    return render_template('upload.html')

@application.route('/webcam')
def webcam():
    return render_template('webcam.html')

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

# Live monitoring routes (simplified for this example)
@application.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    return jsonify({"status": "success", "message": "Live monitoring endpoints available"})

@application.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    return jsonify({"status": "success", "message": "Live monitoring endpoints available"})

@application.route('/process_frame', methods=['POST'])
def process_frame():
    return jsonify({"success": False, "message": "Live monitoring endpoints available"})

@application.route('/get_monitoring_data')
def get_monitoring_data():
    return jsonify({"error": "Live monitoring not active"})

@application.route('/monitoring_status')
def monitoring_status():
    return jsonify({"is_active": False})

@application.route('/check_camera')
def check_camera():
    return jsonify({"camera_available": False})

@application.route('/health')
def health_check():
    try:
        return jsonify({
            "status": "healthy", 
            "timestamp": datetime.now().isoformat(),
            "directories": {
                "uploads": os.path.exists(application.config['UPLOAD_FOLDER']),
                "detected": os.path.exists(application.config['DETECTED_FOLDER']),
                "reports": os.path.exists(application.config['REPORTS_FOLDER']),
                "recordings": os.path.exists(application.config['RECORDINGS_FOLDER'])
            },
            "mediapipe_status": "initialized" if face_detection and face_mesh else "error",
            "version": "upload_analysis_v1.0"
        })
    except Exception as e:
        print(f"Health check error: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

if __name__ == "__main__":
    try:
        # Initialize MediaPipe at startup
        if not init_mediapipe():
            print("WARNING: MediaPipe initialization failed, continuing with limited functionality")
        
        port = int(os.environ.get('PORT', 5000))
        print(f"Starting Smart Focus Alert System on port {port}")
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
