import numpy as np
import cv2
from pupil_apriltags import Detector
import socket  # Import socket module
import time, struct
from flask import Flask, render_template, Response # type: ignore
from concurrent.futures import ThreadPoolExecutor

import base64
import threading
from numba import njit

last_detection_time = time.time()
raw_frame = None
sending_frame = False

# Define the target IP and port
IP_ADDRESS = "10.102.52.2"  # Roborio ip
PORT = 1234

# Create a persistent UDP socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Load camera calibration parameters
camera_matrix = np.load("camera_matrix.npy")
dist_coeffs = np.load("dist_coeffs.npy")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # See templates folder, this is the video client

@njit(cache=True, fastmath=True) # JIT compile this heavy math function, cache for faster compilation next run, fastmath for less precise, but faster, which is ok in our case
def extract_euler_angles(R):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0

    return roll, pitch, yaw

@njit(cache=True, fastmath=True)
def find_closest_tag_index(tvecs):
    min_dist = 1e9
    best_idx = -1
    for i in range(tvecs.shape[0]):
        d = np.sqrt(tvecs[i, 0]**2 + tvecs[i, 1]**2 + tvecs[i, 2]**2)
        if d < min_dist:
            min_dist = d
            best_idx = i
    return best_idx

# AprilTag Detector
at_detector = Detector(
    families="tag36h11",
    nthreads=16,  # Utilize all CPU cores
    quad_decimate=1.0, 
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)

TAG_SIZE = 0.12  # Tag size in meters (12 cm)

print("Starting video capture...")

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Use webcam
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce internal buffering
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Use MJPG to reduce capture latency
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = manual mode (on some cameras)
cap.set(cv2.CAP_PROP_EXPOSURE, -7) 

cap2 = cv2.VideoCapture(2, cv2.CAP_V4L2)
cap2.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce internal buffering
cap2.set(cv2.CAP_PROP_FPS, 60)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Use MJPG to reduce capture latency

print("Video captures started.")

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit(1)

if not cap2.isOpened():
    print("Error: Could not open video stream 2.")
    exit(1)

last_sent_zero_time = None

import signal

def handle_sigint(signum, frame):
    print("\nInterrupt received, releasing resources...")
    cap.release()
    executor.shutdown()
    client_socket.close()
    cv2.destroyAllWindows()
    exit(0)

signal.signal(signal.SIGINT, handle_sigint)

SEND_EVERY_N_FRAMES = 1  # If network is getting oversaturated, turn this up
frame_counter = 0

latest_frame = None
latest_intake_frame = None
intake_annotated_frame = None
annotated_frame = None

print("Starting threads...")
frame_lock = threading.Lock()

annotated_frame_lock = threading.Lock()
intake_frame_lock = threading.Lock()
intake_annotated_frame_lock = threading.Lock()

print("Locks initialized.")

print("Compiling JIT functions...")
# Prime Numba functions to avoid first-call delay
extract_euler_angles(np.eye(3))
find_closest_tag_index(np.random.rand(10, 3))

def capture_main_frame():
    global latest_frame
    while True:
        cap.grab()
        ret, frame = cap.retrieve()
        if not ret or frame is None:
            continue
        with frame_lock:
            latest_frame = frame

def capture_intake_frame():
    global latest_intake_frame
    while True:
        cap2.grab()
        ret, frame = cap2.retrieve()
        if not ret or frame is None:
            continue
        with intake_frame_lock:
            latest_intake_frame = frame

executor = ThreadPoolExecutor(max_workers=5)
executor.submit(capture_main_frame)
executor.submit(capture_intake_frame)

print("Executors started.")

def tag_detection_loop():
    global last_detection_time, last_sent_zero_time, annotated_frame, frame_counter
    print("Starting tag detection loop...")
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = at_detector.detect(gray)
        
        tag_data = []
        for tag in tags:
            corners = np.array(tag.corners, dtype=np.float32)
            obj_points = np.array([
                [-TAG_SIZE / 2, -TAG_SIZE / 2, 0],
                [TAG_SIZE / 2, -TAG_SIZE / 2, 0],
                [TAG_SIZE / 2, TAG_SIZE / 2, 0],
                [-TAG_SIZE / 2, TAG_SIZE / 2, 0]
            ], dtype=np.float32)

            success, rvec, tvec = cv2.solvePnP(obj_points, corners, camera_matrix, dist_coeffs)
            if success:
                tag_data.append((tag, tvec, rvec))

        if tag_data:
            tvec_array = np.array([t[1].flatten() for t in tag_data])
            idx = find_closest_tag_index(tvec_array)
            closest_tag = tag_data[idx]
        else:
            closest_tag = None
        
        if closest_tag:
            tag, tvec, rvec = closest_tag
            #tvec[0][0] = tvec[0][0] + 1
            x_offset, y_offset, z_offset = tvec.flatten()
            
            R, _ = cv2.Rodrigues(rvec)
            roll, pitch, yaw = extract_euler_angles(R)
            

            pitchd = np.degrees(pitch)
            
            frame_counter += 1
            if frame_counter % SEND_EVERY_N_FRAMES == 0:
                client_socket.sendto(struct.pack("ffffff", 0, 0, 0, x_offset, z_offset, pitchd), (IP_ADDRESS, PORT))
                #print("Sent PID corrections:", x_correction, z_correction, yaw_correction)
            
            cv2.putText(frame, f"X: {x_offset:.2f}m, Y: {y_offset:.2f}m, Z: {z_offset:.2f}m", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Roll: {np.degrees(roll):.2f}, Pitch: {pitchd:.2f}, Yaw: {np.degrees(yaw):.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Raw tvec: X={tvec[0][0]:.2f}, Y={tvec[1][0]:.2f}, Z={tvec[2][0]:.2f}", (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            for i in range(4):
                pt1 = tuple(corners[i].astype(int))
                pt2 = tuple(corners[(i + 1) % 4].astype(int))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        with annotated_frame_lock:
            annotated_frame = frame.copy()

        if tags:
            last_detection_time = time.time()
            last_sent_zero_time = None
        else:
            if time.time() - last_detection_time > 0.2:
                if last_sent_zero_time is None:
                    client_socket.sendto(struct.pack("ffffff", 0, 0, 0, 0, 0, 0), (IP_ADDRESS, PORT))

                    last_sent_zero_time = time.time()

executor.submit(tag_detection_loop)

def tag_intake_loop():
    global latest_intake_frame, intake_annotated_frame
    print("Starting tag intake loop...")
    MIN_AREA = 500  # Minimum contour area to be considered a pipe

    while True:
        with intake_frame_lock:
            if latest_intake_frame is None:
                continue
            frame = latest_intake_frame.copy()

        # Convert to HSV and apply white color mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_white = np.array([20, 0, 180])   # Slight yellow tint
        upper_white = np.array([40, 80, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # Additional mask for shadows (darker grays)
        lower_shadow = np.array([0, 0, 50])
        upper_shadow = np.array([180, 50, 150])
        shadow_mask = cv2.inRange(hsv, lower_shadow, upper_shadow)

        # Additional mask for dark-red text
        lower_red1 = np.array([0, 100, 50])
        upper_red1 = np.array([10, 255, 150])
        lower_red2 = np.array([160, 100, 50])
        upper_red2 = np.array([180, 255, 150])
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Combine all masks
        mask = cv2.bitwise_or(mask, shadow_mask)
        mask = cv2.bitwise_or(mask, red_mask)

        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        annotated = frame.copy()

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if 1.5 < aspect_ratio < 10:  # Looks like a pipe seen from the side
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(annotated, "Pipe", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        with intake_annotated_frame_lock:
            intake_annotated_frame = annotated.copy()

executor.submit(tag_intake_loop)

def generate_mjpeg():
    while True:
        with annotated_frame_lock:
            if annotated_frame is None:
                continue
            ret, jpeg = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                continue
            frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_raw_mjpeg():
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            ret, jpeg = cv2.imencode('.jpg', latest_frame)
            if not ret:
                continue
            frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_intake_mjpeg():
    while True:
        with intake_annotated_frame_lock:
            if intake_annotated_frame is None:
                continue
            ret, jpeg = cv2.imencode('.jpg', intake_annotated_frame)
            if not ret:
                continue
            frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/raw_video_feed')
def raw_video_feed():
    return Response(generate_raw_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/intake_video_feed')
def video_intake_feed():
    return Response(generate_intake_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

print("Starting server...")
app.run(host="0.0.0.0", port=5000)
