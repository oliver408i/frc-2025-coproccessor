import numpy as np
import cv2
from pupil_apriltags import Detector
import socket  # Import socket module
import time, struct
from flask import Flask, render_template, Response
import concurrent.futures

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
    return render_template('index.html')  # Create this file in the 'templates' folder

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

print("Video capture started.")

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit(1)

x_target = 0  # Define your desired X target
z_target = 0  # Define your desired Z target
last_sent_zero_time = None

import signal

def handle_sigint(signum, frame):
    print("\nInterrupt received, releasing resources...")
    cap.release()
    client_socket.close()
    cv2.destroyAllWindows()
    executor.shutdown()
    exit(0)

signal.signal(signal.SIGINT, handle_sigint)

SEND_EVERY_N_FRAMES = 1  # Adjust this value as needed
frame_counter = 0

latest_frame = None
annotated_frame = None

print("Starting threads...")
frame_lock = threading.Lock()

annotated_frame_lock = threading.Lock()

print("Locks initialized.")

last_sent_time = time.time()

def send_frames():
    global last_sent_time, sending_frame
    while True:
        now = time.time()
        if now - last_sent_time < 1 / 15:
            time.sleep(0.01)
            continue

        if sending_frame:
            continue  # Prevent buildup of latency

        with annotated_frame_lock:
            if annotated_frame is None:
                continue
            frame_copy = annotated_frame.copy()

        success, buffer = cv2.imencode('.jpg', frame_copy, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if not success:
            continue

        base64_frame = base64.b64encode(buffer).decode('utf-8')
        sending_frame = True
        
        last_sent_time = now

def done_sending():
    global sending_frame
    sending_frame = False

def process_frame():
    global latest_frame
    while True:
        cap.grab()  # Grab the latest frame, discard the rest
        ret, frame = cap.retrieve()
        if not ret or frame is None:
            continue
        
        with frame_lock:
            latest_frame = frame.copy()
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if not success:
            continue
        

print("Threads starting...")
time.sleep(1)  # Allow camera to warm up

print("Compiling JIT functions...")
# Prime Numba functions to avoid first-call delay
extract_euler_angles(np.eye(3))
find_closest_tag_index(np.random.rand(10, 3))
 

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

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/raw_video_feed')
def raw_video_feed():
    return Response(generate_raw_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
executor.submit(process_frame)
executor.submit(tag_detection_loop)

print("Starting server...")
app.run(host="0.0.0.0", port=5000)
