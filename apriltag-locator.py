# Behold, the Tower of Babel of import statements—somehow both redundant *and* incomplete.
import numpy as np
import cv2
from pupil_apriltags import Detector
import socket  # Import socket module
import time, struct
from concurrent.futures import ThreadPoolExecutor
import asyncio

import threading
from numba import njit
import json
import struct
import socket
import threading

# Creating global variables like it’s 1999.
last_detection_time = time.time()
raw_frame = None
sending_frame = False

# Define the target IP and port
IP_ADDRESS = "10.102.52.2"  # Roborio ip
PORT = 1234

loadExecutor = ThreadPoolExecutor(max_workers=2)

# Ah yes, UDP—the protocol of choice when you like your data delivery like your pizza: unordered and possibly missing.
# Create a persistent UDP socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Just broadcasting like it’s a college radio station—no encryption, no shame.
# Enable UDP broadcasting globally
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

# Load camera calibration parameters
camera_matrix = np.load("camera_matrix.npy")
dist_coeffs = np.load("dist_coeffs.npy")

latest_tag_info = None
tag_info_lock = threading.Lock()
frame_lock = threading.Lock()
intake_frame_lock = threading.Lock()  # Added intake_frame_lock
intake_annotated_frame_lock = threading.Lock()

 # Euler angle extraction—because understanding quaternions is for nerds.
@njit(cache=True, fastmath=True) # The JIT functions are fast, but the rest of the code moves at the speed of existential dread.
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
# Brutal nearest neighbor logic. No sorting, just vibes.
def find_closest_tag_index(tvecs):
    min_dist = 1e9
    best_idx = -1
    for i in range(tvecs.shape[0]):
        d = np.sqrt(tvecs[i, 0]**2 + tvecs[i, 1]**2 + tvecs[i, 2]**2)
        if d < min_dist:
            min_dist = d
            best_idx = i
    return best_idx

# Throwing all CPU threads at this problem like it's a coding Final Boss.
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

# Hand-tuned camera magic that only works on THIS laptop, under THIS moon phase.
RES = (640, 480)

cap = None
cap2 = None

CAM_MAIN_PATH = "/dev/v4l/by-id/usb-046d_C270_HD_WEBCAM_B420BF60-video-index0"
CAM_INTAKE_PATH = "/dev/v4l/by-id/usb-046d_C270_HD_WEBCAM_F6B19A10-video-index0"

def initialize_cameras():
    global cap, cap2
    # Two camera streams? Bold. Unstable. Glorious chaos.
    cap = cv2.VideoCapture(CAM_MAIN_PATH, cv2.CAP_V4L2)  # Use webcam
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce internal buffering
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RES[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RES[1])
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Use MJPG to reduce capture latency
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = manual mode (on some cameras)
    cap.set(cv2.CAP_PROP_EXPOSURE, -7) 

    cap2 = cv2.VideoCapture(CAM_INTAKE_PATH, cv2.CAP_V4L2)
    cap2.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce internal buffering
    cap2.set(cv2.CAP_PROP_FPS, 60)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, RES[0])
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, RES[1])
    cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Use MJPG to reduce capture latency
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit(1)

    if not cap2.isOpened():
        print("Error: Could not open video stream 2.")
        exit(1)

    print("Video captures started.")

 # Classic "hope the camera is plugged in" check.

def prime_jit():
     # JIT warm-up ritual to avoid that awkward 2-second delay during real work.
    print("Compiling JIT functions...")
    # Prime Numba functions to avoid first-call delay
    extract_euler_angles(np.eye(3))
    find_closest_tag_index(np.random.rand(10, 3))

loadExecutor.submit(initialize_cameras)
loadExecutor.submit(prime_jit)

last_sent_zero_time = None

import signal

# The signal handler? Perfectly designed to fail gracefully… if you squint hard enough.
def handle_sigint(signum, frame):
    print("\nInterrupt received, releasing resources...")
    cap.release()
    executor.shutdown()
    client_socket.close()
    cv2.destroyAllWindows()
    exit(0)

signal.signal(signal.SIGINT, handle_sigint)

# "SEND_EVERY_N_FRAMES = 1" – because the network definitely wants *all* the frames, all the time.
SEND_EVERY_N_FRAMES = 1  # If network is getting oversaturated, turn this up
frame_counter = 0

latest_frame = None
latest_intake_frame = None
combined_frame = None


print("Starting threads...")
frame_lock = threading.Lock()

print("Locks initialized.")

 # Dedicated threads to babysit two video feeds like helicopter parents.
def capture_main_frame():
    global latest_frame
    while True:
        cap.grab()
        ret, frame = cap.retrieve()
        if not ret or frame is None:
            continue
        with frame_lock:
            latest_frame = frame

 # Dedicated threads to babysit two video feeds like helicopter parents.
def capture_intake_frame():
    global latest_intake_frame
    while True:
        cap2.grab()
        ret, frame = cap2.retrieve()
        if not ret or frame is None:
            continue
        with intake_frame_lock:  # Updated to reapply thread safety
            latest_intake_frame = frame

loadExecutor.shutdown(wait=True)
executor = ThreadPoolExecutor(max_workers=10)
executor.submit(capture_main_frame)
executor.submit(capture_intake_frame)

print("Executors started.")

# "Pipe" detection code so aggressive, it might classify a baguette as industrial plumbing.
def tag_detection_loop():
    global last_detection_time, last_sent_zero_time, frame_counter, latest_tag_info
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
            with tag_info_lock:
                # When in doubt, return a deeply unhelpful structure full of zeros.
                latest_tag_info = {"v": None, "l": []}
        
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
            
            with tag_info_lock:
                latest_tag_info = {
                    "v": [round(x_offset, 2), round(y_offset, 2), round(z_offset, 2),
                          round(np.degrees(roll), 1), round(pitchd, 1), round(np.degrees(yaw), 1)],
                    "l": [[int(corners[i][0]), int(corners[i][1]),
                           int(corners[(i + 1) % 4][0]), int(corners[(i + 1) % 4][1])]
                          for i in range(4)]
                }

        if tags:
            last_detection_time = time.time()
            last_sent_zero_time = None
        else:
            if time.time() - last_detection_time > 0.2:
                if last_sent_zero_time is None:
                    client_socket.sendto(struct.pack("ffffff", 0, 0, 0, 0, 0, 0), (IP_ADDRESS, PORT))

                    last_sent_zero_time = time.time()

executor.submit(tag_detection_loop)

# Pipe detection: turning color masks and contour spaghetti into half-reliable rectangles.
def tag_intake_loop():
    global latest_intake_frame, intake_annotated_frame
    print("Starting tag intake loop...")
    MIN_AREA = 500  # Minimum contour area to be considered a pipe

    while True:
        with intake_frame_lock:  # Updated to reapply thread safety
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

# Do not proccess intake frame
#executor.submit(tag_intake_loop)

# Masterpiece assembler: glues two chaotic streams into one mediocre vertical mosaic.
def frame_combiner_loop():
    global combined_frame
    latest_main = None
    latest_intake = None

    while True:
        updated = False

        if frame_lock.acquire(blocking=False):
            try:
                if latest_frame is not None:
                    latest_main = latest_frame.copy()
                    updated = True
            finally:
                frame_lock.release()

        if intake_frame_lock.acquire(blocking=False):
            try:
                if latest_intake_frame is not None:
                    latest_intake = latest_intake_frame.copy()
                    updated = True
            finally:
                intake_frame_lock.release()

        if not updated:
            time.sleep(0.005)
            continue

        labeled_frames = []
        for label, frame in zip(["Main", "Intake"], [latest_main, latest_intake]):
            if frame is None:
                continue
            labeled = frame.copy()
            cv2.putText(labeled, label, (10, labeled.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            labeled_frames.append(labeled)

        if not labeled_frames:
            continue

        height, width = labeled_frames[0].shape[:2]
        for i in range(len(labeled_frames)):
            labeled_frames[i] = cv2.resize(labeled_frames[i], (width, height))

        combined_frame = cv2.vconcat(labeled_frames)

threading.Thread(target=frame_combiner_loop, daemon=True).start()


# TCP streaming so barebones, it could be used to teach what *not* to do in security class.
def tcp_frame_streaming():
    TCP_PORT = 9999
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    server_socket.bind(("", TCP_PORT))
    
    # One connection only. VIP access. Everyone else can get wrecked.
    server_socket.listen(1)
    print(f"[+] Waiting for TCP client on port {TCP_PORT}...")

    while True:
        conn, addr = server_socket.accept()
        print(f"[+] TCP client connected from {addr}")
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        try:
            last_send_time = 0
            SEND_INTERVAL = 1/20

            while True:
                if combined_frame is None:
                    time.sleep(0.005)
                    continue

                now = time.time()
                if now - last_send_time < SEND_INTERVAL:
                    time.sleep(0.0001)
                    continue
                last_send_time = now

                success, encoded = cv2.imencode('.jpg', combined_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
                if not success:
                    continue

                data = encoded.tobytes()
                tag_json = ""
                with tag_info_lock:
                    if latest_tag_info is not None:
                        def convert(obj):
                            if isinstance(obj, np.generic):
                                return obj.item()
                            if isinstance(obj, tuple):
                                return [convert(i) for i in obj]
                            if isinstance(obj, list):
                                return [convert(i) for i in obj]
                            if isinstance(obj, dict):
                                return {k: convert(v) for k, v in obj.items()}
                            return obj

                        safe_tag_info = convert(latest_tag_info)
                        tag_json = json.dumps(safe_tag_info)
                    else:
                        tag_json = json.dumps({"text_lines": ["No tag detected"], "lines": []})

                tag_bytes = tag_json.encode("utf-8")
                tag_length = len(tag_bytes)

                # Format: [unsigned short tag_len][tag_json][jpeg_data]
                payload = struct.pack(">H", tag_length) + tag_bytes + data
                length = struct.pack(">L", len(payload))
                
                # Hope you're on a LAN, because this blob dump would cry on real-world internet.
                try:
                    conn.sendall(length + payload)
                except (ConnectionResetError, BrokenPipeError):
                    print("[-] TCP client disconnected.")
                    break
        finally:
            conn.close()

#threading.Thread(target=tcp_frame_streaming, daemon=True).start()
tcp_frame_streaming()

