import numpy as np
import cv2
from pupil_apriltags import Detector
import socket  # Import socket module
import time, struct
from concurrent.futures import ThreadPoolExecutor
import asyncio
from aiohttp import web
from aiortc import RTCPeerConnection, VideoStreamTrack, RTCSessionDescription
import av
from aiortc.rtcrtpsender import RTCRtpSender
from aiortc.contrib.media import MediaRelay

import base64
import threading
from numba import njit
import json

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

pcs = set()
relay = MediaRelay()

async def index(request):
    return web.FileResponse('templates/index.html')

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

RES = (640, 480)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Use webcam
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce internal buffering
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, RES[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RES[1])
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Use MJPG to reduce capture latency
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = manual mode (on some cameras)
cap.set(cv2.CAP_PROP_EXPOSURE, -7) 

cap2 = cv2.VideoCapture(2, cv2.CAP_V4L2)
cap2.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce internal buffering
cap2.set(cv2.CAP_PROP_FPS, 60)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, RES[0])
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, RES[1])
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

executor = ThreadPoolExecutor(max_workers=10)
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

class VideoCameraTrack(VideoStreamTrack):
    def __init__(self, source='main', fps_override=None):
        super().__init__()
        self.source = source
        self.last_frame_time = 0
        self.fps_override = fps_override

    async def recv(self):
        # max_fps = self.fps_override["value"] if self.fps_override else 20
        # frame_interval = 1 / max_fps
        # now = time.time()

        # if now - self.last_frame_time < frame_interval:
        #     await asyncio.sleep(0.001)
        #     return await self.recv()

        # self.last_frame_time = now
        pts, time_base = await self.next_timestamp()
        
        if self.source == 'main':
            with frame_lock:
                frame = latest_frame.copy() if latest_frame is not None else None
        elif self.source == 'intake':
            with intake_frame_lock:
                frame = latest_intake_frame.copy() if latest_intake_frame is not None else None
        elif self.source == 'annotated':
            with annotated_frame_lock:
                frame = annotated_frame.copy() if annotated_frame is not None else None
        else:
            return None

        if frame is None:
            await asyncio.sleep(1 / 30)
            return await self.recv()
        #frame = cv2.resize(frame, (320, int(frame.shape[0] * 320 / frame.shape[1])))

        now_str = time.strftime("%H:%M:%S", time.localtime())
        cv2.putText(frame, f"Time: {now_str}", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        frame = av.VideoFrame.from_ndarray(frame, format="bgr24")

        

        frame.pts = pts
        frame.time_base = time_base

        #self.last_frame_time = time.time()
        return frame

async def offer(request):
    if request.method == "OPTIONS":
        return web.Response(
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            }
        )

    stream_id = request.match_info.get('stream_id', 'main')
    
    def set_bandwidth(sdp, kbps=2000):
        lines = sdp.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("m=video"):
                lines.insert(i + 1, f"b=AS:{kbps}")
                break
        return "\n".join(lines)
    
    params = await request.json()

    sdp = set_bandwidth(params["sdp"], kbps=4000)
    #sdp = params["sdp"]
    offer = RTCSessionDescription(sdp=sdp, type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    # Force H264 codec on sender
    def force_codec(pc, sender, forced_codec):
        kind = forced_codec.split("/")[0]
        codecs = RTCRtpSender.getCapabilities(kind).codecs
        transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
        transceiver.setCodecPreferences(
            [codec for codec in codecs if codec.mimeType == forced_codec]
        )

    video_track = VideoCameraTrack(source=stream_id, fps_override=None)
    relayed_track = relay.subscribe(video_track)
    video_sender = pc.addTrack(relayed_track)

    fps_override = {"value": 10}  # Default max FPS

    @pc.on("datachannel")
    def on_datachannel(channel):
        if channel.label == "feedback":
            @channel.on("message")
            def on_message(message):
                try:
                    data = json.loads(message)
                    decode_fps = data.get("fps", 0)
                    fps_override["value"] = min(20, decode_fps-2)
                    print(f"Client decode FPS: {decode_fps}, adjusted max FPS: {fps_override['value']}")
                except Exception as e:
                    print("Failed to parse feedback message:", e)

    force_codec(pc, video_sender, "video/H264") 

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"Connection state is {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    await pc.setRemoteDescription(offer)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    # Filter out IPv6 candidates from the SDP answer
    sdp_lines = pc.localDescription.sdp.split("\n")
    filtered_lines = [line for line in sdp_lines if not ("a=candidate" in line and "fd7a:115c:" in line)]
    local_description = RTCSessionDescription(sdp="\n".join(filtered_lines), type=pc.localDescription.type)
    
    return web.Response(
        content_type="application/json",
        text=json.dumps({"sdp": local_description.sdp, "type": local_description.type}),
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }
    )

async def on_shutdown(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

app2 = web.Application()
app2.router.add_get("/", index)
app2.router.add_route("OPTIONS", "/offer/{stream_id}", offer)
app2.router.add_route("POST", "/offer/{stream_id}", offer)
app2.on_shutdown.append(on_shutdown)

web.run_app(app2, host="0.0.0.0", port=5000)
