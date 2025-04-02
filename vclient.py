import socket
import time
VPN_IP = "100.64.0.5"  # Set to your VPN interface IP
import cv2
import numpy as np
import struct

PORT = 9999
BUFFER_SIZE = 65535  # Max UDP packet size
payload_size = struct.calcsize(">L")

def main():
    server_ip = '100.64.0.18'  # Replace with your server's IP

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.connect((server_ip, PORT))
    print(f"[+] Connected to TCP server at {server_ip}:{PORT}")

    data = b""
    total_bytes = 0
    start_time = time.time()

    try:
        while True:
            # Read message length
            while len(data) < payload_size:
                more = sock.recv(4096)
                if not more:
                    raise ConnectionError("Disconnected from server")
                data += more
            packed_size = data[:payload_size]
            data = data[payload_size:]
            frame_size = struct.unpack(">L", packed_size)[0]

            # Read the entire payload (timestamp + frame data)
            while len(data) < frame_size:
                remaining = frame_size - len(data)
                data += sock.recv(min(4096, remaining))
            payload = data[:frame_size]
            data = data[frame_size:]

            # Unpack timestamp and frame data
            server_timestamp = struct.unpack(">d", payload[:8])[0]
            frame_data = payload[8:]

            # Decode and show frame
            frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                scale_factor = 1.5  # Adjust this factor as needed
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (int(w * scale_factor), int(h * scale_factor)))

            total_bytes += frame_size
            now = time.time()
            latency_ms = (now - server_timestamp) * 1000
            elapsed_time = now - start_time
            if elapsed_time > 0:
                bandwidth_mbps = (total_bytes * 8) / (elapsed_time * 1e6)
                overlay_text = f"Latency: {latency_ms:.1f} ms | Bandwidth: {bandwidth_mbps:.2f} Mbps"
                cv2.putText(frame, overlay_text, (200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.imshow("Live TCP Video", frame)

            if cv2.waitKey(1) == 27:
                break

    except Exception as e:
        print(f"[!] Error: {e}")
    finally:
        sock.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()