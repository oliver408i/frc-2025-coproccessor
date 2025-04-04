import socket
import time
import cv2
import numpy as np
import struct
import json

server_ip = '100.64.0.18'  # Replace with your server's IP
PORT = 9999
BUFFER_SIZE = 65535  # Max UDP packet size
payload_size = struct.calcsize(">L")


def main():
    

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

            # Read the entire payload (frame data)
            while len(data) < frame_size:
                remaining = frame_size - len(data)
                data += sock.recv(min(4096, remaining))
            payload = data[:frame_size]
            data = data[frame_size:]

            # Unpack tag info
            header_size = struct.calcsize(">H")
            (tag_len,) = struct.unpack(">H", payload[:header_size])
            tag_json = payload[header_size:header_size+tag_len].decode("utf-8")
            frame_data = payload[header_size+tag_len:]

            # Decode frame
            frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                # Draw tag overlays
                try:
                    tag_info = json.loads(tag_json)
                    if "l" in tag_info:
                        for x1, y1, x2, y2 in tag_info["l"]:
                            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    if "v" in tag_info and tag_info["v"] is not None:
                        text = f"X:{tag_info['v'][0]} Y:{tag_info['v'][1]} Z:{tag_info['v'][2]} " \
                               f"R:{tag_info['v'][3]} P:{tag_info['v'][4]} Y:{tag_info['v'][5]}"
                        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    elif "v" in tag_info and tag_info["v"] is None:
                        cv2.putText(frame, "No tag detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                except Exception as e:
                    print(f"Failed to parse or draw tag info: {e}")

                scale_factor = 1.5
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (int(w * scale_factor), int(h * scale_factor)))

            total_bytes += frame_size
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                bandwidth_mbps = (total_bytes * 8) / (elapsed_time * 1e6)
                overlay_text = f"Bandwidth: {bandwidth_mbps:.2f} Mbps"
                h, w = frame.shape[:2]
                cv2.putText(frame, overlay_text, (200, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.imshow("Live TCP Video", frame)

            key = cv2.waitKey(1)
            if key == ord('p'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"snapshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"[+] Snapshot saved as {filename}")
            elif key == 27:
                break

    except Exception as e:
        print(f"[!] Error: {e}")
    finally:
        sock.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()