import cv2
import apriltag
import numpy as np
import HiwonderSDK.mecanum as mecanum
import math

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp  # Proportional gain
        self.Ki = Ki  # Integral gain
        self.Kd = Kd  # Derivative gain
        self.prev_error = 0
        self.integral = 0

    def compute(self, error):
        self.integral += error
        derivative = error - self.prev_error
        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        self.prev_error = error
        return output

# Camera Intrinsic Parameters (Replace with actual calibration values)
camera_matrix = np.array([[600, 0, 320],  # fx, 0, cx
                          [0, 600, 240],  # 0, fy, cy
                          [0, 0, 1]])     # 0,  0,  1
dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion (adjust if necessary)

tag_size = 0.1  # AprilTag size in meters

# Open Webcam (Change '0' if multiple webcams are connected)
cap = cv2.VideoCapture(0)

# Ensure webcam opens
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize AprilTag Detector
detector = apriltag.Detector()

# Initialize Mecanum Chassis
robot = mecanum.MecanumChassis()

# Initialize PID controllers
pid_x = PIDController(Kp=1.2, Ki=0.008, Kd=0.1)  # X-axis (strafe) PID
pid_yaw = PIDController(Kp=0.005, Ki=0.0001, Kd=0.001)  # Increased gains for improved yaw correction
pid_strafe = PIDController(Kp=300, Ki=0.01, Kd=0.1)  # Strafe centering PID

# Allow user to input custom target position
x_target = -0.02
y_target = -0.008
z_target = 0.6

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = detector.detect(gray)

        for tag in tags:
            tag_id = tag.tag_id
            corners = np.array(tag.corners, dtype=np.float32)

            # Define 3D model points for AprilTag (real-world coordinates)
            object_points = np.array([
                [-tag_size / 2, -tag_size / 2, 0],  # Bottom-left
                [ tag_size / 2, -tag_size / 2, 0],  # Bottom-right
                [ tag_size / 2,  tag_size / 2, 0],  # Top-right
                [-tag_size / 2,  tag_size / 2, 0]   # Top-left
            ], dtype=np.float32)

            # Solve PnP to estimate pose
            success, rvec, tvec = cv2.solvePnP(object_points, corners, camera_matrix, dist_coeffs)

            if success:
                # Convert rotation vector to rotation matrix
                R, _ = cv2.Rodrigues(rvec)

                # Convert rotation matrix to Euler angles (roll, pitch, yaw)
                _, _, _, _, _, _, eulerAngles = cv2.decomposeProjectionMatrix(np.hstack((R, tvec)))

                # Extract Euler angles in degrees
                yaw = eulerAngles[1][0]  # Rotation around Y-axis
                pitch = eulerAngles[0][0]  # Rotation around X-axis
                roll = eulerAngles[2][0]  # Rotation around Z-axis

                # Display pose information
                position = tuple(tvec.flatten())
                cv2.putText(frame, f"ID: {tag_id} Pos: ({round(position[0], 2)}, {round(position[1], 3)}, {round(position[2], 3)})", (int(corners[0][0]), int(corners[0][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Display rotation information on the frame
                cv2.putText(frame, f"Yaw: {round(yaw, 2)} Pitch: {round(pitch, 2)} Roll: {round(roll, 2)}",
                            (int(corners[0][0]), int(corners[0][1]) - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Draw bounding box around detected tag
                for i in range(4):
                    pt1 = tuple(map(int, corners[i]))
                    pt2 = tuple(map(int, corners[(i + 1) % 4]))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

                # Tuning Parameters (Adjust as needed)
                position_threshold = 20  # Allowable offset in pixels
                speed = 50  # Base movement speed

                frame_width = frame.shape[1]  # Get frame width
                edge_threshold = frame_width * 0.2  # Define how close to the edge is considered "too close"

                # Get tag center X position
                tag_center_x = np.mean(corners[:, 0])

                # Adjust movement without worrying about rotation
                x_offset = position[0] - x_target
                z_distance = position[2]

                # Define movement status text
                movement_text = "Idle"

                if abs(z_distance - z_target) > 0.05 or abs(x_offset) > 0.02 or tag_center_x < edge_threshold or tag_center_x > (frame_width - edge_threshold):
                    # Compute movement components
                    move_speed = max(min((z_distance - z_target) * 600, speed), -speed)

                    # Compute edge proximity factor (reduces yaw correction when the tag is near the edges)
                    edge_proximity = max(0, min(1, (tag_center_x - edge_threshold) / (frame_width / 2 - edge_threshold)))

                    # Reduce yaw correction dynamically based on edge proximity
                    yaw_correction = -max(min(pid_yaw.compute(yaw) * edge_proximity, 5), -5)  # Scale down yaw correction

                    strafe_correction = max(min(pid_strafe.compute(x_offset), speed), -speed)  # Keep tag centered

                    # Compute movement direction dynamically using atan2
                    movement_angle = math.degrees(math.atan2(move_speed, strafe_correction*4))  # Calculate angle
                    speed_magnitude = math.sqrt(move_speed**2 + strafe_correction**2)  # Compute resultant speed

                    # Ensure speed magnitude doesn't exceed limit
                    speed_magnitude = min(speed_magnitude, speed)

                    if speed_magnitude < 20:
                        speed_magnitude = 0
                        yaw_correction = 0

                    # Execute movement command
                    robot.set_velocity(speed_magnitude, movement_angle, yaw_correction)
                    print(f"Speed: {speed_magnitude}, Direction: {movement_angle}, Yaw: {yaw_correction}")
                    movement_text = f"Moving: {round(speed_magnitude, 2)}, Angle: {round(movement_angle, 2)}, Yaw: {round(yaw_correction, 2)}"

                else:
                    robot.set_velocity(0, 0, 0)
                    movement_text = "Holding Position"

                # Display movement information on the frame
                cv2.putText(frame, movement_text, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if tags == []:
            movement_text = "Idle"
            robot.set_velocity(0, 0, 0)
        # Display the frame
        cv2.imshow("AprilTag Detection (Rotated)", frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nEmergency stop triggered! Stopping the robot...")
    robot.set_velocity(0, 0, 0)
finally:
    cap.release()
    cv2.destroyAllWindows()
