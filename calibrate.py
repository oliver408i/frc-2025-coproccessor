import cv2
import numpy as np
import glob

# Define the checkerboard dimensions
CHECKERBOARD = (8,6)
SQUARE_SIZE = 25  # Square size in mm

# Criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (0,0,0), (1,0,0), (2,0,0) ....
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# Arrays to store object points and image points from all images
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Capture images using camera
cap = cv2.VideoCapture(0)  # Use the default camera

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

captured_frames = []
while True:  # Capture images for calibration
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        continue
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    
    key = cv2.waitKey(1) & 0xFF
    if ret:
        cv2.drawChessboardCorners(frame, CHECKERBOARD, corners, ret)
    cv2.imshow('Checkerboard Detection', frame)
    
    if key == ord('c'):
        if ret:
            captured_frames.append((gray, corners))
            print(f"Captured {len(captured_frames)} images")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Perform camera calibration
for gray, corners in captured_frames:
    objpoints.append(objp)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners2)

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

if ret:
    print("Camera matrix:")
    print(camera_matrix)
    print("\nDistortion coefficients:")
    print(dist_coeffs)
    
    # Save calibration results
    np.save("camera_matrix.npy", camera_matrix)
    np.save("dist_coeffs.npy", dist_coeffs)
    print("Calibration parameters saved.")
else:
    print("Calibration failed.")