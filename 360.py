import cv2
import numpy as np

# Define the size of the output bird's-eye view
OUTPUT_SIZE = (640, 480)

# Define homography matrices (adjust these based on your camera setup)
H_front = np.array([[ 1.439,  0.201, -320.0],
                    [ 0.013,  1.750, -240.0],
                    [ 0.001,  0.002,  1.0]])

H_rear = np.array([[ 1.439, -0.201,  320.0],
                   [-0.013,  1.750, -240.0],
                   [ 0.001, -0.002,  1.0]])

def get_birdseye_view(frame, H):
    """ Warps the input frame to a bird's-eye view using homography. """
    return cv2.warpPerspective(frame, H, OUTPUT_SIZE)

# Initialize video capture for both cameras
cap_front = cv2.VideoCapture(0)
cap_rear = cv2.VideoCapture(1)

# Set camera resolution (adjust if needed)
cap_front.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_front.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap_rear.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_rear.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # Read frames from both cameras
    ret_front, frame_front = cap_front.read()
    ret_rear, frame_rear = cap_rear.read()

    if not (ret_front and ret_rear):
        print("Error: Unable to read from cameras")
        break

    # Convert to bird's-eye view
    bev_front = get_birdseye_view(frame_front, H_front)
    bev_rear = get_birdseye_view(frame_rear, H_rear)

    # Flip the rear view for proper alignment
    bev_rear = cv2.flip(bev_rear, 1)

    # Stitch the two views together (side by side)
    stitched_bev = np.hstack((bev_front, bev_rear))

    # Show the top-down stitched view
    cv2.imshow("Bird's Eye View", stitched_bev)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap_front.release()
cap_rear.release()
cv2.destroyAllWindows()