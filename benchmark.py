import numpy as np
import time
from numba import njit

# -------------------------
# 1. EULER ANGLES
# -------------------------

@njit(cache=True)
def extract_euler_angles_numba(R):
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
def extract_euler_angles_fastmath(R):
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

def extract_euler_angles_py(R):
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

# -------------------------
# 2. CLOSEST TAG SELECTION
# -------------------------

@njit(cache=True)
def find_closest_tag_numba(tvecs):
    min_dist = 1e9
    best_idx = -1
    for i in range(tvecs.shape[0]):
        d = np.sqrt(tvecs[i, 0]**2 + tvecs[i, 1]**2 + tvecs[i, 2]**2)
        if d < min_dist:
            min_dist = d
            best_idx = i
    return best_idx

@njit(cache=True, fastmath=True)
def find_closest_tag_fastmath(tvecs):
    min_dist = 1e9
    best_idx = -1
    for i in range(tvecs.shape[0]):
        d = np.sqrt(tvecs[i, 0]**2 + tvecs[i, 1]**2 + tvecs[i, 2]**2)
        if d < min_dist:
            min_dist = d
            best_idx = i
    return best_idx

def find_closest_tag_py(tvecs):
    min_dist = 1e9
    best_idx = -1
    for i in range(tvecs.shape[0]):
        d = np.sqrt(tvecs[i, 0]**2 + tvecs[i, 1]**2 + tvecs[i, 2]**2)
        if d < min_dist:
            min_dist = d
            best_idx = i
    return best_idx

# -------------------------
# 3. PID COMPUTATION
# -------------------------

@njit(cache=True)
def compute_pid_numba(Kp, Ki, Kd, error, prev_error, integral):
    integral += error
    derivative = error - prev_error
    output = (Kp * error) + (Ki * integral) + (Kd * derivative)
    return output, error, integral

def compute_pid_py(Kp, Ki, Kd, error, prev_error, integral):
    integral += error
    derivative = error - prev_error
    output = (Kp * error) + (Ki * integral) + (Kd * derivative)
    return output, error, integral

# -------------------------
# Benchmarking
# -------------------------

def benchmark(func, name, args, iterations=100_000):
    start = time.perf_counter()
    for _ in range(iterations):
        func(*args)
    end = time.perf_counter()
    print(f"{name} took {(end - start) * 1000:.2f} ms")

if __name__ == "__main__":
    print("Priming Numba functions...")
    start_prime = time.perf_counter()
    tvecs = np.random.rand(100, 3)
    extract_euler_angles_numba(np.eye(3))
    extract_euler_angles_fastmath(np.eye(3))
    find_closest_tag_numba(tvecs)
    find_closest_tag_fastmath(tvecs)
    compute_pid_numba(1.0, 0.1, 0.05, 0.5, 0.1, 0.2)
    end_prime = time.perf_counter()
    print(f"Priming took {(end_prime - start_prime) * 1000:.2f} ms")

    print("\nBenchmarking...\n")

    # Euler angles test
    test_R = np.array([
        [0.866, -0.5, 0],
        [0.5, 0.866, 0],
        [0, 0, 1]
    ])
    benchmark(extract_euler_angles_py, "Euler (Python)", [test_R])
    benchmark(extract_euler_angles_numba, "Euler (Numba)", [test_R])
    benchmark(extract_euler_angles_fastmath, "Euler (FastMath)", [test_R])

    # Closest tag test
    benchmark(find_closest_tag_py, "Closest Tag (Python)", [tvecs])
    benchmark(find_closest_tag_numba, "Closest Tag (Numba)", [tvecs])
    benchmark(find_closest_tag_fastmath, "Closest Tag (FastMath)", [tvecs])

    # PID compute test
    benchmark(compute_pid_py, "PID (Python)", [1.0, 0.1, 0.05, 0.5, 0.1, 0.2])
    benchmark(compute_pid_numba, "PID (Numba)", [1.0, 0.1, 0.05, 0.5, 0.1, 0.2])

    idx_numba = find_closest_tag_numba(tvecs)
    idx_fastmath = find_closest_tag_fastmath(tvecs)
    print("\nFastMath precision difference (Closest Tag):")
    print("Equal:", idx_numba == idx_fastmath)
    print("Standard output:", idx_numba)
    print("FastMath output:", idx_fastmath)

    out_numba = extract_euler_angles_numba(test_R)
    out_fastmath = extract_euler_angles_fastmath(test_R)
    print("\nFastMath precision difference:")
    print("Equal:", np.allclose(out_numba, out_fastmath))
    print("Standard output:", out_numba)
    print("FastMath output:", out_fastmath)