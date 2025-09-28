"""
calibrate_all_methods.py
Compute camera intrinsics and hand–eye calibration using Doosan A0509 poses + RealSense images.
Runs all 4 OpenCV solvers for comparison.

Author: JP (CMTI Projects)
"""

import cv2
import numpy as np
import glob
import os
from scipy.spatial.transform import Rotation as R

# === CONFIG ===
SAVE_DIR = r"D:\MSRIT MTECH RAI\Project Files\CMTI Projects\Hand_Eye_Calibration CMTI Git\data"
IMG_DIR = os.path.join(SAVE_DIR, "images")
POSE_DIR = os.path.join(SAVE_DIR, "poses")

# === ChArUco board definition (must match capture.py) ===
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
CHARUCO_BOARD = cv2.aruco.CharucoBoard(
    (7, 5),        # squaresX, squaresY
    0.03,          # squareLength in meters (30 mm)
    0.021,         # markerLength in meters (21 mm)
    ARUCO_DICT
)

# === Utility ===
def homog(Rm, tvec):
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = tvec
    return T

# === STEP 1: Load robot poses ===
robot_R_abs, robot_t_abs = [], []

pose_files = sorted(glob.glob(os.path.join(POSE_DIR, "pose_*.txt")))
for f in pose_files:
    mat = np.loadtxt(f).reshape(4, 4)
    Rm = mat[:3, :3]
    tvec = mat[:3, 3]
    robot_R_abs.append(Rm)
    robot_t_abs.append(tvec)

print(f"Loaded {len(robot_R_abs)} robot poses")

# === STEP 2: Load ChArUco detections (from .npz) ===
all_corners, all_ids, image_size = [], [], None
npz_files = sorted(glob.glob(os.path.join(POSE_DIR, "charuco_*.npz")))
img_files = sorted(glob.glob(os.path.join(IMG_DIR, "img_*.png")))

for f_img, f_npz in zip(img_files, npz_files):
    npz = np.load(f_npz)
    charuco_corners = npz["corners"]
    charuco_ids = npz["ids"]

    if image_size is None:
        img = cv2.imread(f_img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

    if len(charuco_ids) > 0:
        all_corners.append(charuco_corners)
        all_ids.append(charuco_ids)
        print(f"{os.path.basename(f_img)}: {len(charuco_ids)} ChArUco corners")
    else:
        print(f"{os.path.basename(f_img)}: 0 corners (skipped)")

print(f"Loaded ChArUco detections for {len(all_corners)} / {len(img_files)} images")

# === STEP 3: Calibrate Camera Intrinsics ===
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    charucoCorners=all_corners,
    charucoIds=all_ids,
    board=CHARUCO_BOARD,
    imageSize=image_size,
    cameraMatrix=None,
    distCoeffs=None
)

print("\n=== Camera Intrinsics ===")
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coeffs:\n", dist_coeffs.ravel())
print("RMS reprojection error:", ret)

# === STEP 4: Prepare camera poses (absolute) ===
cam_R_abs, cam_t_abs = [], []
for rv, tv in zip(rvecs, tvecs):
    Rm, _ = cv2.Rodrigues(rv)
    cam_R_abs.append(Rm)
    cam_t_abs.append(tv.ravel())

# === STEP 5: Build relative motions ===
robot_R_rel, robot_t_rel = [], []
cam_R_rel, cam_t_rel = [], []

for i in range(len(robot_R_abs)-1):
    # Robot motion: base->tool
    T1 = homog(robot_R_abs[i], robot_t_abs[i])
    T2 = homog(robot_R_abs[i+1], robot_t_abs[i+1])
    A = np.linalg.inv(T1) @ T2
    robot_R_rel.append(A[:3,:3])
    robot_t_rel.append(A[:3,3])

    # Camera motion: board->cam
    T1c = homog(cam_R_abs[i], cam_t_abs[i])
    T2c = homog(cam_R_abs[i+1], cam_t_abs[i+1])
    B = np.linalg.inv(T1c) @ T2c
    cam_R_rel.append(B[:3,:3])
    cam_t_rel.append(B[:3,3])

print(f"\nBuilt {len(robot_R_rel)} relative motions")

# === STEP 6: Run all hand–eye methods ===
# === STEP 6: Run all hand–eye methods ===
methods = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS
}

def sanitize_rotation(Rm):
    """Ensure valid right-handed rotation matrix (det=+1)."""
    U, _, Vt = np.linalg.svd(Rm)
    R_clean = U @ Vt
    if np.linalg.det(R_clean) < 0:
        U[:, -1] *= -1
        R_clean = U @ Vt
    return R_clean

print("\n=== Hand–Eye Calibration Results ===")
for name, method in methods.items():
    R_x, t_x = cv2.calibrateHandEye(
        robot_R_rel, robot_t_rel,
        cam_R_rel, cam_t_rel,
        method=method
    )
    R_x = sanitize_rotation(R_x)  # << only safeguard added
    rot = R.from_matrix(R_x)
    euler_deg = rot.as_euler('xyz', degrees=True)
    print(f"\n{name} method:")
    print("Rotation matrix:\n", R_x)
    print("Translation vector (m):", t_x.ravel())
    print("Translation vector (mm):", (t_x*1000).ravel())
    print("Euler angles (XYZ, degrees):", euler_deg)
