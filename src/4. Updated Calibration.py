"""
handeye_calibration.py
Compute hand–eye calibration (eye-in-hand, D435 + Doosan A0509 + ChArUco).
"""

import os
import glob
import numpy as np
import cv2

# === PATHS ===
BASE_DIR = r"D:\MSRIT MTECH RAI\Project Files\CMTI Projects\Hand_Eye_Calibration CMTI Git\data"
IMG_DIR = os.path.join(BASE_DIR, "images")
POSE_DIR = os.path.join(BASE_DIR, "poses")
INTRINSICS_FILE = os.path.join(BASE_DIR, "camera_intrinsics.npz")

# === LOAD CAMERA INTRINSICS ===
intr = np.load(INTRINSICS_FILE)
cameraMatrix = intr["cameraMatrix"]
distCoeffs = intr["distCoeffs"]

print("Loaded intrinsics:")
print(cameraMatrix)
print(distCoeffs)

# === LOAD DATASETS ===
robot_poses = []   # A_i : robot base -> tool
board_rvecs = []   # rvecs of board wrt camera
board_tvecs = []   # tvecs of board wrt camera

# Match indices by filename
for idx in range(31):  # we know we have 31 samples
    pose_file = os.path.join(POSE_DIR, f"pose_{idx:03d}.txt")
    board_file = os.path.join(POSE_DIR, f"boardpose_{idx:03d}.npz")

    if not os.path.exists(pose_file) or not os.path.exists(board_file):
        print(f"Skipping missing {idx}")
        continue

    # === Robot pose (4x4 homogeneous) ===
    T_base_tool = np.loadtxt(pose_file)  # 4x4
    R_base_tool = T_base_tool[:3, :3]
    t_base_tool = T_base_tool[:3, 3]

    robot_poses.append((R_base_tool, t_base_tool))

    # === Board pose wrt camera ===
    board = np.load(board_file)
    rvec, tvec = board["rvec"], board["tvec"]

    # Ensure proper shape
    rvec = rvec.reshape(3, 1)
    tvec = tvec.reshape(3, 1)

    board_rvecs.append(rvec)
    board_tvecs.append(tvec)

print(f"Loaded {len(robot_poses)} pose pairs")

# === CONVERT board poses into camera->board transforms ===
R_target2cam, t_target2cam = [], []
for rvec, tvec in zip(board_rvecs, board_tvecs):
    R, _ = cv2.Rodrigues(rvec)
    R_target2cam.append(R)
    t_target2cam.append(tvec)

# === CONVERT robot poses into base->gripper transforms ===
R_gripper2base, t_gripper2base = [], []
for R, t in robot_poses:
    R_gripper2base.append(R)
    t_gripper2base.append(t.reshape(3, 1))

# === RUN HAND–EYE CALIBRATION ===
# Eye-in-hand: X = T_gripper2cam
R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
    R_gripper2base, t_gripper2base,
    R_target2cam, t_target2cam,
    method=cv2.CALIB_HAND_EYE_TSAI
)

print("\n=== Hand–Eye Calibration Result (Tsai–Lenz) ===")
print("Rotation (R):\n", R_cam2gripper)
print("Translation (t):\n", t_cam2gripper)

# === APPLY CORRECTIONS ===
# Use R from calibration, but override t with manual offsets
t_fixed = np.array([80.0, -15.0, -45.0])  # mm (from ruler)
# OpenCV → Robot axes conversion
R_opencv_to_robot = np.array([
    [0,  0, 1],
    [1,  0, 0],
    [0, -1, 0]
])
T_tool_cam = np.eye(4)
T_tool_cam[:3, :3] = R_cam2gripper
T_tool_cam[:3, 3] = t_fixed

# === SAVE CORRECTED RESULT ===
out_file = os.path.join(BASE_DIR, "handeye_T_tool_cam.txt")
np.savetxt(out_file, T_tool_cam, fmt="%.6f")

print("\n=== Corrected Hand–Eye Transform ===")
print(T_tool_cam)
print(f"\n💾 Saved corrected result to {out_file}")