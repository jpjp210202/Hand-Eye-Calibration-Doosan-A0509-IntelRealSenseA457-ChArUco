"""
Check_Calib.py
Verify if hand–eye calibration is correct.

Needs:
- pose_live.txt        (base→tool from RoboDK/robot)
- handeye_T_tool_cam.txt (tool→camera from calibration)
- board_pose.txt       (camera→board from ChArUco detection)

Outputs:
- Predicted board position in robot base frame
- Error vs ground truth (if provided)
"""

import numpy as np
import re
import cv2

# ---------------- HELPER ----------------
def load_matrix(path):
    """Load a 4x4 matrix from file (handles RoboDK or plain text)."""
    with open(path, "r") as f:
        text = f.read()
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    nums = [float(n) for n in numbers]
    return np.array(nums).reshape(4, 4)

def load_board_pose(path):
    """Load rvec,tvec from file saved by detection script."""
    rvec, tvec = None, None
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts[0] == "rvec":
                rvec = np.array([float(x) for x in parts[1:]]).reshape(3,1)
            elif parts[0] == "tvec":
                tvec = np.array([float(x) for x in parts[1:]]).reshape(3,1)

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = tvec.flatten()
    return T

# ---------------- LOAD DATA ----------------
T_base_tool  = load_matrix(r"D:\MSRIT MTECH RAI\Project Files\CMTI Projects\Hand_Eye_Calibration CMTI Git\data\pose_live.txt")
T_tool_cam   = load_matrix(r"D:\MSRIT MTECH RAI\Project Files\CMTI Projects\Hand_Eye_Calibration CMTI Git\data\handeye_T_tool_cam.txt")
T_cam_board  = load_board_pose(r"D:\MSRIT MTECH RAI\Project Files\CMTI Projects\Hand_Eye_Calibration CMTI Git\data\board_pose.txt")

print("Robot pose (base→tool):\n", T_base_tool)
print("Hand–eye (tool→cam):\n", T_tool_cam)
print("Board in camera frame:\n", T_cam_board)

# ---------------- COMPUTE ----------------
T_base_board = T_base_tool @ T_tool_cam @ T_cam_board
print("\nPredicted board in base frame:\n", T_base_board)
print(f"Predicted board position [x,y,z] (mm): x:", T_base_board[:1,3]-14.5,"y:", T_base_board[1:2,3]-46,"z:", T_base_board[2:3,3])
print("Board Values JUnk: \n",T_base_board[:3,3])
# ---------------- ORIENTATION ----------------
R = T_base_board[:3,:3]

# Convert to Euler angles (XYZ convention -> roll, pitch, yaw)
sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
singular = sy < 1e-6

if not singular:
    roll  = np.degrees(np.arctan2(R[2,1], R[2,2]))
    pitch = np.degrees(np.arctan2(-R[2,0], sy))
    yaw   = np.degrees(np.arctan2(R[1,0], R[0,0]))
else:
    roll  = np.degrees(np.arctan2(-R[1,2], R[1,1]))
    pitch = np.degrees(np.arctan2(-R[2,0], sy))
    yaw   = 0

# Reference = home position (Rx=180, Ry=0, Rz=0)
home_roll, home_pitch, home_yaw = 180.0, 0.0, 0.0

d_roll  = roll  + home_roll
d_pitch = pitch - home_pitch
d_yaw   = yaw   - home_yaw

print(f"Predicted board orientation [Roll, Pitch, Yaw] (deg): "
      f"[{roll:.3f}, {pitch:.3f}, {yaw:.3f}]")
print(f"Offset from home [ΔR, ΔP, ΔY] (deg): "
      f"[{d_roll:.3f}, {d_pitch:.3f}, {d_yaw:.3f}]")


# ---------------- CUSTOM 4x4 OUTPUT ----------------
# Apply your offsets
x_corr = T_base_board[0,3] - 14.5
y_corr = T_base_board[1,3] - 46.0
z_corr = 90.0   # fixed height

T_corrected = np.eye(4)
T_corrected[:3,:3] = R   # keep orientation
T_corrected[0,3] = x_corr
T_corrected[1,3] = y_corr
T_corrected[2,3] = z_corr

print("\nCorrected 4x4 pose (paste into RoboDK):")
for row in T_corrected:
    print("[ " + ", ".join(f"{val: .6f}" for val in row) + " ];")

 # # ---------------- ORIENTATION (Yaw around Z) ----------------
# # # Extract rotation matrix
# R = T_base_board[:3,:3]
# #
# # # Compute yaw angle (rotation about Z axis), in degrees
# yaw_deg = np.degrees(np.arctan2(R[1,0], R[0,0]))
# #
# print("Predicted board yaw angle (deg):", yaw_deg, -180-yaw_deg)

# ---------------- OPTIONAL ERROR CHECK ----------------
# If you know the real [x,y,z] of the board origin (from RoboDK target or ruler), put it here:
ground_truth = None  # Example: np.array([500.0, -60.0, 250.0])
if ground_truth is not None:
    pred = T_base_board[:3,3]
    error = np.linalg.norm(pred - ground_truth)
    print("Ground truth (mm):", ground_truth)
    print("Error (mm):", error)
    if error < 10:
        print("✅ Calibration looks VALID")
    else:
        print("❌ Calibration may be wrong")
