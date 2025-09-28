"""
Detection_SaveBoardPose.py
Detect ChArUco board with Intel RealSense D435,
save rvec/tvec to board_pose.txt AND robot pose to pose_live.txt
for calibration check.

Additional:
- Detects individual ArUco IDs in the live feed
- Maps IDs to alphabet keys: id0='a', id1='b', etc.
- When you press the corresponding key, it saves that marker's pose.
"""

import cv2
import numpy as np
import pyrealsense2 as rs
from robodk import robolink

# ---------------- CONFIG ----------------
BOARD_ROWS = 5
BOARD_COLS = 7
SQUARE_LENGTH = 30.0   # mm
MARKER_LENGTH = 22.0   # mm
DICT = cv2.aruco.DICT_4X4_50

intrinsics_file = r"D:\MSRIT MTECH RAI\Project Files\CMTI Projects\Hand_Eye_Calibration CMTI Git\data\camera_intrinsics.npz"
board_file = r"D:\MSRIT MTECH RAI\Project Files\CMTI Projects\Hand_Eye_Calibration CMTI Git\data\board_pose.txt"
pose_file  = r"D:\MSRIT MTECH RAI\Project Files\CMTI Projects\Hand_Eye_Calibration CMTI Git\data\pose_live.txt"

# ---------------- LOAD INTRINSICS ----------------
data = np.load(intrinsics_file)
cameraMatrix = data["cameraMatrix"]
distCoeffs = data["distCoeffs"]

# ---------------- ARUCO + CHARUCO ----------------
aruco_dict = cv2.aruco.getPredefinedDictionary(DICT)
board = cv2.aruco.CharucoBoard(
    (BOARD_COLS, BOARD_ROWS), SQUARE_LENGTH, MARKER_LENGTH, aruco_dict
)
params = cv2.aruco.DetectorParameters()

# ---------------- ROBODK CONNECTION ----------------
RDK = robolink.Robolink()
robot = RDK.Item('', robolink.ITEM_TYPE_ROBOT)

# ---------------- REALSENSE ----------------
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipe.start(cfg)

print("🔍 Point the camera at the ChArUco board...")
print("Press 'a' for ID0, 'b' for ID1, ... 'q' to quit.")

try:
    while True:
        frames = pipe.wait_for_frames()
        color = np.asanyarray(frames.get_color_frame().get_data())

        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

        if ids is not None and len(ids) > 0:
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(color, corners, ids)

            # --- Estimate poses of each marker ---
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, MARKER_LENGTH, cameraMatrix, distCoeffs
            )

            for i, marker_id in enumerate(ids.flatten()):
                rvec = rvecs[i].reshape(3,1)
                tvec = tvecs[i].reshape(3,1)

                cv2.drawFrameAxes(color, cameraMatrix, distCoeffs, rvec, tvec, 20)
                c = tuple(corners[i][0].mean(axis=0).astype(int))
                label = f"ID{marker_id}"
                cv2.putText(color, label, c,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # ---------------- KEYBOARD INPUT ----------------
        k = cv2.waitKey(1) & 0xFF

        # Quit
        if k == ord("q"):
            break

        # Save specific marker if key pressed (id0='a', id1='b', etc)
        if ids is not None and len(ids) > 0:
            for i, marker_id in enumerate(ids.flatten()):
                expected_key = ord('a') + marker_id
                if k == expected_key:
                    rvec = rvecs[i].reshape(3,1)
                    tvec = tvecs[i].reshape(3,1)

                    # Save this marker pose
                    with open(board_file, "w") as f:
                        f.write("rvec " + " ".join(map(str, rvec.ravel())) + "\n")
                        f.write("tvec " + " ".join(map(str, tvec.ravel())) + "\n")
                    print(f"✅ Saved board pose for ID{marker_id} to:", board_file)

                    # Save robot pose (base→tool)
                    pose_mat = robot.Pose()
                    T_base_tool = np.array(pose_mat.rows)  # guaranteed 4x4

                    with open(pose_file, "w") as f:
                        for row in T_base_tool:
                            f.write("[ " + ", ".join(f"{val: .6f}" for val in row) + " ];\n")
                    print("✅ Saved robot pose to:", pose_file)

        cv2.imshow("ChArUco + ArUco Detection", color)

finally:
    pipe.stop()
    cv2.destroyAllWindows()
