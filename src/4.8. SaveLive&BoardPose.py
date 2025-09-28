"""
Detection_SaveBoardPose.py
Detect ChArUco board with Intel RealSense D435,
save rvec/tvec to board_pose.txt AND robot pose to pose_live.txt
for calibration check.
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

print("🔍 Point the camera at the ChArUco board... Press 'q' to quit")

try:
    while True:
        frames = pipe.wait_for_frames()
        color = np.asanyarray(frames.get_color_frame().get_data())

        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

        if ids is not None and len(ids) > 0:
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board
            )
            if charuco_ids is not None and len(charuco_ids) > 3:
                retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                    charuco_corners, charuco_ids, board,
                    cameraMatrix, distCoeffs, None, None
                )
                if retval:
                    # Draw axis for visualization
                    cv2.drawFrameAxes(color, cameraMatrix, distCoeffs, rvec, tvec, 50)
                    cv2.putText(color, "Board Detected", (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                    # Save board pose
                    with open(board_file, "w") as f:
                        f.write("rvec " + " ".join(map(str, rvec.ravel())) + "\n")
                        f.write("tvec " + " ".join(map(str, tvec.ravel())) + "\n")
                    print("✅ Saved board pose to:", board_file)

                    # Save robot pose (base→tool)
                    pose_mat = robot.Pose()
                    T_base_tool = np.array(pose_mat.rows)  # guaranteed 4x4

                    with open(pose_file, "w") as f:
                        for row in T_base_tool:
                            f.write("[ " + ", ".join(f"{val: .6f}" for val in row) + " ];\n")
                    print("✅ Saved robot pose to:", pose_file)

        cv2.imshow("ChArUco Detection", color)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    pipe.stop()
    cv2.destroyAllWindows()
