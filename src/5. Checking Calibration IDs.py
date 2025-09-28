"""
5. Checking Calibration IDs (RoboDK Simulation, Corrected)
Detect IDs -> display coords -> choose ID -> move robot in RoboDK.
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs
from robodk import robolink, robomath as rm

# === CONFIG ===
Z_TARGET = 200.0   # mm (hover height)
MARKER_SIZE = 22.0 # mm (ArUco box size)
INTRINSICS_FILE = r"D:\MSRIT MTECH RAI\Project Files\CMTI Projects\Hand_Eye_Calibration CMTI Git\data\camera_intrinsics.npz"
HANDEYE_FILE    = r"D:\MSRIT MTECH RAI\Project Files\CMTI Projects\Hand_Eye_Calibration CMTI Git\data\handeye_T_tool_cam.txt"
R_opencv_to_robot = np.array([
    [0,  0, 1],
    [1,  0, 0],
    [0, -1, 0]
])
T_tool_cam = np.loadtxt(HANDEYE_FILE)

# Insert correction on the rotation part
R_tool_cam = T_tool_cam[:3,:3]
t_tool_cam = T_tool_cam[:3,3]

R_tool_cam_fixed = R_tool_cam @ R_opencv_to_robot  # rotate axes
T_tool_cam[:3,:3] = R_tool_cam_fixed
T_tool_cam[:3,3]  = t_tool_cam

# === LOAD CAMERA INTRINSICS + HAND-EYE ===
intr = np.load(INTRINSICS_FILE)
mtx, dist = intr["cameraMatrix"], intr["distCoeffs"]
T_tool_cam = np.loadtxt(HANDEYE_FILE)

# === ROBO DK CONNECT ===
RDK = robolink.Robolink()
robot = RDK.Item('Doosan Robotics A0509')
if not robot.Valid():
    raise Exception("❌ Could not find robot Doosan Robotics A0509 in RoboDK")

print("✅ Loaded intrinsics, hand-eye, and connected to RoboDK")

# === CAMERA SETUP ===
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

# === FUNCTION: compute marker pose in base frame ===
def get_marker_pose_base(corners, mtx, dist, T_base_tool, T_tool_cam, Z_TARGET=200):
    # Estimate pose wrt camera
    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, mtx, dist)
    R_cam_marker, _ = cv2.Rodrigues(rvec[0][0])
    t_cam_marker = tvec[0][0].reshape(3, 1)

    T_cam_marker = np.eye(4, dtype=float)
    T_cam_marker[:3, :3] = R_cam_marker
    T_cam_marker[:3, 3] = t_cam_marker.flatten()

    # Force all to numpy arrays
    # Force to numpy arrays, no reshape
    T_base_tool = np.array(T_base_tool, dtype=float)
    T_tool_cam = np.array(T_tool_cam, dtype=float)
    T_cam_marker = np.array(T_cam_marker, dtype=float)

    assert T_base_tool.shape == (4, 4), f"Expected 4x4, got {T_base_tool.shape}"

    # Debug: check shapes
    # print("Shapes:", T_base_tool.shape, T_tool_cam.shape, T_cam_marker.shape)

    # Multiply 4x4 matrices
    T_base_marker = T_base_tool @ T_tool_cam @ T_cam_marker

    # Override Z
    T_base_marker = np.array(T_base_marker, dtype=float).reshape(4,4)
    T_base_marker[2, 3] = Z_TARGET
    return T_base_marker


# === MAIN LOOP ===
try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color = np.asanyarray(color_frame.get_data())
        corners, ids, _ = aruco.detectMarkers(color, aruco_dict, parameters=parameters)

        if ids is not None:
            aruco.drawDetectedMarkers(color, corners, ids)

            # Display coords
            for j, id in enumerate(ids.flatten()):
                T_base_tool = np.array(robot.Pose().rows)  # base->tool from RoboDK
                T_base_marker = get_marker_pose_base(corners[j], mtx, dist, T_base_tool, T_tool_cam, Z_TARGET)
                x, y = T_base_marker[0,3], T_base_marker[1,3]
                text = f"ID {id}: X={x:.1f}, Y={y:.1f}"
                cv2.putText(color, text, (20, 30 + j*20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

        cv2.imshow("ChArUco ID Check", color)

        # Keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key != 255 and ids is not None:
            try:
                chosen_id = int(chr(key))
                if chosen_id in ids.flatten():
                    idx = list(ids.flatten()).index(chosen_id)
                    T_base_tool = np.array(robot.Pose().rows)  # base->tool as 4x4 numpy
                    T_base_marker = get_marker_pose_base(corners[idx], mtx, dist, T_base_tool, T_tool_cam, Z_TARGET)
                    print(f"✅ Moving to ID {chosen_id} at {T_base_marker[:3, 3]}")
                    robot.MoveJ(rm.Mat(T_base_marker.tolist()))

            except:
                pass

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("✅ Shutdown complete")
