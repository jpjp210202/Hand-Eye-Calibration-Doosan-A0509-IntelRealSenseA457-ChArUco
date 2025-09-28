import cv2, glob
import numpy as np

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()

images = glob.glob(r"D:\MSRIT MTECH RAI\Project Files\CMTI Projects\Hand_Eye_Calibration CMTI Git\data\intrinsics_images\calib_001.png")

for fname in images[:5]:  # check first 5
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)
    img_marked = cv2.aruco.drawDetectedMarkers(img.copy(), corners, ids)
    cv2.imshow("Check", img_marked)
    cv2.waitKey(0)

cv2.destroyAllWindows()
