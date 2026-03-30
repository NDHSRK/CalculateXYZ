# Starting point on 3/29/2026 Paco Garcia's GitHub repo:
# https://github.com/pacogarcia3/hta0-horizontal-robot-arm
# File: initial_perspective_calibration.py

import numpy as np
import cv2
import glob
import argparse # PY
import camera_realworldxyz

ap = argparse.ArgumentParser()
ap.add_argument("--save_dir", type=str)
args = vars(ap.parse_args())

save_dir = args["save_dir"]

##TODO PY For the initial calibration we don't need the entire class
# camera_realtimeXYZ() inside of camera_realworldxyz.py. We only
# need enough of a subset for the function def calculate_XYZ(self,u,v)
# to work.
cameraXYZ = camera_realworldxyz.camera_realtimeXYZ()

# PY actually a misnomer because the camera is not involved,
# only the image points.
calculatefromCam = True

# PY not used in the original file imgdir = "/home/pi/Desktop/Captures/"

##TODO PY put this on the command line
writeValues = False

# test camera calibration against all points, calculating XYZ

# load camera calibration
cam_mtx = np.load(save_dir + 'cam_mtx.npy')
dist = np.load(save_dir + 'dist.npy')
newcam_mtx = np.load(save_dir + 'newcam_mtx.npy')
roi = np.load(save_dir + 'roi.npy')

# load center points from New Camera matrix
cx = newcam_mtx[0, 2]
cy = newcam_mtx[1, 2]
fx = newcam_mtx[0, 0]
print("cx: " + str(cx) + ",cy " + str(cy) + ",fx " + str(fx))

##TODO MANUALLY INPUT YOUR MEASURED POINTS HERE
# ENTER (X,Y,d*)
# d* is the distance from your point to the camera lens. (d* = Z for the camera center)
# we will calculate Z in the next steps after extracting the new_cam matrix

# world center + 9 world points
total_points_used = 10

X_center = 10.9
Y_center = 10.7
Z_center = 43.4
# PY these are the world points for Garcia's setup. See how
# he does this in https://medium.com/@pacogarcia3/calculate-x-y-z-real-world-coordinates-from-image-coordinates-using-opencv-from-fdxlabs-0adf0ec37cef
worldPoints = np.array([[X_center, Y_center, Z_center],
                        [5.5, 3.9, 46.8],
                        [14.2, 3.9, 47.0],
                        [22.8, 3.9, 47.4],
                        [5.5, 10.6, 44.2],
                        [14.2, 10.6, 43.8],
                        [22.8, 10.6, 44.8],
                        [5.5, 17.3, 43],
                        [14.2, 17.3, 42.5],
                        [22.8, 17.3, 44.4]], dtype=np.float32)

# MANUALLY INPUT THE DETECTED IMAGE COORDINATES HERE
# PY As Garcia says: "We then use the 9 circle template I created to calculate the Image points".

# [u,v] center + 9 Image points
imagePoints = np.array([[cx, cy],
                        [502, 185],
                        [700, 197],
                        [894, 208],
                        [491, 331],
                        [695, 342],
                        [896, 353],
                        [478, 487],
                        [691, 497],
                        [900, 508]], dtype=np.float32)

# FOR REAL WORLD POINTS, CALCULATE Z from d*

for i in range(1, total_points_used):
    # start from 1, given for center Z=d*
    # to center of camera
    wX = worldPoints[i, 0] - X_center
    wY = worldPoints[i, 1] - Y_center
    wd = worldPoints[i, 2]

    d1 = np.sqrt(np.square(wX) + np.square(wY))
    wZ = np.sqrt(np.square(wd) - np.square(d1))
    worldPoints[i, 2] = wZ

print(worldPoints)

print("Camera Matrix")
print(cam_mtx)
print("Distortion Coeff")
print(dist)

print("Region of Interest")
print(roi)
print("New Camera Matrix")
print(newcam_mtx)
inverse_newcam_mtx = np.linalg.inv(newcam_mtx)
print("Inverse New Camera Matrix")
print(inverse_newcam_mtx)
if writeValues: np.save(savedir + 'inverse_newcam_mtx.npy', inverse_newcam_mtx)

print(">==> Calibration Loaded")

print("solvePNP")
ret, rvec1, tvec1 = cv2.solvePnP(worldPoints, imagePoints, newcam_mtx, dist)
print("solvePNP result: " + str(ret)) # PY

print("pnp rvec1 - Rotation")
print(rvec1)
if writeValues: np.save(savedir + 'rvec1.npy', rvec1)

print("pnp tvec1 - Translation")
print(tvec1)
if writeValues: np.save(savedir + 'tvec1.npy', tvec1)

print("R - rodrigues vecs")
R_mtx, jac = cv2.Rodrigues(rvec1)
print(R_mtx)
if writeValues: np.save(savedir + 'R_mtx.npy', R_mtx)

print("R|t - Extrinsic Matrix")
Rt = np.column_stack((R_mtx, tvec1))
print(Rt)
if writeValues: np.save(savedir + 'Rt.npy', Rt)

print("newCamMtx*R|t - Projection Matrix")
P_mtx = newcam_mtx.dot(Rt)
print(P_mtx)
if writeValues: np.save(savedir + 'P_mtx.npy', P_mtx)

# [XYZ1]


# LET'S CHECK THE ACCURACY HERE
s_arr = np.array([0], dtype=np.float32)
s_describe = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

for i in range(0, total_points_used):
    print("=======POINT # " + str(i) + " =========================")

    print("Forward: From World Points, Find Image Pixel")
    XYZ1 = np.array([[worldPoints[i, 0], worldPoints[i, 1], worldPoints[i, 2], 1]], dtype=np.float32)
    XYZ1 = XYZ1.T
    print("{{-- XYZ1")
    print(XYZ1)
    suv1 = P_mtx.dot(XYZ1)
    print("//-- suv1")
    print(suv1)
    s = suv1[2, 0]
    uv1 = suv1 / s
    print(">==> uv1 - Image Points")
    print(uv1)
    print(">==> s - Scaling Factor")
    print(s)
    s_arr = np.array([s / total_points_used + s_arr[0]], dtype=np.float32)
    s_describe[i] = s
    if writeValues: np.save(savedir + 's_arr.npy', s_arr)

    print("Solve: From Image Pixels, find World Points")

    uv_1 = np.array([[imagePoints[i, 0], imagePoints[i, 1], 1]], dtype=np.float32)
    uv_1 = uv_1.T
    print(">==> uv1")
    print(uv_1)
    suv_1 = s * uv_1
    print("//-- suv1")
    print(suv_1)

    print("get camera coordinates, multiply by inverse Camera Matrix, subtract tvec1")
    xyz_c = inverse_newcam_mtx.dot(suv_1)
    xyz_c = xyz_c - tvec1
    print("      xyz_c")
    inverse_R_mtx = np.linalg.inv(R_mtx)
    XYZ = inverse_R_mtx.dot(xyz_c)
    print("{{-- XYZ")
    print(XYZ)

    ##TODO PY This will break until we make a subset of
    # camera_realtimeXYZ() inside of camera_realworldxyz.py
    # with only the function calculate_XYZ.
    if calculatefromCam:
        cXYZ = cameraXYZ.calculate_XYZ(imagePoints[i, 0], imagePoints[i, 1])
        print("camXYZ")
        print(cXYZ)

s_mean, s_std = np.mean(s_describe), np.std(s_describe)

print(">>>>>>>>>>>>>>>>>>>>> S RESULTS")
print("Mean: " + str(s_mean))
# print("Average: " + str(s_arr[0]))
print("Std: " + str(s_std))

print(">>>>>> S Error by Point")

for i in range(0, total_points_used):
    print("Point " + str(i))
    print("S: " + str(s_describe[i]) + " Mean: " + str(s_mean) + " Error: " + str(s_describe[i] - s_mean))

