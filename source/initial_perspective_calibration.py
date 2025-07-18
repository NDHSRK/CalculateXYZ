# first model
# https://github.com/pacogarcia3/hta0-horizontal-robot-arm/blob/master/initial_camera_calibration.py
# 7/14/2025 replace by
# https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
# up to and including the call to cv2.projectPoints

import numpy as np
import cv2
import glob
import argparse # PY

##########################################################
# Main
##########################################################

ap = argparse.ArgumentParser()
ap.add_argument("--image_dir", type=str)
ap.add_argument("--save_dir", type=str)
args = vars(ap.parse_args())

image_dir = args["image_dir"]
save_dir = args["save_dir"]

# load the image
image_filename = "NineDotTemplate.jpg"
image_full_path = image_dir + image_filename
image = cv2.imread(image_full_path)
if image is None:
    print('File not found')
    sys.exit()

# load camera calibration
cam_mtx = np.load(save_dir + 'cam_mtx.npy')
#load center points from New Camera matrix
cx = cam_mtx[0,2]
cy = cam_mtx[1,2]
print("cx: " + str(cx) + ",cy " + str(cy))

dist=np.load(save_dir + 'dist.npy')

#MANUALLY INPUT YOUR MEASURED POINTS HERE
# world center + 9 world points
## PY center of one gray tile: 29x 29y cm
##PY our world points come from the measurement of the x and y coordinates
# in cm of each of the 9 circles in the patter.
total_points_used=10
X_center=29.0
Y_center=29.0
Z_center=48.0
worldPoints=np.array([[X_center,Y_center,Z_center],
                       [19.4, 17.8, 57.0],
                       [27.8, 15.0, 56.0],
                       [36.4, 18.0, 56.5],
                       [19.4, 24.7, 52.5],
                       [27.8, 24.5, 51.5],
                       [36.3, 24.6, 52.0],
                       [19.4, 30.9, 48.0],
                       [27.8, 31.0, 47.4],
                       [36.3, 31.1, 48.0]], dtype=np.float32)

#MANUALLY INPUT THE DETECTED IMAGE COORDINATES HERE
##PY these come measuring the center point of each dot
# in the 9-dot template in Gimp.
#[u,v] center + 9 Image points
imagePoints=np.array([[cx,cy],
                      [210, 188],  # undistorted [216, 202]
                      [325, 190],  # undistorted [325, 205]
                      [438, 189],  # undistorted [432, 204]
                      [203, 265],  # undistorted [210, 276]
                      [327, 265],  # undistorted [326, 277]
                      [448, 267],  # undistorted [440, 276]
                      [193, 353],  # undistorted [210, 360]
                      [329, 353],  # undistorted [327, 361]
                      [461, 355]], dtype=np.float32) # undistorted [452, 361]

##PY Follow other examples and use the height of the camera as Z.
##**TODO This is *not* correct, Z must be measured from the camera
# outwards. So follow Paco Garcia and measure what he calls d* from
# the camera to each real world point, then calculate Z.

#FOR REAL WORLD POINTS, CALCULATE Z from d*
'''
for i in range(1,total_points_used):
    #start from 1, given for center Z=d*
    #to center of camera
    wX=worldPoints[i,0]-X_center
    wY=worldPoints[i,1]-Y_center
    wd=worldPoints[i,2]

    d1=np.sqrt(np.square(wX)+np.square(wY))
    wZ=np.sqrt(np.square(wd)-np.square(d1))
    worldPoints[i,2]=wZ

print(worldPoints)
'''

print("Camera Matrix")
print(cam_mtx)
print("Distortion Coeff")
print(dist)
print(">==> Calibration Loaded")

##**TODO Note: CameraCalibration actually performs these steps
# and saves the results in a file.
# Follow the OpenCV tutorial and undistort the image.
h, w = image.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cam_mtx, dist, (w, h), 1, (w, h))

##**TODO What is this for exactly? undistort should be paired with newcameramtx,
# cam_mtx should be paired with the distorted image - projectPoints will
# compensate.
undistorted = cv2.undistort(image, cam_mtx, dist, None, newcameramtx)

# crop the image
#x, y, w, h = roi
#undistorted = undistorted[y:y + h, x:x + w]
cv2.imwrite(image_dir + 'undistorted.png', undistorted)

print("solvePNP")
ret, rvec1, tvec1 = cv2.solvePnP(worldPoints, imagePoints, newcameramtx, dist)

print("pnp return value: " + str(ret))
print("pnp rvec1: rotation " + str(rvec1))
print("pnp tvec1: translation " + str(tvec1))

##**TODO It's not clear what I'm supposed to do with this number.
mean_error = 0
for i in range(len(worldPoints)):
    # imgpoints2 is ndarray [1, 1, 2]
    imgpoints2, _ = cv2.projectPoints(worldPoints[i], rvec1, tvec1, cam_mtx, dist)
    error = cv2.norm(imagePoints[i], imgpoints2[0][0], cv2.NORM_L2) / len(imagePoints)
    mean_error += error

print("total error: {}".format(mean_error / len(worldPoints)))

# Draw the center and all 9 dots at their 2D positions.
projected_image_points, _ = cv2.projectPoints(worldPoints, rvec1, tvec1, cam_mtx, dist)
for p in projected_image_points:
    cv2.circle(image, (int(p[0][0]), int(p[0][1])), 5, (0, 255, 0), 2)

cv2.imwrite(image_dir + '2D_positions.png', image)

# Project a 3D point of the dot at position 3 onto the image plane.
# Just draw a circle at that point.
#position_3 = np.array([[37.5, 17.5, 38.1]], dtype=np.float32)
#center_point2D, jacobian = cv2.projectPoints(position_3, rvec1, tvec1, newcameramtx, dist)

# Draw a circle around the projected point.
#projected_point = (int(center_point2D[0][0][0]), int(center_point2D[0][0][1]))
#cv2.circle(image, projected_point, 5, (0, 255, 0), 2)

# For comparison draw a circle at image point 5.
#cv2.circle(image, (int(imagePoints[5][0]), int(imagePoints[5][1])), 5, (0, 0, 255), 2)

# Display image
cv2.imshow("2D points", image)
cv2.waitKey(0)

# Now go back the other way: start with the image coordinates
# and compute the 3D world point.
'''
        u (int): x-coordinate of the point in the image.
        v (int): y-coordinate of the point in the image.
        mtx (ndarray): Camera matrix.
        dist (ndarray): Distortion coefficients.
        rvec (ndarray): Rotation vector.
        tvec (ndarray): Translation vector.
'''



