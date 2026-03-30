# A combination of
# https://docs.opencv.org/3.3.0/dc/dbb/tutorial_py_calibration.html
# https://github.com/pacogarcia3/hta0-horizontal-robot-arm/blob/master/initial_camera_calibration.py
# https://learnopencv.com/camera-calibration-using-opencv/

import numpy as np
import cv2
import glob
import argparse # PY

ap = argparse.ArgumentParser()
ap.add_argument("--image_dir", type=str)
ap.add_argument("--save_dir", type=str)
args = vars(ap.parse_args())

image_dir = args["image_dir"]
save_dir = args["save_dir"]

# termination criteria

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

# PY we're following the learnopencv example, which uses a 9x6 chessboard
# in the landscape orientation. So the number of rows is 6, the number of
# columns 9.
objp = np.zeros((6 * 9, 3), np.float32)

# PY follow learnopencv
objp[:, :2] = np.mgrid[0:6, 0:9].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
images = glob.glob(image_dir + '*.jpg')

win_name = "Verify"
cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("getting images")
for fname in images:
    img = cv2.imread(fname)
    print(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (6, 9), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (6, 9), corners2, ret)
        cv2.imshow(win_name, img)
        cv2.waitKey(500)

    img1 = img

cv2.destroyAllWindows()

print(">==> Starting calibration")
ret, cam_mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# The return value is the root mean square (RMS) re-projection error,
# which should be between .1 and 1.0.
print(ret)
print("Camera Matrix")
print(cam_mtx)
np.save(save_dir + 'cam_mtx.npy', cam_mtx)

print("Distortion Coeff")
print(dist)
np.save(save_dir + 'dist.npy', dist)

print("r vecs")
print(rvecs[2])

print("t Vecs")
print(tvecs[2])

print(">==> Calibration ended")

h, w = img1.shape[:2]
print("Image Width, Height")
print(w, h)
# if using Alpha 0, so we discard the black pixels from the distortion.  this helps make the entire region of interest is the full dimensions of the image (after undistort)
# if using Alpha 1, we retain the black pixels, and obtain the region of interest as the valid pixels for the matrix.
# Follow the standard OpenCV example and use Alpha 1 and then run undistort
newcam_mtx, roi = cv2.getOptimalNewCameraMatrix(cam_mtx, dist, (w, h), 1, (w, h))

print("Region of Interest")
print(roi)
np.save(save_dir + 'roi.npy', roi)

print("New Camera Matrix")
# print(newcam_mtx)
np.save(save_dir + 'newcam_mtx.npy', newcam_mtx)
print(np.load(save_dir + 'newcam_mtx.npy'))

inverse = np.linalg.inv(newcam_mtx)
print("Inverse New Camera Matrix")
print(inverse)

# undistort
undst = cv2.undistort(img1, cam_mtx, dist, None, newcam_mtx)

# crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# cv2.circle(dst,(308,160),5,(0,255,0),2)
cv2.imshow('img1', img1)
cv2.waitKey(5000)
cv2.destroyAllWindows()
cv2.imshow('img1', undst)
cv2.waitKey(5000)
cv2.destroyAllWindows()