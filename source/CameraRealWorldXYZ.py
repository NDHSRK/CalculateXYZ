# Starting point on 3/30/2026 Paco Garcia's GitHub repo:
# https://github.com/pacogarcia3/hta0-horizontal-robot-arm
# File: camera_realworldxyz.py
# Last commit before the removal of commented-out code and
# unnecessary TODOs: 6f03c99

import numpy as np
import cv2

##TODO This class should be instantiated by the main driver of the toy
# picker project.

class CameraRealtimeXYZ:

    def __init__(self, data_dir):
        self.data_dir = data_dir

        # Load the files created by camera calibration and initial perspective calibration.
        self.cam_mtx = np.load(data_dir + 'cam_mtx.npy')  # PY used in undistort_image
        self.dist = np.load(data_dir + 'dist.npy')  # PY used in undistort_image
        self.newcam_mtx = np.load(data_dir + 'newcam_mtx.npy')  # PY used in constructor and in undistort_image
        self.roi = np.load(data_dir + 'roi.npy')  # PY not used
        self.rvec1 = np.load(data_dir + 'rvec1.npy')  # PY not used
        self.tvec1 = np.load(data_dir + 'tvec1.npy')  # PY needed for PerspectiveUtils
        self.R_mtx = np.load(data_dir + 'R_mtx.npy')  # PY can be local to constructor
        self.Rt = np.load(data_dir + 'Rt.npy')  # PY not used
        self.P_mtx = np.load(data_dir + 'P_mtx.npy')  # PY not used

        s_arr = np.load(data_dir + 's_arr.npy')
        self.scalingfactor = s_arr[0]  # PY needed for PerspectiveUtils

        self.inverse_newcam_mtx = np.linalg.inv(self.newcam_mtx)  # PY needed in undistort_image and PerspectiveUtils
        self.inverse_R_mtx = np.linalg.inv(self.R_mtx)  # PY needed for PerspectiveUtils

    ##TODO call cv2.undistort, which applies our camera calibration, in the main
    # driver before calling object recognition.
    # Note that it needs arguments for self.cam_mtx, self.dist, None, self.newcam_mtx,
    # which it can load in its constructor.

    ##TODO only called locally in the original; don't need a function for this.
    def undistort_image(self, image):
        image_undst = cv2.undistort(image, self.cam_mtx, self.dist, None, self.newcam_mtx)
        return image_undst

    ##TODO Call from the main driver after object recognition.
    def calculate_real_world_xyz(self, image_pixel_x, image_pixel_y):
        final_xyz_real_world = PerspectiveUtils.calculate_xyz(image_pixel_x, image_pixel_y,
                                                   self.scaling_factor, self.inverse_newcam_mtx, self.tvec1,
                                                   self.inverse_r_mtx)

        print(final_xyz_real_world)
        return final_real_world_xyz
