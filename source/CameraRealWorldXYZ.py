# Starting point on 3/30/2026 Paco Garcia's GitHub repo:
# https://github.com/pacogarcia3/hta0-horizontal-robot-arm
# File: camera_realworldxyz.py

import numpy as np
import cv2
##TODO PY defer or replace import image_recognition_singlecam

# PY constructed in Garcia's main_loop.py. Assumes that CalibrateCameraXYZ
# and InitialPerspectiveCalibrationXYZ have been run separately and that
# their output files have been copied into this project's to camera_data.

##TODO This class should be instantiated by the main driver of the toy
# picker project.

##TODO Python class names should be UpperCamelCase: camera_realtimeXYZ ->
class CameraRealtimeXYZ:

    def __init__(self, data_dir):

        self.data_dir = data_dir

        # PY function definition from image_recognition_singlecam
        #def __init__(self, print_status=True, write_images=False,
        #             image_Path="/home/pi/Desktop/Captures/", testing_Path="/home/pi/Desktop/Captures/",
        #             preview_images=False, preview_autoclose=True, print_img_labels=True):
        ##TODO PY defer self.imageRec = image_recognition_singlecam.image_recognition(False, False, imgdir, imgdir, False, True, False)

        # Load the files created by camera calibration and initial perspective calibration.
        self.cam_mtx = np.load(data_dir + 'cam_mtx.npy') # PY used in undistort_image
        self.dist = np.load(data_dir + 'dist.npy') # PY used in undistort_image
        self.newcam_mtx = np.load(data_dir + 'newcam_mtx.npy') # PY used in constructor and in undistort_image
        self.roi = np.load(data_dir + 'roi.npy') # PY not used
        self.rvec1 = np.load(data_dir + 'rvec1.npy') # PY not used
        self.tvec1 = np.load(data_dir + 'tvec1.npy') # PY needed for PerspectiveUtils
        self.R_mtx = np.load(data_dir + 'R_mtx.npy') # PY can be local to constructor
        self.Rt = np.load(data_dir + 'Rt.npy') # PY not used
        self.P_mtx = np.load(data_dir + 'P_mtx.npy') # PY not used

        s_arr = np.load(data_dir + 's_arr.npy')
        self.scalingfactor = s_arr[0] # PY needed for PerspectiveUtils

        self.inverse_newcam_mtx = np.linalg.inv(self.newcam_mtx) # PY needed in undistort_image and PerspectiveUtils
        self.inverse_R_mtx = np.linalg.inv(self.R_mtx) # PY needed for PerspectiveUtils

    ##TODO PY by Python convention should be preview_image
    # PY called locally (commented out) and many times from
    # image_recognition_singlecam.py for debugging.
    '''
    def previewImage(self, text, img):
        # show full screen
        cv2.namedWindow(text, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(text, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cv2.imshow(text, img)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
    '''

    ##TODO only called locally in the original; don't need a function for this -
    # check whether cv2.undistort makes a real difference with a webcam image.
    def undistort_image(self, image):
        image_undst = cv2.undistort(image, self.cam_mtx, self.dist, None, self.newcam_mtx)
        return image_undst

    # PY the background image is used for OpenCV background
    # subtraction (absdiff), which we wouldn't use for FTC
    # IntoTheDeep. Only called from main_loop.py.
    #def load_background(self, background):
    #    self.bg_undst = self.undistort_image(background)
    #   self.bg = background

    ##TODO PY bypass image recognition until we can verify the
    # two important first steps: camera calibration and initial
    # perspective calibration.
    '''
    # PY only called from capturefromPiCamera (image loop) in main_loop.py
    def detect_xyz(self, image, calcXYZ=True, calcarea=False):

        image_src = image.copy() # PY this makes a deep copy

        # PY commented out in the original
        # if calcXYZ==True:
        #    img= self.undistort_image(image_src)
        #    bg = self.bg_undst
        # else:
        img = image_src # PY this just creates a reference
        bg = self.bg

        XYZ = []
        # self.previewImage("capture image",img_undst)
        # self.previewImage("bg image",self.bg_undst)
        obj_count, detected_points, img_output = self.imageRec.run_detection(img, self.bg)

        if (obj_count > 0):

            ##TODO PY return a class
            for i in range(0, obj_count):
                x = detected_points[i][0]
                y = detected_points[i][1]
                w = detected_points[i][2]
                h = detected_points[i][3]
                cx = detected_points[i][4]
                cy = detected_points[i][5]

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # draw center
                cv2.circle(img, (cx, cy), 3, (0, 255, 0), 2)

                cv2.putText(img, "cx,cy: " + str(self.truncate(cx, 2)) + "," + str(self.truncate(cy, 2)),
                            (x, y + h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                if calcXYZ == True:
                    ##TODO PY call PerspectiveUtils.calculate_xyz with additional arguments.
                    XYZ.append(self.calculate_XYZ(cx, cy))
                    cv2.putText(img,
                                "X,Y: " + str(self.truncate(XYZ[i][0], 2)) + "," + str(self.truncate(XYZ[i][1], 2)),
                                (x, y + h + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if calcarea == True:
                    cv2.putText(img, "area: " + str(self.truncate(w * h, 2)), (x, y - 12), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)

        return img, XYZ
'''

    # PY only called from the function detect_xyz in this file
    # and from initial_perspective_calibration.
    # PY Move to PerspectiveUtils.py and add parameters so that
    # it can be static.
    # def calculate_XYZ(self, u, v):

    #PY only called for text display in this file.
    '''
    def truncate(self, n, decimals=0):
        n = float(n)
        multiplier = 10 ** decimals
        return int(n * multiplier) / multiplier
    '''
