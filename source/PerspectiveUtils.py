import numpy as np

class PerspectiveUtils:

    # PY extracted from original camera_realworldxyz.py because it
    # is used both for the initial perspective calibration without
    # a camera and then later by the real-time application.
    # Originally only called from the initial_perspective_calibration.py
    # and from the function detect_xyz in camera_realworldxyz.py.
    # Changed the function name to all lower case and added parameters
    # so that this function does not need to access any saved files.

    @staticmethod
    def calculate_xyz(image_pixel_x, image_pixel_y, scaling_factor, inverse_newcam_mtx,
                      tvec1, inverse_r_mtx):
        # Solve: From Image Pixel points find World Points
        uv_1 = np.array([[image_pixel_x, image_pixel_y, 1]], dtype=np.float32)
        uv_1 = uv_1.T
        suv_1 = scaling_factor * uv_1
        xyz_c = inverse_newcam_mtx.dot(suv_1)
        xyz_c = xyz_c - tvec1
        final_xyz = inverse_r_mtx.dot(xyz_c)
        return final_xyz