# This is a copy of HeadPose.py modified to use
# the 9-dot pattern instead of the head pose.

import cv2
import numpy as np
import argparse  # PY

ap = argparse.ArgumentParser()
ap.add_argument("--image_dir", type=str)
args = vars(ap.parse_args())

# Read Image
image_dir = args["image_dir"]
image_filename = "NineDotTemplate.jpg"
image_full_path = image_dir + image_filename
image = cv2.imread(image_full_path)
if image is None:
    print('File not found ' + image_filename)
    sys.exit()

size = image.shape

##!! NOTE: you need at least 4 points or solvePnP throws an exception.
# We'll choose dots 1, 3, 7, 9
# 2D image points. If you change the image, you need to change vector
image_points = np.array([
    (320, 240),  # center of 640x480 image
    (210, 188),  # upper left dot
    (438, 189), # upper right
    (193, 353), # lower left
    (461, 355) # lower right
], dtype="double")

# 3D model points.
model_points = np.array([
    (0.0, 0.0, 0.0),  # center
    (-9.5, 11.0, 0),  # upper left dot
    (7.5, 11.0, 0), # upper right
    (-9.5, -1.9, 0), # lower left
    (7.5, -1.9, 0) # lower right
])

# Camera internals

focal_length = size[1]
center = (size[1] / 2, size[0] / 2)
camera_matrix = np.array(
    [[focal_length, 0, center[0]],
     [0, focal_length, center[1]],
     [0, 0, 1]], dtype="double"
)

print("Camera Matrix :\n {0}".format(camera_matrix))

dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                              dist_coeffs)  ##PY, flags=cv2.CV_ITERATIVE)

print("solvePnP return " + str(success))
print("Rotation Vector:\n {0}".format(rotation_vector))
print("Translation Vector:\n {0}".format(translation_vector))

# Project the 3D point of the dot on the upper left onto the image plane.
# Draw a line from the image center to this point.
# Now add a world point approximately half-way between dots 4 and 7.
# (-9.5, 3.3, 0)

(dots_2D, jacobian) = cv2.projectPoints(np.array([(-9.5, 11.0, 0), (-9.5, 4.6, 0)]), rotation_vector,
                                                 translation_vector, camera_matrix, dist_coeffs)

##!! Note - he does not *project* the center point, he just takes it from
# the image points.
for p in image_points:
    cv2.circle(image, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

center = (int(image_points[0][0]), int(image_points[0][1]))
upper_left_dot = (int(dots_2D[0][0][0]), int(dots_2D[0][0][1]))
cv2.line(image, center, upper_left_dot, (255, 0, 0), 2)

left_column_dot = (int(dots_2D[1][0][0]), int(dots_2D[1][0][1]))
cv2.line(image, center, left_column_dot, (255, 0, 0), 2)

# Display image
cv2.imshow("Output", image)
cv2.waitKey(0)