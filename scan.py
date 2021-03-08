#! /usr/bin/env python3

"""
Capture projection pattern and decode both xy.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'calibration/'))

import cv2
import numpy as np

import structuredlight as sl
from fullscreen.fullscreen import *
import calibration.calibrate as calibrate
import calibration.perspective as perspective

CAMERA_NUM = 2
PROP_FILE = './etc/camera_config.json'
PERSPECTIVE_FILE = './etc/perspective.json'
CAMERA_RESOLUTION = (1920, 1080)       # 480p

"""
Step 0: Camera and Projector Calibration

Before we get started, we need the following parameters:
    - camera_K    - Camera Intrinsics
    - camera_d    - Camera Distortion Coefficients
    - projector_K - Projector Intrinsics
    - projector_d - Projector Distortion Coefficients
    - projector_R - Projector Rotation wrt Camera
    - projector_t - Projector Translation wrt Camera

We can assume that the camera is at world origin.

We'll want to compute this data before-hand and just load it here.

Example:

    with open("stereo_calibration.pckl","r") as f:
        d = pickle.load(f)
        camera_K    = d['camera_K']
        camera_d    = d['camera_d']
        projector_K = d['projector_K']
        projector_d = d['projector_d']
        projector_R = d['projector_R']
        projector_t = d['projector_t']
"""

camera_matrix, dist_coeffs = calibrate.load_camera_props(PROP_FILE)
mapx, mapy = calibrate.get_undistort_maps(camera_matrix, dist_coeffs)
m, max_width, max_height = perspective.load_perspective(PERSPECTIVE_FILE)


"""
Step 1: Project and Capture Structured Light Pattern

Project and capture structured light patterns

Our output here is a stack of N images which contain a structured light pattern
We likely want to use positive / negative thresholding here

Example:

    captures = [imshowAndCapture(cap, img) for img in patterns]
"""

def imshowAndCapture(cap, img_pattern, delay=1):
    #  cv2.imshow("", img_pattern)
    screen = FullScreen(0)
    screen.imshow(img_pattern)
    cv2.waitKey(delay)
    ret, img_frame = cap.read()
    img_frame = calibrate.undistort_image(img_frame, mapx, mapy)
    img_frame = cv2.warpPerspective(img_frame, m, (max_width, max_height))
    img_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
    return img_gray


width, height = CAMERA_RESOLUTION

cap = cv2.VideoCapture(CAMERA_NUM)  # External web camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])

# Generate Stripe Pattern
stripe = sl.Stripe()
stripe_patterns = stripe.generate((width // 8, height // 8))

# Project and Capture Stripe Pattern
stripe_captures = [imshowAndCapture(cap, img) for img in stripe_patterns]


"""
Step 2: Create XY Correspondences in Camera and Projector Image Space

Iterating over our capture stack of N images:

For each pixel in our projector image space, that pixel has a certain
binary value associated with it (based on the stripes)

Similarly, for each pixel in our camera image space, we need to find
which binary value corresponds to it (based on the stripes projected onto it).

We may need special decoding algorithms to get the proper binary values

We want a list of [y_p, x_p] projector X,Y values for each [y, x] camera pixel
This gives us 2D correspondences
"""

# Decode Stripe Pattern
img_index = stripe.decode(stripe_captures)

# Visualize decode result
img_correspondence = np.clip(img_index/width*255.0, 0, 255).astype(np.uint8)
cv2.imshow("Correspondence Map", img_correspondence)
cv2.waitKey(0)
#  cv2.imwrite("correspondence.png", img_correspondence)
cv2.destroyAllWindows()
cap.release()


"""
Step 3: Undistort Correspondences

Use camera and projector intrinsics and distortion coefficiens
to undistort our 2D correspondences

Example:
    camera_norm = cv2.undistortPoints(camera_points, camera_K, camera_d,
                                      P=camera_K)
    proj_norm = cv2.undistortPoints(projector_points, projector_K, projector_d,
                                    P=projector_K)
"""


"""
Step 4: Triangulation

Formulate intrinsics and extrinsics of camera and projector.
Use those to triangulate the 3D positions of each pixel.

Example:
    P0 = np.dot(camera_K, np.array([[1,0,0,0],
                                    [0,1,0,0],
                                    [0,0,1,0]]))

    P1 = np.concatenate((np.dot(projector_K,projector_R),
                         np.dot(projector_K,projector_t)), axis=1)

    triang_res = cv2.triangulatePoints(P0, P1, camera_norm, proj_norm)
    points_3d = cv2.convertPointsFromHomogeneous(triang_res.T)
"""
