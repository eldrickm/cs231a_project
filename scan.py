#! /usr/ bin/env python3

"""
Capture projection pattern and decode both xy.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'calibration/'))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

import structuredlight as sl
import fullscreen.fullscreen as fs
import calibration.calibrate as calibrate
import calibration.perspective as perspective

CAMERA_NUM = 2
PROP_FILE = './etc/camera_config.json'
PERSPECTIVE_FILE = './etc/perspective.json'
#CAMERA_RESOLUTION = (1920, 1080)       # 1080p
# Need to really downsample resolution if using stripe
CAMERA_RESOLUTION = (640 // 2, 360 // 2)         # 480p
VIDEO_BUFFER_LEN = 4

def flush_cap(cap):
    """
    An attempt to flush the video capture buffer.
    Needed because the first VIDEO_BUFFER_LEN frames of a series of captures
    essentially won't update properly / will be stale.
    This is unfortunately a hardware issue and can't be easily disabled
    """
    for i in range(5):
        cap.grab()

#%%
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

camera_K, camera_d = calibrate.load_camera_props(PROP_FILE)

# Placeholders
projector_K = camera_K
projector_d = camera_d
projector_R = np.eye(3)
projector_t = np.array([[ 5, 5, 5], ]).T
#mapx, mapy = calibrate.get_undistort_maps(camera_K, camera_d)
#m, max_width, max_height = perspective.load_perspective(PERSPECTIVE_FILE)

#%%
"""
Step 1: Project and Capture Structured Light Pattern

Project and capture structured light patterns

Our output here is a stack of N images which contain a structured light pattern
We likely want to use positive / negative thresholding here

Example:

    captures = [imshowAndCapture(cap, img) for img in patterns]
"""
# Capture Background
cap = cv2.VideoCapture(CAMERA_NUM)  # External web camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])

black = np.zeros((CAMERA_RESOLUTION))

screen = fs.FullScreen(0)
screen.imshow(black)
flush_cap(cap)
_, background = cap.read()

background = cv2.resize(background, CAMERA_RESOLUTION)
background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

white = np.ones((CAMERA_RESOLUTION)) * 255
screen.imshow(white)
cv2.waitKey(10)
flush_cap(cap)
_, illuminated = cap.read()

illuminated = cv2.resize(illuminated, CAMERA_RESOLUTION)
illuminated = cv2.cvtColor(illuminated, cv2.COLOR_BGR2GRAY)


cap.release()
cv2.destroyAllWindows()

is_projectable = (illuminated > (background + 75))


plt.figure()
plt.imshow(background, cmap='gray')
plt.axis('off')
plt.title('Background Image')

plt.figure()
plt.imshow(illuminated, cmap='gray')
plt.axis('off')
plt.title('Illuminated Image')

plt.figure()
plt.imshow(is_projectable, cmap='gray')
plt.axis('off')
plt.title('Projectable Area')

#%%
def imshowAndCapture(cap, img_pattern, screen, delay=10):
    screen.imshow(img_pattern)
    cv2.waitKey(delay)
    ret, img_frame = cap.read()
    img_frame = cv2.resize(img_frame, CAMERA_RESOLUTION)
    img_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
    return img_gray

# Generate Scan Pattern
# Stripe Pattern
stripe = sl.Stripe()
x_patterns = stripe.generate(CAMERA_RESOLUTION)
y_patterns = stripe.generate((CAMERA_RESOLUTION[1], CAMERA_RESOLUTION[0]))
y_patterns = sl.transpose(y_patterns)

# Webcams sometimes buffer by a few frames.
# We want to remove those first still frames
# So we append the last frame for as long as the video buffer
# And chop off the first frames that are lagging
for i in range(VIDEO_BUFFER_LEN):
    x_patterns.append(x_patterns[-1])
    y_patterns.append(y_patterns[-1])


# Project and Capture Stripe Pattern
cap = cv2.VideoCapture(CAMERA_NUM)  # External web camera
cap.set(cv2.CAP_PROP_BUFFERSIZE, 0);
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])

screen = fs.FullScreen(0)
screen.imshow(black)
flush_cap(cap)
x_captures = [imshowAndCapture(cap, img, screen) for img in x_patterns]
y_captures = [imshowAndCapture(cap, img, screen) for img in y_patterns]

# Chop off lagging frames
x_captures = x_captures[VIDEO_BUFFER_LEN:]
y_captures = y_captures[VIDEO_BUFFER_LEN:]

cap.release()
cv2.destroyAllWindows()

# Background Frame Subtraction
# TODO: Speed this up - list comprehension is slow
x_subtracted = [capture.astype(np.float64) - background.astype(np.float64) for
               capture in x_captures]
y_subtracted = [capture.astype(np.float64) - background.astype(np.float64) for
               capture in y_captures]
x_subtracted = np.clip(x_subtracted, 0, 255).astype(np.uint8)
y_subtracted = np.clip(y_subtracted, 0, 255).astype(np.uint8)

# Uncomment below to visualize the scan process
# for i in range(len(subtracted)):
#     fig, ax = plt.subplots(nrows=3, ncols=1, squeeze=True, figsize=(20, 20))
#     ax[0].imshow(patterns[i], cmap='gray')
#     ax[0].set_title('Stripe %d' % i)
#     ax[0].axis('off')
#     ax[1].imshow(captures[i], cmap='gray')
#     ax[1].set_title('Capture - Frame %d' % i)
#     ax[1].axis('off')
#     ax[2].imshow(subtracted[i], cmap='gray')
#     ax[2].set_title('Background Subtracted - Frame %d' % i)
#     ax[2].axis('off')


#%%
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
x_index = stripe.decode(x_subtracted)
y_index = stripe.decode(y_subtracted)

# Visualize decode result
img_correspondence = cv2.merge([0.0 * np.zeros_like(x_index),
                                x_index / CAMERA_RESOLUTION[0],
                                y_index / CAMERA_RESOLUTION[1]])

img_correspondence = np.clip(img_correspondence * 255, 0,255)
img_correspondence = img_correspondence.astype(np.uint8)

plt.figure()
plt.imshow(img_correspondence)
plt.axis('off')
plt.title('Correspondence Map')

x_p = np.expand_dims(x_index[is_projectable].flatten(), -1)
y_p = np.expand_dims(y_index[is_projectable].flatten(), -1)
projector_points = np.hstack((x_p, y_p))

xx, yy = np.meshgrid(np.arange(CAMERA_RESOLUTION[0]),
                   np.arange(CAMERA_RESOLUTION[1]))
x = np.expand_dims(xx[is_projectable].flatten(), -1)
y = np.expand_dims(yy[is_projectable].flatten(), -1)
#x = np.expand_dims(np.arange(CAMERA_RESOLUTION[1]), -1)
#y = np.expand_dims(np.arange(CAMERA_RESOLUTION[0]), -1)
camera_points = np.hstack((x, y))

#%%
# Needed for OpenCV Triangulation
camera_points = np.expand_dims(camera_points, 1).astype(np.float32)
projector_points = np.expand_dims(projector_points, 1).astype(np.float32)

#%%
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

camera_norm = cv2.undistortPoints(camera_points, camera_K, camera_d,
                                  P=camera_K)
proj_norm = cv2.undistortPoints(projector_points, projector_K, projector_d,
                                P=camera_K)

#%%
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
# Camera Perspective Matrix - Location at World Origin
P0 = np.dot(camera_K, np.array([[1,0,0,0],
                                [0,1,0,0],
                                [0,0,1,0]]))

# Projector Perspective Matrix
P1 = np.concatenate((np.dot(projector_K, projector_R),
                     np.dot(projector_K, projector_t)), axis=1)

triangulated_points = cv2.triangulatePoints(P0, P1, camera_norm, proj_norm)
points_3d = cv2.convertPointsFromHomogeneous(triangulated_points.T)

filtered = points_3d[:, 0, :]
#filtered = points_3d[points_3d[:, :, 2] < 5000]
#filtered = points_3d[points_3d[:, :, 2] > 4500]


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(filtered)
o3d.visualization.draw_geometries([pcd]) # Visualize the point cloud