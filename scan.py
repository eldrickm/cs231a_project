#! /usr/ bin/env python3

"""
Project, capture, and decode a structured light pattern scan.
Plot 3D Points
"""

import cv2
import numpy as np
import scipy.stats as stats
import json
import matplotlib.pyplot as plt
import open3d as o3d

import structuredlight as sl
import fullscreen.fullscreen as fs

CAMERA_NUM = 0
CALIBRATION_FILE = './etc/calibration_result.json'
#CAMERA_RESOLUTION = (1920, 1080)       # 1080p
# Need to really downsample resolution if using stripe
CAMERA_RESOLUTION = (1280 // 4, 720 // 4)         # 720p
VIDEO_BUFFER_LEN = 4


#%%
"""
Step 1: Project and Capture Structured Light Pattern

Project and capture structured light patterns

Our output here is a stack of N images which contain a structured light pattern
We likely want to use positive / negative thresholding here
"""

def flush_cap(cap):
    """
    An attempt to flush the video capture buffer.
    Needed because the first VIDEO_BUFFER_LEN frames of a series of captures
    essentially won't update properly / will be stale.
    This is unfortunately a hardware issue and can't be easily disabled
    """
    for i in range(5):
        cap.grab()


# Capture Background
cap = cv2.VideoCapture(CAMERA_NUM)  # External web camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])

black = np.zeros((CAMERA_RESOLUTION))

screen = fs.FullScreen(0)
screen.imshow(black)
flush_cap(cap)
_, background = cap.read()

color_image = cv2.resize(background, CAMERA_RESOLUTION)
color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
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

subtracted = illuminated.astype(np.float32) - background.astype(np.float32)
subtracted = np.clip(subtracted, 0, 255)
is_projectable = (subtracted > np.mean(subtracted))


plt.figure()
plt.imshow(background, cmap='gray')
plt.axis('off')
plt.title('Background Image')

plt.figure()
plt.imshow(illuminated, cmap='gray')
plt.axis('off')
plt.title('Illuminated Image')

plt.figure()
plt.imshow(subtracted, cmap='gray')
plt.axis('off')
plt.title('Subtracted Image')

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

# Merge X and Y correspondences
img_correspondence = cv2.merge([0.0 * np.zeros_like(x_index),
                                x_index / CAMERA_RESOLUTION[0],
                                y_index / CAMERA_RESOLUTION[1]])

# Clip correspondences to 8b
img_correspondence = np.clip(img_correspondence * 255, 0,255)
img_correspondence = img_correspondence.astype(np.uint8)

# Mask correspondences with projectable area to eliminate sprurious points
img_correspondence[~is_projectable] = 0

# Visualize correspondences
plt.figure()
plt.imshow(img_correspondence)
plt.axis('off')
plt.title('Correspondence Map')

# Construct 2D correspondences
# x_p and y_p are the 2D coordinates in projector image space
x_p = np.expand_dims(x_index[is_projectable].flatten(), -1)
y_p = np.expand_dims(y_index[is_projectable].flatten(), -1)
projector_points = np.hstack((x_p, y_p))

# x and y are the 2D coordinates in camera image space
xx, yy = np.meshgrid(np.arange(CAMERA_RESOLUTION[0]),
                   np.arange(CAMERA_RESOLUTION[1]))
x = np.expand_dims(xx[is_projectable].flatten(), -1)
y = np.expand_dims(yy[is_projectable].flatten(), -1)
camera_points = np.hstack((x, y))

# Get RGB values for all valid points
color_points = color_image[camera_points[:, 1], camera_points[:, 0]]
color_points = color_points.astype(np.float64) / 255


# Needed for OpenCV Triangulation - add a middle axis, convert to float
camera_points = np.expand_dims(camera_points, 1).astype(np.float32)
projector_points = np.expand_dims(projector_points, 1).astype(np.float32)


"""
Step 3: Undistort Correspondences

We need the following parameters:
    - camera_K    - Camera Intrinsics
    - camera_d    - Camera Distortion Coefficients
    - projector_K - Projector Intrinsics
    - projector_d - Projector Distortion Coefficients
    - projector_R - Projector Rotation wrt Camera
    - projector_t - Projector Translation wrt Camera

We can assume that the camera is at world origin.

We'll want to compute this data before-hand and just load it here.

Use camera and projector intrinsics and distortion coefficiens
to undistort our 2D correspondences
"""

with open(CALIBRATION_FILE, 'r') as f:
    data = json.load(f)

camera_K = np.array(data.get('cam_int')['data']).reshape(3,3)
camera_d = np.array(data.get('cam_dist')['data'])

# Placeholders
projector_K = np.array(data.get('proj_int')['data']).reshape(3,3)
projector_d = np.array(data.get('proj_dist')['data'])
projector_R = np.array(data.get('rotation')['data']).reshape(3,3)
projector_t = np.array(data.get('translation')['data']).reshape(3,1)


camera_norm = cv2.undistortPoints(camera_points, camera_K, camera_d,
                                  P=camera_K)
proj_norm = cv2.undistortPoints(projector_points, projector_K, projector_d,
                                P=camera_K)


"""
Step 4: Triangulation

Formulate intrinsics and extrinsics of camera and projector.
Use those to triangulate the 3D positions of each pixel.
"""
# Camera Perspective Matrix - Location at World Origin
P0 = np.dot(camera_K, np.array([[1,0,0,0],
                                [0,1,0,0],
                                [0,0,1,0]]))

# Projector Perspective Matrix
# P1 = np.concatenate((projector_K @ projector_R, projector_K @ projector_t),
#                     axis=1)
P1 = np.concatenate((camera_K @ projector_R, camera_K @ projector_t),
                    axis=1)


# Triangulate and de-homogenize points into 3D space
triangulated_points = cv2.triangulatePoints(P0, P1, camera_norm, proj_norm)
points_3d = cv2.convertPointsFromHomogeneous(triangulated_points.T)

color_3d = np.hstack((points_3d[:, 0, :], color_points))

# Filter points as needed
filtered = color_3d[np.logical_and(color_3d[:, 2] < 0.865, color_3d[:, 2] > 0.81)]
#filtered = color_3d

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(filtered[:, :3])
pcd.colors = o3d.utility.Vector3dVector(filtered[:, 3:])
o3d.visualization.draw_geometries([pcd]) # Visualize the point cloud
