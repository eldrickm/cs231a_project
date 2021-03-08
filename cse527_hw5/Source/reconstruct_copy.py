# ================================================
# Skeleton codes for HW5
# Read the skeleton codes carefully and put all your
# codes into function "reconstruct_from_binary_patterns"
# ================================================

import cv2
import numpy as np
from math import log, ceil, floor
import matplotlib.pyplot as plt
import pickle
import sys
import open3d as o3d

scale_factor = 1.0
ref_bonus = cv2.imread("../Results/Input/images_new/images_new/aligned001.jpg", cv2.IMREAD_COLOR)
ref_white = cv2.resize(cv2.imread("../Results/Input/images_new/images_new/aligned000.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
ref_black = cv2.resize(cv2.imread("../Results/Input/images_new/images_new/aligned001.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
ref_avg   = (ref_white + ref_black) / 2.0
ref_on    = ref_avg + 0.05 # a threshold for ON pixels
ref_off   = ref_avg - 0.05 # add a small buffer region

h,w = ref_white.shape

# mask of pixels where there is projection
proj_mask = (ref_white > (ref_black + 0.05))

scan_bits = np.zeros((h,w), dtype=np.uint16)

# analyze the binary patterns from the camera
for i in range(0,15):
    # read the file
    patt_gray = cv2.resize(cv2.imread("../Results/Input/images_new/images_new/aligned%03d.jpg"%(i+2), cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)

    # mask where the pixels are ON
    on_mask = (patt_gray > ref_on) & proj_mask

    # this code corresponds with the binary pattern code
    bit_code = np.uint16(1 << i)

    # TODO: populate scan_bits by putting the bit_code according to on_mask
    scan_bits[on_mask == True] |= bit_code

print("load codebook")
# the codebook translates from <binary code> to (x,y) in projector screen space
with open("binary_codes_ids_codebook.pckl","rb") as f:
    binary_codes_ids_codebook = pickle.load(f, encoding='bytes')

corresp_image = np.zeros((960,1280,3), np.uint8)
colormesh = []
camera_points = []
projector_points = []
for x in range(w):
    for y in range(h):
        if not proj_mask[y,x]:
            continue # no projection here
        if scan_bits[y,x] not in binary_codes_ids_codebook:
            continue # bad binary code

        # TODO: use binary_codes_ids_codebook[...] and scan_bits[y,x] to
        # TODO: find for the camera (x,y) the projector (p_x, p_y).
        # TODO: store your points in camera_points and projector_points
        x_p = binary_codes_ids_codebook[scan_bits[y,x]][0]
        y_p = binary_codes_ids_codebook[scan_bits[y,x]][1]
        if x_p >= 1279 or y_p >= 799: # filter
            continue

        camera_points.append((x/2.0,y/2.0))
        projector_points.append((x_p,y_p))
        colormesh.append((ref_bonus[y][x][2], ref_bonus[y][x][1], ref_bonus[y][x][0]))
        corresp_image[y,x][2] = x_p*255/(x_p+y_p)
        corresp_image[y,x][1] = y_p*255/(x_p+y_p)
        # IMPORTANT!!! : due to differences in calibration and acquisition - divide the camera points by 2

# now that we have 2D-2D correspondances, we can triangulate 3D points!

# load the prepared stereo calibration between projector and camera
with open("stereo_calibration.pckl","rb") as f:
    d = pickle.load(f, encoding='bytes')
    camera_K    = d[b'camera_K']
    camera_d    = d[b'camera_d']
    projector_K = d[b'projector_K']
    projector_d = d[b'projector_d']
    projector_R = d[b'projector_R']
    projector_t = d[b'projector_t']
#%%
# TODO: use cv2.undistortPoints to get normalized points for camera, use camera_K and camera_d
# TODO: use cv2.undistortPoints to get normalized points for projector, use projector_K and projector_d
#Reference: https://programtalk.com/python-examples/cv2.undistortPoints/
camera_uv = np.array(camera_points, dtype=np.float32)
cam_pts = camera_uv.size // 2
camera_uv.shape = (cam_pts, 1, 2)
# print camera_uv
camera_norm = cv2.undistortPoints(camera_uv, camera_K, camera_d,P=camera_K)
# print 'camera_norm',camera_norm

proj_uv = np.array(projector_points, dtype=np.float32)
proj_pts = proj_uv.size // 2
proj_uv.shape = (proj_pts, 1, 2)
# print proj_uv.shape
proj_norm = cv2.undistortPoints(proj_uv, projector_K, projector_d,P=projector_K)
# print 'proj_norm',proj_norm


# TODO: use cv2.triangulatePoints to triangulate the normalized points
P0 = np.dot(camera_K, np.array([ [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0]   ]))

P1 = np.concatenate((np.dot(projector_K,projector_R),np.dot(projector_K,projector_t)), axis = 1)
triang_res = cv2.triangulatePoints(P0, P1, camera_norm, proj_norm)


# TODO: use cv2.convertPointsFromHomogeneous to get real 3D points
points_3d = cv2.convertPointsFromHomogeneous(triang_res.T)


	# TODO: name the resulted 3D points as "points_3d"
count = 0
color_3d = []
mask_3d = []
for i in range(points_3d.shape[0]):
    if (points_3d[i][0][2] > 200) & (points_3d[i][0][2] < 1400):
        mask_3d.append((round(points_3d[i][0][0]), round(points_3d[i][0][1]), round(points_3d[i][0][2])))
        color_3d.append((colormesh[i]))
        count = count + 1

# print 'count', count
ans = np.array(mask_3d)
final_3d = ans.reshape(ans.shape[0],-1).reshape(ans.shape[0],1,3)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(final_3d[:, 0, :])
o3d.visualization.draw_geometries([pcd]) # Visualize the point cloud

#return filter_3d

# mask = (points_3d[:, :, 2] > 200) & (points_3d[:, :, 2] < 1400)
# return points_3d[mask]
# return points_3d[mask]
