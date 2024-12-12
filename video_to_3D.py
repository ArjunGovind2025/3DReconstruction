import os
import glob
import cv2
import numpy as np
import open3d as o3d
from pytransform3d.rotations import matrix_from_quaternion
from replicate import Client

# Paths and Directories
data_dir = "./video1873OM"  # Path to the directory containing input files
output_frames_dir = "frames"
os.makedirs(output_frames_dir, exist_ok=True)

# Collect Input Files
video_files = glob.glob(os.path.join(data_dir, "*.mp4"))
abc_files = glob.glob(os.path.join(data_dir, "*.abc"))

# Camera Parameters for pinhoel cam- this can be adjusted
fx, fy = 1000, 1000
cx, cy = 640, 360
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0,  0,  1]], dtype=np.float64)

# Extract frames from video
video_path = video_files[0]  
cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frames = []
for i in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break
    frame_path = os.path.join(output_frames_dir, f"frame_{i}.png")
    cv2.imwrite(frame_path, frame)
    frames.append(frame_path)
cap.release()

# generates camera positonn without rotation
camera_poses = []
for i in range(len(frames)):
    pos = np.array([0.0, 0.0, -1.0 * i])
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    R = matrix_from_quaternion(quat)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    camera_poses.append(T)

camera_poses = np.array(camera_poses)

# feature detection using SIFT this method can be chaged
sift = cv2.SIFT_create()
all_keypoints = []
all_descriptors = []

for frame_path in frames:
    img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    kp, des = sift.detectAndCompute(img, None)
    all_keypoints.append(kp)
    all_descriptors.append(des)

# match keypoints between frames
bf = cv2.BFMatcher()
test = .75 # can be changed
matches_between_frames = []
for i in range(len(all_descriptors) - 1):
    matches = bf.knnMatch(all_descriptors[i], all_descriptors[i + 1], k=2)
    good = [m for m, n in matches if m.distance < test * n.distance]
    matches_between_frames.append(good)

# Projection matrix from poses
def projection_matrix_from_pose(K, T):
    Tcw = np.linalg.inv(T)
    return K @ Tcw[:3, :]

# 3D reconstruction using Triangulation
all_3d_points = []

for i, good_matches in enumerate(matches_between_frames):
    kp1 = all_keypoints[i]
    kp2 = all_keypoints[i + 1]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    P1 = projection_matrix_from_pose(camera_matrix, camera_poses[i])
    P2 = projection_matrix_from_pose(camera_matrix, camera_poses[i + 1])
    pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3d = (pts4d[:3] / pts4d[3]).T
    all_3d_points.append(pts3d)

all_3d_points = np.vstack(all_3d_points) if all_3d_points else np.empty((0, 3))

# Depth Estimation Using Depth-Anything API
client = Client("depth-anything/Depth-Anything-V2")

dense_points = []

for frame_index, frame_path in enumerate(frames):
    image = open(frame_path, "rb")
    result = client.predict(image=image, api_name="/on_submit")

    depth = np.array(result["depth"])  # Assuming depth is part of the result

    h, w = depth.shape
    i_coords, j_coords = np.indices((h, w))
    X = (j_coords - cx) * depth / fx
    Y = (i_coords - cy) * depth / fy
    Z = depth
    frame_points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

    T = camera_poses[frame_index]
    ones = np.ones((frame_points.shape[0], 1))
    frame_points_h = np.hstack([frame_points, ones])
    world_points_h = (T @ frame_points_h.T).T
    world_points = world_points_h[:, :3]

    dense_points.append(world_points)

dense_points = np.vstack(dense_points) if dense_points else np.empty((0, 3))

# combine points 
all_points = np.vstack([all_3d_points, dense_points])
all_points = all_points[np.isfinite(all_points).all(axis=1)]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_points)
o3d.io.write_point_cloud("output_point_cloud.ply", pcd)

# mesh generation via poisson reconstruciton
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
mesh.compute_vertex_normals()

# remove low density points
vertices_to_remove = densities < np.quantile(densities, 0.01)
mesh.remove_vertices_by_mask(vertices_to_remove)

o3d.io.write_triangle_mesh("reconstructed_mesh.obj", mesh)
