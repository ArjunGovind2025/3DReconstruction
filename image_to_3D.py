import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image
from gradio_client import Client, handle_file
from google.colab import files


image_path = "image1.png"

# Open the image using PIL
image = Image.open(image_path)

# Use Depth-Anything for depth estimation
client = Client("depth-anything/Depth-Anything-V2")
result = client.predict(
    image=handle_file(image_path), 
    api_name="/on_submit"
)

print(type(result))
print(result)

depth_map_path = result[1]  # Use the second element in the tuple

# open depth map image
depth_map_image = Image.open(depth_map_path)

# convert to  numpy array
depth_map = np.array(depth_map_image)

# Display the depth map
plt.imshow(depth_map, cmap='viridis')
plt.colorbar()
plt.title("Depth Map")
plt.show()


original_image = Image.open(image_path)

# convert to a NumPy array to access its dimensions and colors
image_np = np.array(original_image)

height, width, _ = image_np.shape
depth_map = depth_map / np.max(depth_map)

# create point cloud
xx, yy = np.meshgrid(np.arange(width), np.arange(height))
xx = xx.flatten()
yy = yy.flatten()
z = depth_map.flatten()

#use to filter by depth to isolate subject
valid = (z > 0)
x = xx[valid]
y = yy[valid]
z = z[valid]


x = (x - width / 2) / width
y = -(y - height / 2) / height

valid = valid.reshape(height, width)  # Reshape to (height, width)
colors = image_np[valid, :3] / 255.0  # Select RGB channels, then apply mask

# combine into a point cloud
points = np.stack([x, y, z], axis=1)
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
point_cloud.colors = o3d.utility.Vector3dVector(colors)

# remove outliers 
cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=5, std_ratio=3.0)  # Adjusted values
cleaned_point_cloud = point_cloud.select_by_index(ind)

# reduce size fro easier computation
voxel_size = 0.002 
downsampled_point_cloud = cleaned_point_cloud.voxel_down_sample(voxel_size=voxel_size)

# get normals
downsampled_point_cloud.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=15)
)
downsampled_point_cloud.orient_normals_consistent_tangent_plane(k=10)

# Generate a mesh using Poisson
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    downsampled_point_cloud, depth=10  # Use a lower depth value for faster results
)
# Fill holes in the mesh
mesh = mesh.remove_degenerate_triangles()
mesh = mesh.remove_duplicated_triangles()
mesh = mesh.remove_duplicated_vertices()
mesh = mesh.remove_non_manifold_edges()

#remove low density points
density_threshold = np


output_mesh_file = "image1.ply"
o3d.io.write_triangle_mesh(output_mesh_file, mesh)
print(f"Cleaned mesh saved to {output_mesh_file}")

