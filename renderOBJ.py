import open3d as o3d

# Load the .obj file
obj_path = "OBJ/reconstructed_mesh_boba.obj"  # Replace with your .obj file path
mesh = o3d.io.read_triangle_mesh(obj_path)

# Visualize the mesh
print("Visualizing the .obj file...")
o3d.visualization.draw_geometries([mesh])
