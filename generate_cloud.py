import open3d as o3d
import numpy as np

print("Generating 3D point cloud...")

# 1. Create a point cloud object
pcd = o3d.geometry.PointCloud()

# 2. Generate 5,000 random 3D points inside a sphere shape
np.random.seed(42)
points = np.random.randn(5000, 3)
# Normalize points to sit beautifully inside a sphere shell
points /= np.linalg.norm(points, axis=1)[:, np.newaxis] 
pcd.points = o3d.utility.Vector3dVector(points)

# 3. Give the points a vibrant neon blue/green color gradient
colors = np.zeros_like(points)
colors[:, 0] = (points[:, 0] + 1) / 2  # Red channel gradient
colors[:, 2] = (points[:, 2] + 1) / 2  # Blue channel gradient
pcd.colors = o3d.utility.Vector3dVector(colors)

print("Opening 3D Interactive Window... (Use your mouse to orbit/zoom)")

# 4. Pop open the interactive 3D viewer window
o3d.visualization.draw_geometries([pcd], window_name="Physical AI - 3D Point Cloud")
