import os
from yourdfpy import URDF

urdf_path = os.path.join("urdf", "my_robot.urdf")
robot = URDF.load(urdf_path)

print("\n🎉 SUCCESS! Robot loaded correctly.")
print(f"Robot Name: {robot.robot.name}")
print(f"Links found: {[link.name for link in robot.robot.links]}")
