import cv2
import mujoco
import lerobot

print("🎉 Success! All systems are connected:")
print(f" -> OpenCV version: {cv2.__version__}")
print(f" -> MuJoCo version: {mujoco.__version__}")
print(f" -> LeRobot path:   {lerobot.__file__}")
