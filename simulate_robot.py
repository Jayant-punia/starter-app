import mujoco
import mujoco.viewer
import time

# Try to load your URDF file
try:
   model = mujoco.MjModel.from_xml_path("urdf/my_robot.urdf") # Change this to your exact filename
   data = mujoco.MjData(model)
   print("✅ Success! URDF is fully compatible with MuJoCo.")

   # Open the 3D window
   with mujoco.viewer.launch_passive(model, data) as viewer:
       while viewer.is_running():
           mujoco.mj_step(model, data)
           viewer.sync()
           time.sleep(0.01)
except Exception as e:
   print(f"❌ Test Failed! MuJoCo error: {e}")
