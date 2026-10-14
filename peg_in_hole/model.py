"""Build a self-contained MJCF model for peg-in-hole.

The hole is a square aperture (side = 2 * hole_radius) surrounded by four box
collision geoms. A translucent cylinder marks the nominal circular hole for
visualization only (contype/conaffinity = 0).
"""

from __future__ import annotations

from .config import PegInHoleConfig

# Fixed kinematic offsets used by both MJCF and Cartesian mapping.
Y_CARRIAGE_Z = -0.02  # body pos of y_carriage relative to x_carriage
EE_Z_FROM_Z_JOINT = -0.10
EE_BLOCK_HALF_H = 0.02


def peg_tip_from_z_joint(cfg: PegInHoleConfig) -> float:
    """Z of peg tip relative to the z-carriage joint frame origin."""
    # Tip is at the bottom of the peg, which starts at the bottom of the EE block.
    return EE_Z_FROM_Z_JOINT - EE_BLOCK_HALF_H - 2.0 * cfg.peg_half_length


def peg_tip_from_base(cfg: PegInHoleConfig) -> float:
    """Z of peg tip relative to robot_base when all slide joints are at 0."""
    return Y_CARRIAGE_Z + peg_tip_from_z_joint(cfg)


def build_mjcf(cfg: PegInHoleConfig) -> str:
    """Return MJCF XML string for the configured peg-in-hole scene."""
    hr = cfg.hole_radius
    ow = cfg.hole_outer_half
    hz = cfg.hole_plate_half_height
    hx, hy = cfg.hole_center_xy
    hz_c = cfg.hole_plate_z
    bx, by, bz = cfg.robot_base_pos
    pr = cfg.peg_radius
    ph = cfg.peg_half_length
    kp = cfg.actuator_kp
    dr = cfg.actuator_dampratio
    dt = cfg.timestep

    # Four walls forming a square hole of half-width `hr`.
    y_wall_hy = (ow - hr) / 2.0
    y_wall_y = (ow + hr) / 2.0
    x_wall_hx = (ow - hr) / 2.0
    x_wall_x = (ow + hr) / 2.0

    # Peg hangs below the EE block inside the ee body frame.
    peg_center_in_ee = -EE_BLOCK_HALF_H - ph
    tip_in_ee = peg_center_in_ee - ph

    return f"""
<mujoco model="peg_in_hole">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{dt}" gravity="0 0 -9.81" integrator="implicitfast"
          cone="elliptic" impratio="10"/>

  <visual>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.25 0.25 0.25" specular="0.2 0.2 0.2"/>
    <rgba haze="0.15 0.2 0.25 1"/>
  </visual>

  <default>
    <!-- margin must be << clearance so a centered peg is not falsely in contact -->
    <geom friction="0.5 0.005 0.0001" solref="0.012 1" solimp="0.9 0.95 0.001"
          condim="3" margin="0.00005"/>
    <joint damping="8" armature="0.01"/>
    <position kp="{kp}" dampratio="{dr}" ctrlrange="-0.5 0.5" forcerange="-200 200"/>
  </default>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.85 0.85 0.88"
             rgb2="0.7 0.7 0.75" width="512" height="512"/>
    <material name="grid_mat" texture="grid" texrepeat="4 4" reflectance="0.05"/>
    <material name="metal" rgba="0.45 0.48 0.52 1" specular="0.4" shininess="0.6"/>
    <material name="peg_mat" rgba="0.85 0.25 0.15 1" specular="0.3" shininess="0.4"/>
    <material name="hole_mat" rgba="0.35 0.55 0.75 1" specular="0.2"/>
    <material name="hole_viz" rgba="0.2 0.7 0.9 0.25"/>
  </asset>

  <worldbody>
    <light name="key" pos="0.4 0.3 1.2" dir="-0.2 -0.2 -1" diffuse="0.9 0.9 0.9"/>
    <light name="fill" pos="-0.2 -0.3 0.8" dir="0.2 0.2 -1" diffuse="0.35 0.35 0.4"/>
    <geom name="floor" type="plane" size="0.8 0.8 0.05" material="grid_mat"/>

    <!-- Hole fixture: square aperture ≈ circular hole of radius {hr:.5f} m -->
    <body name="hole_fixture" pos="{hx} {hy} {hz_c}">
      <geom name="wall_yp" type="box" size="{ow} {y_wall_hy} {hz}"
            pos="0 {y_wall_y} 0" material="hole_mat"/>
      <geom name="wall_ym" type="box" size="{ow} {y_wall_hy} {hz}"
            pos="0 {-y_wall_y} 0" material="hole_mat"/>
      <geom name="wall_xp" type="box" size="{x_wall_hx} {hr} {hz}"
            pos="{x_wall_x} 0 0" material="hole_mat"/>
      <geom name="wall_xm" type="box" size="{x_wall_hx} {hr} {hz}"
            pos="{-x_wall_x} 0 0" material="hole_mat"/>
      <!-- Visual-only circular hole outline (no collision) -->
      <geom name="hole_circle_viz" type="cylinder" size="{hr} {hz * 1.01}"
            material="hole_viz" contype="0" conaffinity="0"/>
      <site name="hole_center" pos="0 0 {hz}" size="0.003" rgba="0.1 0.9 0.3 1"/>
    </body>

    <!-- 3-axis Cartesian robot: slide_x, slide_y, slide_z -->
    <!-- Swap this subtree later for a UR5e; keep site name peg_tip and actuators. -->
    <!-- Structural geoms are visual/inertial only (contype=0) so they do not
         self-collide and lock the slides. Only the peg participates in contact. -->
    <body name="robot_base" pos="{bx} {by} {bz}">
      <geom name="base_plate" type="box" size="0.06 0.06 0.015" pos="0 0 0.02"
            material="metal" mass="1.0" contype="0" conaffinity="0"/>

      <body name="x_carriage" pos="0 0 0">
        <joint name="slide_x" type="slide" axis="1 0 0" range="-0.15 0.15"/>
        <geom name="x_block" type="box" size="0.035 0.035 0.018"
              material="metal" mass="0.4" contype="0" conaffinity="0"/>

        <body name="y_carriage" pos="0 0 {Y_CARRIAGE_Z}">
          <joint name="slide_y" type="slide" axis="0 1 0" range="-0.15 0.15"/>
          <geom name="y_block" type="box" size="0.03 0.03 0.015"
                material="metal" mass="0.3" contype="0" conaffinity="0"/>

          <body name="z_carriage" pos="0 0 0">
            <joint name="slide_z" type="slide" axis="0 0 1" range="-0.50 0.05"/>
            <geom name="z_rail" type="capsule" size="0.012"
                  fromto="0 0 0.02 0 0 -0.08" material="metal" mass="0.25"
                  contype="0" conaffinity="0"/>

            <body name="ee" pos="0 0 {EE_Z_FROM_Z_JOINT}">
              <geom name="ee_block" type="box" size="0.025 0.025 {EE_BLOCK_HALF_H}"
                    material="metal" mass="0.15" contype="0" conaffinity="0"/>
              <geom name="peg" type="cylinder" size="{pr} {ph}"
                    pos="0 0 {peg_center_in_ee}" material="peg_mat" mass="0.05"/>
              <site name="peg_tip" pos="0 0 {tip_in_ee}" size="0.004"
                    rgba="1 1 0 1"/>
              <!-- Site for optional MuJoCo force sensor (educational; FSM uses contacts) -->
              <site name="wrist_site" pos="0 0 0" size="0.006" rgba="0.2 0.8 1 0.5"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <position name="act_x" joint="slide_x" kp="{kp}" dampratio="{dr}"/>
    <position name="act_y" joint="slide_y" kp="{kp}" dampratio="{dr}"/>
    <position name="act_z" joint="slide_z" kp="{kp}" dampratio="{dr}"/>
  </actuator>

  <sensor>
    <!--
      Optional force sensor at the wrist site.
      LIMITATIONS:
      - Reports constraint force at the site; not a calibrated 6-axis F/T model.
      - No sensor noise, bias, or mechanical compliance of a real wrist sensor.
      - The FSM uses contact pair forces (mj_contactForce), not this sensor.
    -->
    <force name="wrist_force" site="wrist_site"/>
  </sensor>

  <keyframe>
    <key name="home" qpos="0 0 -0.05" ctrl="0 0 -0.05"/>
  </keyframe>
</mujoco>
"""
