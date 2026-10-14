"""Robot abstraction for peg-in-hole.

CartesianRobot maps world-frame tip targets to the 3 slide joints directly.
A future UR5eRobot can implement the same interface with inverse kinematics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import mujoco
import numpy as np

from .config import PegInHoleConfig
from .model import peg_tip_from_base


class RobotBase(ABC):
    """Minimal interface so the FSM does not depend on joint layout."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, cfg: PegInHoleConfig):
        self.model = model
        self.data = data
        self.cfg = cfg
        self.tip_site_id = model.site(cfg.peg_tip_site).id

    @abstractmethod
    def set_tip_target(self, xyz: np.ndarray) -> None:
        """Command the tip toward a world-frame XYZ target (meters)."""

    @abstractmethod
    def reset_joints(self, qpos: np.ndarray) -> None:
        """Teleport joints to qpos and zero velocities."""

    def tip_position(self) -> np.ndarray:
        return self.data.site_xpos[self.tip_site_id].copy()


class CartesianRobot(RobotBase):
    """3-axis gantry: slide_x, slide_y, slide_z with position actuators."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, cfg: PegInHoleConfig):
        super().__init__(model, data, cfg)
        self.joint_ids = [model.joint(n).id for n in cfg.joint_names]
        self.qpos_addrs = [model.jnt_qposadr[j] for j in self.joint_ids]
        self.dof_addrs = [model.jnt_dofadr[j] for j in self.joint_ids]
        self.act_ids = [model.actuator(n).id for n in cfg.actuator_names]
        self.base = np.array(cfg.robot_base_pos, dtype=float)
        # Constant Z offset from robot_base to tip when qz=0 (includes fixed body poses).
        self._tip_z_off = peg_tip_from_base(cfg)

        # Verify actuator ↔ joint wiring
        for aname, jname, aid in zip(cfg.actuator_names, cfg.joint_names, self.act_ids):
            jnt_id = int(model.actuator_trnid[aid, 0])
            expected = model.joint(jname).id
            if jnt_id != expected:
                raise RuntimeError(
                    f"Actuator '{aname}' drives joint id {jnt_id}, "
                    f"expected '{jname}' (id {expected})"
                )

    def world_to_qpos(self, xyz: np.ndarray) -> np.ndarray:
        """Convert world tip XYZ to (slide_x, slide_y, slide_z)."""
        qx = xyz[0] - self.base[0]
        qy = xyz[1] - self.base[1]
        qz = xyz[2] - self.base[2] - self._tip_z_off
        return np.array([qx, qy, qz], dtype=float)

    def qpos_to_tip_world(self, q: np.ndarray) -> np.ndarray:
        return np.array(
            [
                self.base[0] + q[0],
                self.base[1] + q[1],
                self.base[2] + q[2] + self._tip_z_off,
            ],
            dtype=float,
        )

    def set_tip_target(self, xyz: np.ndarray) -> None:
        q_des = self.world_to_qpos(xyz)
        # Clamp to joint ranges
        for i, jid in enumerate(self.joint_ids):
            lo, hi = self.model.jnt_range[jid]
            q_des[i] = float(np.clip(q_des[i], lo, hi))
        # Gravity compensation on the vertical slide so position actuators hold height.
        # ctrl bias ≈ m g / kp  (slide_z positive is +world Z, gravity pulls −Z).
        kp = float(self.cfg.actuator_kp)
        z_body = self.model.body("z_carriage").id
        z_mass = float(self.model.body_subtreemass[z_body])
        q_des[2] += (z_mass * abs(float(self.model.opt.gravity[2]))) / max(kp, 1.0)

        for aid, val in zip(self.act_ids, q_des):
            self.data.ctrl[aid] = val

    def reset_joints(self, qpos: np.ndarray) -> None:
        for addr, val in zip(self.qpos_addrs, qpos):
            self.data.qpos[addr] = val
        for addr in self.dof_addrs:
            self.data.qvel[addr] = 0.0
        for aid, val in zip(self.act_ids, qpos):
            self.data.ctrl[aid] = val
        mujoco.mj_forward(self.model, self.data)


class UR5eRobot(RobotBase):
    """Placeholder for a future 6-DOF UR5e replacement.

    Implement inverse kinematics in set_tip_target and map to six position
    actuators. The FSM only needs tip XYZ commands, so no controller changes
    are required beyond constructing this class instead of CartesianRobot.
    """

    def set_tip_target(self, xyz: np.ndarray) -> None:
        raise NotImplementedError(
            "UR5eRobot is a structural placeholder. Replace the Cartesian MJCF "
            "subtree and implement IK before using this class."
        )

    def reset_joints(self, qpos: np.ndarray) -> None:
        raise NotImplementedError("UR5eRobot is not implemented yet.")
