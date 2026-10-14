"""Finite-state peg-in-hole insertion controller.

Stages: APPROACH → DESCEND → SEARCH ↔ INSERT → SUCCESS | FAILURE

Uses MuJoCo contact forces (mj_contactForce) for collision feedback. Does not
fake force-torque sensing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from .config import PegInHoleConfig
from .contact import ContactInfo, peg_contact_info
from .robot import CartesianRobot


class Stage(Enum):
    APPROACH = auto()
    DESCEND = auto()
    SEARCH = auto()
    INSERT = auto()
    SUCCESS = auto()
    FAILURE = auto()


@dataclass
class AttemptLog:
    attempt_id: int
    initial_offset: tuple[float, float]
    success: bool = False
    completion_time: float = 0.0
    contact_events: int = 0
    final_stage: str = ""
    tip_final: tuple[float, float, float] = (0.0, 0.0, 0.0)
    notes: list[str] = field(default_factory=list)


class PegInHoleController:
    def __init__(self, robot: CartesianRobot, cfg: PegInHoleConfig):
        self.robot = robot
        self.cfg = cfg
        self.stage = Stage.APPROACH
        self.target = np.zeros(3)
        self.spiral_angle = 0.0
        self.spiral_radius = 0.0
        self.t0 = 0.0
        self.attempt_id = 0
        self.offset = np.zeros(2)
        self.log: AttemptLog | None = None
        self._last_contacting = False
        self._search_clear_timer = 0.0
        self.rng = np.random.default_rng(cfg.seed)

    @property
    def hole_xy(self) -> np.ndarray:
        return np.array(self.cfg.hole_center_xy, dtype=float)

    @property
    def plate_top(self) -> float:
        return self.cfg.plate_top_z

    def start_attempt(self, sim_time: float, attempt_id: int) -> AttemptLog:
        """Randomize lateral offset and reset FSM for a new trial."""
        self.attempt_id = attempt_id
        self.stage = Stage.APPROACH
        self.spiral_angle = 0.0
        self.spiral_radius = 0.0
        self.t0 = sim_time
        self._last_contacting = False
        self._search_clear_timer = 0.0

        ox = float(self.rng.uniform(-self.cfg.offset_range, self.cfg.offset_range))
        oy = float(self.rng.uniform(-self.cfg.offset_range, self.cfg.offset_range))
        self.offset = np.array([ox, oy])

        tip_z = self.plate_top + self.cfg.approach_height_above_plate
        start_tip = np.array(
            [self.hole_xy[0] + ox, self.hole_xy[1] + oy, tip_z], dtype=float
        )
        q = self.robot.world_to_qpos(start_tip)
        self.robot.reset_joints(q)
        self.target = start_tip.copy()
        self._search_xy = start_tip[:2].copy()

        self.log = AttemptLog(
            attempt_id=attempt_id,
            initial_offset=(ox, oy),
        )
        return self.log

    def _elapsed(self, sim_time: float) -> float:
        return sim_time - self.t0

    def _record_contact(self, info: ContactInfo) -> None:
        contacting = info.n_contacts > 0 and info.force_norm > 0.05
        if contacting and not self._last_contacting and self.log is not None:
            self.log.contact_events += 1
            self.log.notes.append(
                f"contact#{self.log.contact_events}: F={info.force_norm:.2f}N "
                f"with {info.geom_names}"
            )
        self._last_contacting = contacting

    def _approach_target(self) -> np.ndarray:
        z = self.plate_top + self.cfg.approach_height_above_plate
        # Keep the randomized lateral offset (simulates pose uncertainty).
        # SEARCH is responsible for finding the true hole center.
        return np.array(
            [self.hole_xy[0] + self.offset[0], self.hole_xy[1] + self.offset[1], z],
            dtype=float,
        )

    def update(self, sim_time: float) -> Stage:
        cfg = self.cfg
        tip = self.robot.tip_position()
        info = peg_contact_info(self.robot.model, self.robot.data, cfg)
        self._record_contact(info)
        dt = cfg.timestep
        elapsed = self._elapsed(sim_time)

        if self.stage not in (Stage.SUCCESS, Stage.FAILURE) and elapsed > cfg.max_insertion_time:
            self.stage = Stage.FAILURE
            self._finalize(False, elapsed, tip, "timeout")
            return self.stage

        if self.stage == Stage.APPROACH:
            self.target = self._approach_target()
            xy_ok = np.linalg.norm(tip[:2] - self.target[:2]) < 0.002
            # Allow small gravity sag; require tip near/above approach height.
            z_ok = tip[2] > self.target[2] - 0.01
            if xy_ok and z_ok:
                self.stage = Stage.DESCEND
                self.target = self._approach_target()
                self.target[2] = tip[2]

        elif self.stage == Stage.DESCEND:
            # Lower while holding the (possibly misaligned) approach XY.
            # Integrate setpoint (do not re-anchor to tip each step).
            self.target[0] = self.hole_xy[0] + self.offset[0]
            self.target[1] = self.hole_xy[1] + self.offset[1]
            self.target[2] -= cfg.descend_speed * dt
            blocked = info.force_norm >= cfg.contact_force_threshold
            depth = self.plate_top - tip[2]
            if blocked and depth < cfg.success_insertion_depth * 0.5:
                self.stage = Stage.SEARCH
                self.spiral_angle = 0.0
                self.spiral_radius = 0.0
                # Hold just above the contact height while searching
                self.target[2] = max(tip[2], self.plate_top + 0.002)
            elif depth >= cfg.success_insertion_depth:
                self.stage = Stage.SUCCESS
                self._finalize(True, elapsed, tip, "inserted during descend")
            elif tip[2] < self.plate_top - 0.002 and info.force_norm < cfg.contact_force_threshold * 0.5:
                self.stage = Stage.INSERT

        elif self.stage == Stage.SEARCH:
            self.spiral_angle += cfg.spiral_omega * dt
            self.spiral_radius = min(
                cfg.spiral_pitch * self.spiral_angle, cfg.max_spiral_radius
            )
            cx, cy = self.hole_xy
            self.target[0] = cx + self.spiral_radius * np.cos(self.spiral_angle)
            self.target[1] = cy + self.spiral_radius * np.sin(self.spiral_angle)
            # Light downward pressure; stay on the plate while hunting
            self.target[2] = min(self.target[2], self.plate_top + 0.003)
            self.target[2] -= 0.003 * dt

            if info.force_norm < cfg.contact_force_threshold * 0.4:
                self._search_clear_timer += dt
            else:
                self._search_clear_timer = 0.0

            # Also treat geometric alignment as clearance to insert
            aligned = np.linalg.norm(tip[:2] - self.hole_xy) < (cfg.clearance + 0.0005)
            if self._search_clear_timer >= cfg.search_clear_time or aligned:
                self.stage = Stage.INSERT
                self._search_clear_timer = 0.0
                self.target[:2] = tip[:2]
                self.target[2] = tip[2]

        elif self.stage == Stage.INSERT:
            # Pull XY toward hole center while lowering
            self.target[0] = tip[0] + 0.35 * (self.hole_xy[0] - tip[0])
            self.target[1] = tip[1] + 0.35 * (self.hole_xy[1] - tip[1])
            self.target[2] -= cfg.insert_speed * dt

            if info.force_norm >= cfg.contact_force_threshold * 2.0:
                self.stage = Stage.SEARCH
                self.spiral_angle = max(self.spiral_angle, 0.0)
                self.spiral_radius = min(max(self.spiral_radius, 0.001), cfg.max_spiral_radius)
                self.target[2] = max(tip[2], self.plate_top + 0.002)
                self._search_clear_timer = 0.0
            elif self.plate_top - tip[2] >= cfg.success_insertion_depth:
                self.stage = Stage.SUCCESS
                self._finalize(True, elapsed, tip, "inserted")

        self.robot.set_tip_target(self.target)
        return self.stage

    def _finalize(self, success: bool, elapsed: float, tip: np.ndarray, note: str) -> None:
        if self.log is None:
            return
        self.log.success = success
        self.log.completion_time = elapsed
        self.log.final_stage = self.stage.name
        self.log.tip_final = (float(tip[0]), float(tip[1]), float(tip[2]))
        self.log.notes.append(note)

    def status_line(self, sim_time: float) -> str:
        tip = self.robot.tip_position()
        info = peg_contact_info(self.robot.model, self.robot.data, self.cfg)
        off = self.offset
        return (
            f"[attempt {self.attempt_id}] stage={self.stage.name:<8} "
            f"t={self._elapsed(sim_time):5.2f}s  "
            f"tip=({tip[0]:+.4f},{tip[1]:+.4f},{tip[2]:+.4f})  "
            f"offset=({off[0]*1e3:+.1f},{off[1]*1e3:+.1f})mm  "
            f"contacts={info.n_contacts} F={info.force_norm:5.2f}N"
        )
