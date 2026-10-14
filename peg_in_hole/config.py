"""Configuration constants for the peg-in-hole MuJoCo simulation.

Tune these values without changing controller logic. Units are SI (meters, seconds, Newtons).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PegInHoleConfig:
    # --- Geometry ---
    peg_radius: float = 0.010  # ~10 mm
    peg_half_length: float = 0.040  # total peg length = 80 mm
    # Radial clearance: hole_radius = peg_radius + clearance
    clearance: float = 0.0008  # 0.8 mm (tight but solvable with contact search)
    hole_plate_half_height: float = 0.020
    hole_outer_half: float = 0.080
    hole_center_xy: tuple[float, float] = (0.25, 0.0)
    hole_plate_z: float = 0.080  # plate body center height

    # --- Robot mount (Cartesian base frame origin in world) ---
    robot_base_pos: tuple[float, float, float] = (0.25, 0.0, 0.55)
    # Tip Z offset from robot_base when slides are at 0 (updated on load from MJCF)
    tip_z_offset_from_z_joint: float = -0.220

    # --- Initial misalignment (uniform random lateral offset) ---
    offset_range: float = 0.008  # ±8 mm in X/Y
    seed: int | None = None  # None => nondeterministic

    # --- Motion / FSM ---
    approach_height_above_plate: float = 0.080  # tip height above plate top
    descend_speed: float = 0.030  # m/s while descending / inserting
    search_speed: float = 0.020  # m/s lateral during SEARCH
    insert_speed: float = 0.020  # m/s after alignment
    spiral_pitch: float = 0.0008  # radial growth per radian
    spiral_omega: float = 4.0  # rad/s for spiral search
    max_spiral_radius: float = 0.012
    success_insertion_depth: float = 0.025  # tip below plate top
    max_insertion_time: float = 12.0  # seconds of sim time per attempt

    # Contact threshold using MuJoCo contact forces (NOT a fake F/T sensor).
    # When the normal force magnitude on the peg exceeds this during DESCEND,
    # the controller transitions to SEARCH.
    contact_force_threshold: float = 1.0  # Newtons (sum of contact force norms)
    # Hysteresis: leave SEARCH only when force stays low for this long
    search_clear_time: float = 0.15

    # --- Position actuator gains ---
    actuator_kp: float = 2000.0
    actuator_dampratio: float = 1.0

    # --- Simulation ---
    timestep: float = 0.002
    realtime: bool = True
    status_print_interval: float = 0.25  # wall / sim print cadence (sim seconds)
    num_attempts: int = 3  # consecutive trials in one viewer session

    # Joint / actuator names (must match MJCF)
    joint_names: tuple[str, str, str] = ("slide_x", "slide_y", "slide_z")
    actuator_names: tuple[str, str, str] = ("act_x", "act_y", "act_z")
    peg_geom_name: str = "peg"
    peg_tip_site: str = "peg_tip"
    hole_body_name: str = "hole_fixture"

    notes: str = field(
        default=(
            "Contact forces come from MuJoCo's contact solver via mj_contactForce. "
            "An optional wrist force *sensor* site is included for education only; "
            "the FSM does not rely on it. A true force/torque sensor would need a "
            "compliant element and still reports forces in the site frame with "
            "filtering/noise characteristics that differ from hardware."
        )
    )

    @property
    def hole_radius(self) -> float:
        return self.peg_radius + self.clearance

    @property
    def plate_top_z(self) -> float:
        return self.hole_plate_z + self.hole_plate_half_height


DEFAULT_CONFIG = PegInHoleConfig()
