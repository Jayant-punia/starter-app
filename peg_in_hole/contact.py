"""Contact sensing utilities using MuJoCo's contact API.

IMPORTANT: This module uses actual MuJoCo contact pairs and mj_contactForce.
It does NOT invent or approximate force-torque readings. An optional wrist
force sensor exists in the MJCF for comparison/education only.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .config import PegInHoleConfig


@dataclass
class ContactInfo:
    n_contacts: int
    force_norm: float  # sum of ||force|| over peg contacts (Newtons)
    max_force: float
    geom_names: list[str]


def peg_contact_info(model: mujoco.MjModel, data: mujoco.MjData, cfg: PegInHoleConfig) -> ContactInfo:
    """Summarize contacts involving the peg geom."""
    peg_id = model.geom(cfg.peg_geom_name).id
    force_sum = 0.0
    max_force = 0.0
    names: list[str] = []
    n_peg = 0
    force6 = np.zeros(6, dtype=np.float64)

    for i in range(data.ncon):
        con = data.contact[i]
        if con.geom1 != peg_id and con.geom2 != peg_id:
            continue
        n_peg += 1
        other = con.geom2 if con.geom1 == peg_id else con.geom1
        names.append(model.geom(other).name or f"geom_{other}")
        mujoco.mj_contactForce(model, data, i, force6)
        # force6[:3] is force in the contact frame (normal along x)
        fn = float(np.linalg.norm(force6[:3]))
        force_sum += fn
        max_force = max(max_force, fn)

    return ContactInfo(
        n_contacts=n_peg,
        force_norm=force_sum,
        max_force=max_force,
        geom_names=names,
    )


def wrist_force_sensor(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray | None:
    """Read the optional MuJoCo force sensor if present.

    Returns a 3-vector in the sensor frame, or None if the sensor is missing.
    This is provided for inspection only — see MJCF comments for limitations.
    """
    try:
        sid = model.sensor("wrist_force").id
    except KeyError:
        return None
    adr = model.sensor_adr[sid]
    dim = model.sensor_dim[sid]
    return data.sensordata[adr : adr + dim].copy()
