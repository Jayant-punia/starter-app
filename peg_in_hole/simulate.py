"""Run the interactive peg-in-hole MuJoCo simulation.

Usage:
    python -m peg_in_hole
    python peg_in_hole.py
"""

from __future__ import annotations

import sys
import time
import traceback

import mujoco
import mujoco.viewer
import numpy as np

from .config import DEFAULT_CONFIG, PegInHoleConfig
from .contact import wrist_force_sensor
from .controller import PegInHoleController, Stage
from .model import build_mjcf, peg_tip_from_base
from .robot import CartesianRobot


def validate_model(model: mujoco.MjModel, cfg: PegInHoleConfig) -> None:
    """Check joints, actuators, geoms, and sites exist and are wired correctly."""
    errors: list[str] = []
    for name in cfg.joint_names:
        try:
            j = model.joint(name)
            if j.type != mujoco.mjtJoint.mjJNT_SLIDE:
                errors.append(f"Joint '{name}' is not a slide joint")
        except KeyError:
            errors.append(f"Missing joint '{name}'")

    for aname, jname in zip(cfg.actuator_names, cfg.joint_names):
        try:
            act = model.actuator(aname)
            jid = int(model.actuator_trnid[act.id, 0])
            if jid != model.joint(jname).id:
                errors.append(
                    f"Actuator '{aname}' not attached to joint '{jname}' "
                    f"(got id {jid})"
                )
        except KeyError:
            errors.append(f"Missing actuator '{aname}'")

    for gname in (cfg.peg_geom_name, "wall_yp", "wall_ym", "wall_xp", "wall_xm"):
        try:
            model.geom(gname)
        except KeyError:
            errors.append(f"Missing geom '{gname}'")

    for sname in (cfg.peg_tip_site, "wrist_site", "hole_center"):
        try:
            model.site(sname)
        except KeyError:
            errors.append(f"Missing site '{sname}'")

    # Peg radius sanity
    peg = model.geom(cfg.peg_geom_name)
    if abs(float(peg.size[0]) - cfg.peg_radius) > 1e-9:
        errors.append(
            f"Peg radius mismatch: geom={peg.size[0]} cfg={cfg.peg_radius}"
        )

    tip_off = peg_tip_from_base(cfg)
    if abs(tip_off - cfg.tip_z_offset_from_z_joint) > 1e-6:
        print(
            f"Note: computed tip_z_offset_from_base={tip_off:.6f}, "
            f"config field={cfg.tip_z_offset_from_z_joint:.6f} "
            f"(using computed value)."
        )

    if errors:
        raise RuntimeError("MJCF validation failed:\n  - " + "\n  - ".join(errors))


def load_sim(cfg: PegInHoleConfig | None = None) -> tuple[mujoco.MjModel, mujoco.MjData, PegInHoleConfig]:
    cfg = cfg or DEFAULT_CONFIG
    # Keep documented tip offset field in sync with MJCF geometry
    cfg.tip_z_offset_from_z_joint = peg_tip_from_base(cfg)

    xml = build_mjcf(cfg)
    try:
        model = mujoco.MjModel.from_xml_string(xml)
    except Exception as exc:
        raise RuntimeError(f"Failed to compile MJCF: {exc}") from exc
    data = mujoco.MjData(model)
    validate_model(model, cfg)
    return model, data, cfg


def run(cfg: PegInHoleConfig | None = None) -> list:
    """Launch viewer and run insertion attempts. Returns attempt logs."""
    model, data, cfg = load_sim(cfg)
    robot = CartesianRobot(model, data, cfg)
    controller = PegInHoleController(robot, cfg)

    print("=" * 72)
    print("peg_in_hole — MuJoCo Cartesian insertion baseline")
    print("=" * 72)
    print(f"  peg radius     : {cfg.peg_radius * 1e3:.2f} mm")
    print(f"  hole clearance : {cfg.clearance * 1e3:.2f} mm  (hole r={cfg.hole_radius*1e3:.2f} mm)")
    print(f"  offset range   : ±{cfg.offset_range * 1e3:.1f} mm")
    print(f"  contact thresh : {cfg.contact_force_threshold:.1f} N  (from mj_contactForce)")
    print(f"  max time       : {cfg.max_insertion_time:.1f} s")
    print(f"  attempts       : {cfg.num_attempts}")
    print()
    print("Contact feedback: MuJoCo contact pairs + mj_contactForce (real solver forces).")
    print("Optional wrist_force sensor is available for inspection but is NOT used by the FSM.")
    print(cfg.notes)
    print("-" * 72)

    logs = []
    attempt = 1
    controller.start_attempt(data.time, attempt)
    print(controller.status_line(data.time))
    last_print = data.time
    done_pause_until = None

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Camera aimed at the hole
            viewer.cam.lookat[:] = [*cfg.hole_center_xy, cfg.plate_top_z]
            viewer.cam.distance = 0.55
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25

            while viewer.is_running():
                step_start = time.time()

                if done_pause_until is not None:
                    if data.time >= done_pause_until:
                        logs.append(controller.log)
                        _print_attempt_summary(controller.log)
                        if attempt >= cfg.num_attempts:
                            print("\nAll attempts finished. Close the viewer to exit.")
                            # Hold final pose; still step physics lightly
                            done_pause_until = None
                            attempt = cfg.num_attempts + 1
                        else:
                            attempt += 1
                            controller.start_attempt(data.time, attempt)
                            print(controller.status_line(data.time))
                            done_pause_until = None
                    else:
                        mujoco.mj_step(model, data)
                        viewer.sync()
                        if cfg.realtime:
                            _sleep_realtime(model, step_start)
                        continue

                if attempt <= cfg.num_attempts:
                    stage = controller.update(data.time)
                    mujoco.mj_step(model, data)

                    if stage in (Stage.SUCCESS, Stage.FAILURE) and done_pause_until is None:
                        print(controller.status_line(data.time))
                        print(
                            f">>> {stage.name} in {controller.log.completion_time:.2f}s "
                            f"(contacts={controller.log.contact_events})"
                        )
                        # Brief pause so the user can see the final pose
                        done_pause_until = data.time + 1.5
                else:
                    mujoco.mj_step(model, data)

                if data.time - last_print >= cfg.status_print_interval:
                    if attempt <= cfg.num_attempts and controller.stage not in (
                        Stage.SUCCESS,
                        Stage.FAILURE,
                    ):
                        # Show optional sensor reading for education (not used by FSM)
                        wf = wrist_force_sensor(model, data)
                        extra = ""
                        if wf is not None:
                            extra = f"  wrist_sensor‖F‖={np.linalg.norm(wf):.2f}N(unused)"
                        print(controller.status_line(data.time) + extra)
                    last_print = data.time

                viewer.sync()
                if cfg.realtime:
                    _sleep_realtime(model, step_start)

    except Exception:
        print("Simulation aborted with an error:", file=sys.stderr)
        traceback.print_exc()
        raise

    # If viewer closed mid-attempt, still return collected logs
    if controller.log is not None and controller.log not in logs:
        if controller.stage in (Stage.SUCCESS, Stage.FAILURE):
            logs.append(controller.log)

    _print_session_summary(logs)
    return logs


def _sleep_realtime(model: mujoco.MjModel, step_start: float) -> None:
    remaining = model.opt.timestep - (time.time() - step_start)
    if remaining > 0:
        time.sleep(remaining)


def _print_attempt_summary(log) -> None:
    if log is None:
        return
    result = "SUCCESS" if log.success else "FAILURE"
    print(
        f"\n--- Attempt {log.attempt_id} {result} ---\n"
        f"  initial offset : ({log.initial_offset[0]*1e3:+.2f}, {log.initial_offset[1]*1e3:+.2f}) mm\n"
        f"  completion time: {log.completion_time:.3f} s\n"
        f"  contact events : {log.contact_events}\n"
        f"  final tip      : ({log.tip_final[0]:.4f}, {log.tip_final[1]:.4f}, {log.tip_final[2]:.4f})\n"
        f"  notes          : {'; '.join(log.notes)}"
    )


def _print_session_summary(logs: list) -> None:
    if not logs:
        return
    n_ok = sum(1 for L in logs if L and L.success)
    print("\n" + "=" * 72)
    print(f"Session summary: {n_ok}/{len(logs)} successful insertions")
    for L in logs:
        if L is None:
            continue
        print(
            f"  #{L.attempt_id}: {'OK' if L.success else 'FAIL':4}  "
            f"t={L.completion_time:5.2f}s  "
            f"offset=({L.initial_offset[0]*1e3:+.1f},{L.initial_offset[1]*1e3:+.1f})mm  "
            f"contacts={L.contact_events}"
        )
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    cfg = PegInHoleConfig()
    # Lightweight CLI overrides
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--no-realtime":
            cfg.realtime = False
        elif arg == "--attempts" and i + 1 < len(argv):
            i += 1
            cfg.num_attempts = int(argv[i])
        elif arg == "--clearance" and i + 1 < len(argv):
            i += 1
            cfg.clearance = float(argv[i])
        elif arg == "--offset" and i + 1 < len(argv):
            i += 1
            cfg.offset_range = float(argv[i])
        elif arg == "--seed" and i + 1 < len(argv):
            i += 1
            cfg.seed = int(argv[i])
        elif arg == "--max-time" and i + 1 < len(argv):
            i += 1
            cfg.max_insertion_time = float(argv[i])
        elif arg in ("-h", "--help"):
            print(__doc__)
            print(
                "Options:\n"
                "  --no-realtime       Run as fast as possible\n"
                "  --attempts N        Number of insertion trials (default 3)\n"
                "  --clearance M       Hole radial clearance in meters (default 0.0005)\n"
                "  --offset M          Max |lateral| initial offset in meters\n"
                "  --seed N            RNG seed for offsets\n"
                "  --max-time S        Max sim seconds per attempt\n"
            )
            return 0
        else:
            print(f"Unknown argument: {arg}", file=sys.stderr)
            return 2
        i += 1

    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())