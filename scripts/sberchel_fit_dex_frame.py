"""Measure the operator->robot hand rotation for DexPilot retargeting.

``_OPERATOR2MANO`` in ``sberchel_retargeter.py`` maps the operator's hand into
the frame the robot's hand URDF is expressed in. It was copied from IsaacLab's
GR1T2 helper, and GR1's hand frame is not Sberchel's -- our own wrist path needs
a 120 deg offset to reach ``hand_base_link``. A wrong matrix here rotates the
landmarks into a frame the solver does not expect, which shows up as output that
depends on hand *orientation* rather than curl, flipping between full closure and
sensible poses.

Rather than guess candidate matrices, this measures it. Hold a flat open hand in
XR; the retargeter prints ``TIPFIT <side> <15 floats>`` -- the five fingertip unit
vectors in the operator frame. This script Kabsch-fits those onto the robot's own
open-hand fingertip directions, which is exactly the rotation that was missing.

Usage:
    # capture: run the recorder with --retargeter dex, hold a flat open hand,
    # copy one TIPFIT line per hand out of the log, then:
    python scripts/sberchel_fit_dex_frame.py \\
        --line "TIPFIT right +0.1 -0.2 ... " \\
        --line "TIPFIT left  +0.3 +0.1 ... "

Fit quality matters as much as the matrix: a residual much above ~15 deg means
the two hands do not differ by a rigid rotation alone, and no single matrix will
fix the retargeting.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

ASSETS = Path(__file__).resolve().parents[1] / "dsynth_isaac" / "assets" / "sberchel_xr"
DIGITS = ("thumb", "index", "middle", "ring", "pinky")
HANDS = {
    "left": ("sberchel_hand_left_dexpilot.yml", "sberchel_left_hand.urdf", "left_hand_base_link"),
    "right": ("sberchel_hand_right_dexpilot.yml", "sberchel_right_hand.urdf", "right_hand_base_link"),
}


def robot_tip_units(work: Path, side: str, pose: str) -> np.ndarray:
    """Robot fingertip unit vectors from the wrist, at the open or fist pose."""
    from dex_retargeting.retargeting_config import RetargetingConfig

    cfg_name, urdf_name, base_link = HANDS[side]
    cfg_path = work / cfg_name
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["retargeting"]["urdf_path"] = str((work / urdf_name).resolve())
    cfg_path.write_text(yaml.dump(cfg))

    robot = RetargetingConfig.load_from_file(str(cfg_path)).build().optimizer.robot
    limits = np.array([(l, h) for l, h in robot.joint_limits])
    robot.compute_forward_kinematics(limits[:, 0] if pose == "open" else limits[:, 1])
    origin = robot.get_link_pose(robot.get_link_index(base_link))[:3, 3]

    out = []
    for digit in DIGITS:
        tip = robot.get_link_pose(robot.get_link_index(f"{side}_{digit}_tip"))[:3, 3] - origin
        out.append(tip / np.linalg.norm(tip))
    return np.asarray(out)


def kabsch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation R minimising |a @ R - b|, both (N, 3) of unit vectors."""
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return u @ np.diag([1.0, 1.0, d]) @ vt


def parse_line(line: str) -> tuple[str, str, np.ndarray]:
    """Parse ``[open|fist] TIPFIT <side> <15 floats>``; pose defaults to open."""
    text = line.strip()
    pose = "open"
    for tag in ("open", "fist"):
        if text.lower().startswith(tag):
            pose, text = tag, text[len(tag):]
            break
    parts = text.replace("TIPFIT", " ").split()
    side = parts[0].lower()
    if side not in HANDS:
        raise SystemExit(f"unknown side {side!r}; expected left or right")
    vals = np.array([float(x) for x in parts[1:]], dtype=float)
    if vals.size != 15:
        raise SystemExit(f"{side}: expected 15 floats (5 tips x xyz), got {vals.size}")
    return side, pose, vals.reshape(5, 3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--line", action="append", required=True,
                    help='A TIPFIT line from the log, optionally prefixed "open " or '
                         '"fist " (default open). Repeat: several captures per pose per hand.')
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for f in ASSETS.iterdir():
            shutil.copy2(f, work / f.name)

        captures: dict[tuple[str, str], list[np.ndarray]] = {}
        for line in args.line:
            side, pose, human = parse_line(line)
            captures.setdefault((side, pose), []).append(human)

        results = {}
        for side in sorted({s for s, _ in captures}):
            poses = [p for p in ("open", "fist") if (side, p) in captures]
            if "fist" not in poses:
                print(f"--- {side} hand")
                print("  WARNING: no fist capture. A rotation fitted on the open pose alone")
                print("  does not constrain which way the fingers travel as they curl, and")
                print("  has previously produced fingers that moved BACKWARDS. Capture a")
                print("  fist too and pass it with --pose fist.\n")

            human_rows, robot_rows = [], []
            for pose in poses:
                stack = np.stack(captures[(side, pose)])
                mean = stack.mean(axis=0)
                human_rows.append(mean / np.linalg.norm(mean, axis=1, keepdims=True))
                robot_rows.append(robot_tip_units(work, side, pose))
            human = np.vstack(human_rows)
            robot = np.vstack(robot_rows)
            R = kabsch(human, robot)

            aligned = human @ R
            angles = np.degrees(np.arccos(np.clip(
                np.einsum("ij,ij->i", aligned, robot), -1.0, 1.0)))
            results[side] = (R, angles)

            print(f"--- {side} hand")
            for k, pose in enumerate(poses):
                n = len(captures[(side, pose)])
                seg = angles[k * 5:(k + 1) * 5]
                note = "  <-- fit from 1 frame is noisy; capture several" if n < 3 else ""
                print(f"  {pose:5s} ({n} capture{'s' if n != 1 else ''}): "
                      f"mean {seg.mean():5.1f} deg, worst {seg.max():5.1f} deg{note}")
                for digit, ang in zip(DIGITS, seg):
                    print(f"      {digit:7s} {ang:6.1f}")
            print(f"  overall worst {angles.max():.1f} deg, mean {angles.mean():.1f} deg")
            print("  fitted operator->robot rotation:")
            for row in R:
                print("    [" + ", ".join(f"{v:+.6f}" for v in row) + "],")
            print()

    worst = max(a.max() for _, a in results.values())
    if worst > 8.0:
        print(f"RESULT: poor fit ({worst:.1f} deg worst residual).")
        print("The two hands do not differ by a rigid rotation alone, so replacing")
        print("_OPERATOR2MANO will not by itself fix the retargeting. Check that the")
        print("captured pose really was a flat open hand before concluding.")
        return 1
    print(f"RESULT: good fit ({worst:.1f} deg worst residual).")
    print("Paste the matrices into _OPERATOR2MANO in sberchel_retargeter.py, then")
    print("re-run the recorder with --retargeter dex and watch whether an open hand")
    print("now yields open joint angles instead of 1.70.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
