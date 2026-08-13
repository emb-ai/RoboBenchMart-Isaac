"""Generate the DexPilot hand URDFs for Sberchel from the robot MJCF.

The original hand stubs modelled each finger as ONE revolute joint with the tip
rigidly attached. That is not how the robot's finger moves: the real chain
continues through ``*_intermediate`` (and ``*_distal`` on the thumb), which is
what lets a fingertip curl toward the palm instead of sweeping a shallow arc.

Why that matters for retargeting: DexPilot matches fingertip geometry, so the
model's tip has to travel like the operator's. With the 1-DOF stub it did not,
and the consequence was measurable -- a frame rotation fitted on a flat open
hand left a closed fist 32.7 deg off, and the robot's fingers moved *opposite*
to the operator's. Modelling the full chain drops that to 12.2 deg.

pinocchio (which dex-retargeting builds on) ignores URDF ``<mimic>``, so the
intermediate joints are emitted as free DOFs. That is fine: the optimizer gets a
realistic finger shape to solve against, and ``SberchelDexRetargeter`` reads back
only the joints the robot can actually command, discarding the rest.

``--tip-length`` sets the intermediate->tip offset. A sweep showed the frame
residual is insensitive to it (12.1-13.3 deg over 1.1-5.0 cm) while the fingertip
workspace scales directly (5.6-13.2 cm), so it is a correspondence knob rather
than a physical measurement. The default puts the workspace near a human's ~11 cm
while staying plausible next to the 3.2 cm proximal segment.

Usage:
    python scripts/sberchel_build_dex_hand_urdf.py
    python scripts/sberchel_check_dex_retargeting.py    # verify the result
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "dsynth_isaac" / "assets" / "sberchel_xr"
_DEFAULT_GC = REPO.parent / "green_challenge"
GC = Path(os.environ.get("GREEN_CHALLENGE_PATH", _DEFAULT_GC))
MJCF = (
    GC
    / "assets/robot_v3_0a/dependencies/robots/v3_0_alpha_locomanipulator"
    / "v3_0_alpha_locomanipulator.xml"
)
DIGITS = ("index", "middle", "ring", "pinky")

# NOTE on digit length mismatch: a human's digits are unequal (pinky ~18% shorter
# than middle) while the robot's four digits are equal length. Reshaping THIS
# model to human proportions was tried and does not work -- the required pinky
# length is shorter than the joint chain itself, so the tip collapses onto the
# intermediate joint and that joint stops moving the tip at all. The mismatch is
# instead corrected on the INPUT side: ``SberchelDexRetargeter`` scales each
# operator fingertip to the model's open-pose length (``dex_human_open_cm``).


def _load(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"robot MJCF not found: {path}\nSet GREEN_CHALLENGE_PATH.")
    return path.read_text()


def _body(text: str, name: str):
    m = re.search(r'<body name="%s"([^>]*)>' % re.escape(name), text)
    if not m:
        raise SystemExit(f"body {name!r} missing from MJCF")
    attrs = m.group(1)
    pos = re.search(r'pos="([^"]*)"', attrs)
    quat = re.search(r'quat="([^"]*)"', attrs)
    P = np.array([float(x) for x in pos.group(1).split()]) if pos else np.zeros(3)
    if quat:
        w, x, y, z = (float(v) for v in quat.group(1).split())
        Q = R.from_quat([x, y, z, w])
    else:
        Q = R.identity()
    return P, Q


def _joint(text: str, name: str):
    m = re.search(r'<joint name="%s"([^>]*)/>' % re.escape(name), text)
    if not m:
        raise SystemExit(f"joint {name!r} missing from MJCF")
    attrs = m.group(1)
    axis = [float(v) for v in re.search(r'axis="([^"]*)"', attrs).group(1).split()]
    rng = [float(v) for v in re.search(r'range="([^"]*)"', attrs).group(1).split()]
    return axis, rng


def _xyz(v) -> str:
    return " ".join(f"{x:.6f}" for x in v)


def _rpy(Q) -> str:
    return " ".join(f"{x:.6f}" for x in Q.as_euler("xyz"))


def _revolute(name, parent, child, P, Q, axis, rng) -> list[str]:
    return [
        f'  <joint name="{name}" type="revolute">',
        f'    <parent link="{parent}"/><child link="{child}"/>',
        f'    <origin xyz="{_xyz(P)}" rpy="{_rpy(Q)}"/>',
        f'    <axis xyz="{axis[0]} {axis[1]} {axis[2]}"/>',
        f'    <limit lower="{rng[0]}" upper="{rng[1]}" effort="10" velocity="10"/>',
        "  </joint>",
    ]


def build(text: str, side: str, tip_length: float) -> str:
    base = f"{side}_hand_base_link"
    out = [f'<robot name="sberchel_{side}_hand">', f'  <link name="{base}"/>']

    for d in DIGITS:
        Pp, Qp = _body(text, f"{side}_{d}_proximal")
        Pi, Qi = _body(text, f"{side}_{d}_intermediate")
        ax_p, rg_p = _joint(text, f"{side}_{d}_proximal_joint")
        ax_i, rg_i = _joint(text, f"{side}_{d}_intermediate_joint")
        tip_dir = Pi / np.linalg.norm(Pi)
        out += [
            f'  <link name="{side}_{d}_proximal_link"/>',
            f'  <link name="{side}_{d}_intermediate_link"/>',
            f'  <link name="{side}_{d}_tip"/>',
        ]
        out += _revolute(f"{side}_{d}_proximal_joint", base, f"{side}_{d}_proximal_link",
                         Pp, Qp, ax_p, rg_p)
        out += _revolute(f"{side}_{d}_intermediate_joint", f"{side}_{d}_proximal_link",
                         f"{side}_{d}_intermediate_link", Pi, Qi, ax_i, rg_i)
        # The tip continues along the finger's own axis. Derive that direction
        # from the intermediate offset rather than hardcoding +y: the left hand's
        # fingers extend along -y and the right's along +y, so a fixed sign folds
        # one hand backwards (left span came out NEGATIVE before this).
        out += [
            f'  <joint name="{side}_{d}_tip_fixed" type="fixed">',
            f'    <parent link="{side}_{d}_intermediate_link"/><child link="{side}_{d}_tip"/>',
            f'    <origin xyz="{_xyz(tip_dir * tip_length)}" rpy="0 0 0"/>',
            "  </joint>",
        ]

    Pb, Qb = _body(text, f"{side}_thumb_proximal_base")
    Pp, Qp = _body(text, f"{side}_thumb_proximal")
    Pi, Qi = _body(text, f"{side}_thumb_intermediate")
    Pd, Qd = _body(text, f"{side}_thumb_distal")
    ax_y, rg_y = _joint(text, f"{side}_thumb_proximal_yaw_joint")
    ax_p, rg_p = _joint(text, f"{side}_thumb_proximal_pitch_joint")
    ax_i, rg_i = _joint(text, f"{side}_thumb_intermediate_joint")
    out += [
        f'  <link name="{side}_thumb_yaw_link"/>',
        f'  <link name="{side}_thumb_proximal_link"/>',
        f'  <link name="{side}_thumb_intermediate_link"/>',
        f'  <link name="{side}_thumb_tip"/>',
    ]
    out += _revolute(f"{side}_thumb_proximal_yaw_joint", base, f"{side}_thumb_yaw_link",
                     Pb, Qb, ax_y, rg_y)
    out += _revolute(f"{side}_thumb_proximal_pitch_joint", f"{side}_thumb_yaw_link",
                     f"{side}_thumb_proximal_link", Pp, Qp, ax_p, rg_p)
    out += _revolute(f"{side}_thumb_intermediate_joint", f"{side}_thumb_proximal_link",
                     f"{side}_thumb_intermediate_link", Pi, Qi, ax_i, rg_i)
    out += [
        f'  <joint name="{side}_thumb_tip_fixed" type="fixed">',
        f'    <parent link="{side}_thumb_intermediate_link"/><child link="{side}_thumb_tip"/>',
        f'    <origin xyz="{_xyz(Pd + Qd.apply(np.array([0.018, 0.0, 0.0])))}" rpy="0 0 0"/>',
        "  </joint>",
        "</robot>",
    ]
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tip-length", type=float, default=0.032,
                    help="intermediate->tip offset in metres (default 0.032)")
    ap.add_argument("--out-dir", type=Path, default=ASSETS)
    args = ap.parse_args()

    text = _load(MJCF)
    for side in ("left", "right"):
        path = args.out_dir / f"sberchel_{side}_hand.urdf"
        path.write_text(build(text, side, args.tip_length))
        print(f"wrote {path}")
    print("\nNow run: python scripts/sberchel_check_dex_retargeting.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
