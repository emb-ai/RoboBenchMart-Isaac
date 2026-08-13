"""Offline consistency check for recorded Sberchel XR demos.

Verifies, purely from the HDF5 and with no simulator involved, that what the
dataset claims as the commanded action is what the simulator actually applied.

The specific failure this guards against: ``PinkJointHold`` bypasses the Pink IK
term's ``apply_actions`` during locomotion, which also bypasses the assignment to
``_processed_actions``. Without the fix in that class, IsaacLab's
``PostStepProcessedActionsRecorder`` logs a stale vector from the last real IK
solve for the whole locomotion segment -- so the dataset and the physics disagree
exactly where the grasp lives.

Both quantities are recorded post-step, so for the joints the upper-body term
controls they must be equal to floating-point noise:

    processed_actions[legs + neck :]   ==   joint_pos_target[:, controlled_ids]

Results are split by locomotion vs static, because the bug was locomotion-only --
a clean static half with a broken locomotion half is the signature.

Usage:
    python scripts/sberchel_check_dataset.py datasets/<file>.hdf5
"""

from __future__ import annotations

import argparse
import json
import sys

import h5py
import numpy as np

LEGS_DIM = 12
NECK_DIM = 2
TOL = 1e-4  # float32 round-trip through HDF5


def _resolve_controlled_ids(env_args: dict) -> tuple[list[int], list[str]]:
    joint_names = env_args.get("sberchel_joint_names")
    controlled = env_args.get("sberchel_upper_controlled_joint_names")
    if not joint_names or not controlled:
        raise SystemExit(
            "Dataset is missing joint-name provenance (sberchel_joint_names / "
            "sberchel_upper_controlled_joint_names). It was recorded before the "
            "provenance recorder existed; re-record before checking."
        )
    index = {name: i for i, name in enumerate(joint_names)}
    return [index[n] for n in controlled], controlled


def check_episode(ep, controlled_ids: list[int], names: list[str]) -> dict:
    processed = np.asarray(ep["processed_actions"])
    targets = np.asarray(ep["joint_targets/joint_pos_target"])
    upper = processed[:, LEGS_DIM + NECK_DIM :]
    applied = targets[:, controlled_ids]

    n = min(len(upper), len(applied))
    diff = np.abs(upper[:n] - applied[:n])

    state = ep.get("teleop_state")
    if state is not None:
        is_loco = np.asarray(state)[:n, 0] > 0.5
    else:
        is_loco = np.zeros(n, dtype=bool)

    per_joint = diff.max(axis=0)
    return {
        "steps": n,
        "max_abs": float(diff.max()) if n else 0.0,
        "bad_steps": int((diff.max(axis=1) > TOL).sum()),
        "loco_max": float(diff[is_loco].max()) if is_loco.any() else None,
        "static_max": float(diff[~is_loco].max()) if (~is_loco).any() else None,
        "loco_steps": int(is_loco.sum()),
        "worst_joint": names[int(per_joint.argmax())] if n else "-",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Path to a recorded .hdf5 demo file")
    args = parser.parse_args()

    with h5py.File(args.dataset, "r") as f:
        data = f["data"]
        env_args = json.loads(data.attrs["env_args"])
        controlled_ids, names = _resolve_controlled_ids(env_args)

        print(f"dataset : {args.dataset}")
        print(f"episodes: {len(data.keys())}")
        print(f"gains   : legacy_stiff_teleop={env_args.get('sberchel_legacy_stiff_teleop')} "
              f"squeeze_scale={env_args.get('sberchel_squeeze_scale')} "
              f"gravity_comp={env_args.get('sberchel_gravity_compensation')}")
        print(f"upper joints controlled: {len(controlled_ids)}")
        print()

        failed = 0
        for name in sorted(data.keys()):
            r = check_episode(data[name], controlled_ids, names)
            ok = r["bad_steps"] == 0
            failed += not ok
            loco = "n/a" if r["loco_max"] is None else f"{r['loco_max']:.2e}"
            static = "n/a" if r["static_max"] is None else f"{r['static_max']:.2e}"
            print(
                f"{'PASS' if ok else 'FAIL'} {name:16s} steps={r['steps']:6d} "
                f"loco={r['loco_steps']:6d}  max|diff|={r['max_abs']:.2e} "
                f"(loco {loco} / static {static})  bad_steps={r['bad_steps']}"
                + ("" if ok else f"  worst={r['worst_joint']}")
            )

    print()
    if failed:
        print(f"RESULT: {failed} episode(s) FAILED — the dataset disagrees with the")
        print("simulator. If the error is confined to locomotion steps, PinkJointHold")
        print("is not writing _processed_actions.")
        return 1
    print("RESULT: PASS — recorded commands match what the simulator applied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
