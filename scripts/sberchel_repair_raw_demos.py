#!/usr/bin/env python3
"""Repair already-collected Sberchel teleop HDF5 files (raw or camera-augmented).

Each fix is opt-in and lossless — everything is recomputed from data the file
already carries:

- ``--stamp_success``: set ``success=True`` on every demo. Valid for
  operator-curated sessions (every exported demo was a deliberate SAVE press);
  older record-script versions left the recorder default ``False`` on all
  demos, which makes ``--only_successful`` filtering export nothing.
- ``--relabel_target_items``: rewrite ``target_item``/``task_instruction``
  attrs by re-deriving each demo's target from its ``seed`` attr with the
  corrected item names. Selection is seed-index based, so the positional
  old->new mapping is exact; use for recordings made with outdated product
  names. Also updates ``env_args['sberchel_target_items']``.
- ``--fix_obs_seams``: repair the 1-row stale-observation glitch that older
  record-script versions left at every V/R1 checkpoint-restore seam
  (``restore_checkpoint`` did not refresh ``env.obs_buf``, so one obs row per
  restore came from the abandoned pre-restore timeline). The pre-step obs is
  redundant with the post-step streams, so the repair is exact:
  ``obs/robot_joint_pos[i] := joint_targets/joint_pos[i-1]`` and
  ``obs/actions[i] := actions[i-1]`` wherever they disagree beyond tolerance.

Usage (repaired copy; use --in_place to modify the input):
    python scripts/sberchel_repair_raw_demos.py datasets/dex_X.hdf5 \
        --output datasets/dex_X_repaired.hdf5 \
        --stamp_success --fix_obs_seams \
        --relabel_target_items "Monster Pipeline Punch,Kagome Juice,Tomato Paste,Palmolive Shampoo"
"""

import argparse
import hashlib
import json
import shutil
import sys

import h5py
import numpy as np


def derive_target_index(episode_seed: int, num_items: int) -> int:
    """Pick-target index from the demo's seed attr (mirrors the record script)."""
    digest = hashlib.blake2b(f"{episode_seed}:target".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % num_items


def target_task_instruction(item: str) -> str:
    """Natural-language pick task for a target item (mirrors the record script)."""
    article = "an" if item.strip()[:1].lower() in "aeiou" else "a"
    return f"Pick {article} {item.strip()} and place it on the table."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("input", help="Teleop HDF5 file to repair.")
    parser.add_argument("--output", default=None,
                        help="Write a repaired copy here instead of modifying the input.")
    parser.add_argument("--in_place", action="store_true",
                        help="Modify the input file directly (mutually exclusive with --output).")
    parser.add_argument("--dry_run", action="store_true",
                        help="Report what would change without writing anything.")
    parser.add_argument("--stamp_success", action="store_true",
                        help="Set success=True on every demo (operator-curated session).")
    parser.add_argument("--relabel_target_items", default=None,
                        help="Comma-separated corrected item names; targets are re-derived "
                             "from each demo's seed attr (positional old->new mapping).")
    parser.add_argument("--fix_obs_seams", action="store_true",
                        help="Repair stale obs rows at checkpoint-restore seams from the "
                             "redundant post-step streams.")
    parser.add_argument("--seam_tol", type=float, default=0.05,
                        help="Joint-angle mismatch (rad) above which an obs row counts as a "
                             "seam glitch. Clean rows match the post-step stream exactly.")
    args = parser.parse_args()

    if not (args.stamp_success or args.relabel_target_items or args.fix_obs_seams):
        sys.exit("Nothing to do: pass --stamp_success, --relabel_target_items and/or --fix_obs_seams.")
    if args.output and args.in_place:
        sys.exit("--output and --in_place are mutually exclusive.")

    if args.dry_run:
        target, mode = args.input, "r"
    elif args.output:
        shutil.copy2(args.input, args.output)
        target, mode = args.output, "r+"
    elif args.in_place:
        target, mode = args.input, "r+"
    else:
        sys.exit("Refusing to modify the input silently: pass --output <copy> or --in_place "
                 "(or --dry_run to preview).")

    items = None
    if args.relabel_target_items:
        items = [s.strip() for s in args.relabel_target_items.split(",") if s.strip()]

    stamped = relabeled = seams = 0
    with h5py.File(target, mode) as f:
        env_args = json.loads(f["data"].attrs.get("env_args", "{}"))
        old_items = env_args.get("sberchel_target_items")
        demo_names = sorted(f["data"], key=lambda k: int(k.split("_")[-1]))
        for demo_name in demo_names:
            demo = f[f"data/{demo_name}"]

            if args.stamp_success and not bool(demo.attrs.get("success", False)):
                print(f"[{demo_name}] success: False -> True")
                stamped += 1
                if not args.dry_run:
                    demo.attrs["success"] = True

            if items:
                seed = demo.attrs.get("seed")
                if seed is None:
                    sys.exit(f"{demo_name}: no 'seed' attr; cannot relabel the target.")
                idx = derive_target_index(int(seed), len(items))
                # Sanity: the same index into the recorded (old) list must
                # reproduce the stored attr, or the seed scheme has drifted.
                stored = demo.attrs.get("target_item")
                if old_items and stored is not None and old_items[idx] != stored:
                    sys.exit(
                        f"{demo_name}: seed-derived old item {old_items[idx]!r} != stored "
                        f"target_item {stored!r}; refusing to relabel (scheme mismatch)."
                    )
                item = items[idx]
                if stored != item:
                    print(f"[{demo_name}] target_item: {stored!r} -> {item!r}")
                    relabeled += 1
                if not args.dry_run:
                    demo.attrs["target_item"] = item
                    demo.attrs["task_instruction"] = target_task_instruction(item)

            if args.fix_obs_seams:
                # The two obs streams can go stale independently (joint pos was
                # fixed by the obs_buf refresh before the action-manager rollback
                # existed), so each is detected against its own invariant.
                obs_jp = demo["obs/robot_joint_pos"]
                post_jp = demo["joint_targets/joint_pos"]
                err = np.abs(obs_jp[1:] - post_jp[:-1]).max(axis=1)
                for row in (np.nonzero(err > args.seam_tol)[0] + 1).tolist():
                    print(f"[{demo_name}] obs joint-pos seam at row {row}: max err "
                          f"{err[row - 1]:.3f} rad -> repaired from post-step state")
                    seams += 1
                    if not args.dry_run:
                        obs_jp[row] = post_jp[row - 1]
                if "obs/actions" in demo and "actions" in demo:
                    obs_act = demo["obs/actions"]
                    act = demo["actions"]
                    err_a = np.abs(obs_act[1:] - act[:-1]).max(axis=1)
                    for row in (np.nonzero(err_a > args.seam_tol)[0] + 1).tolist():
                        print(f"[{demo_name}] obs last-action seam at row {row}: max err "
                              f"{err_a[row - 1]:.3f} -> repaired from action stream")
                        seams += 1
                        if not args.dry_run:
                            obs_act[row] = act[row - 1]

        if items and not args.dry_run:
            env_args["sberchel_target_items"] = items
            f["data"].attrs["env_args"] = json.dumps(env_args)

    verb = "would be" if args.dry_run else "were"
    print(f"\n{target}: {stamped} success attr(s), {relabeled} relabel(s), "
          f"{seams} obs seam row(s) {verb} repaired across {len(demo_names)} demo(s).")


if __name__ == "__main__":
    main()
