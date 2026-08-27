#!/usr/bin/env bash
# Record teleop demos into ./datasets/dex_<timestamp>.hdf5. Run from the repo root.
set -euo pipefail

python scripts/sberchel_kb_record_demos.py \
    --task Isaac-Demo-Sberchel-XR-v0 --teleop_device handtracking --enable_pinocchio --device cpu \
    --dataset_file ./datasets/dex_$(date +%Y%m%d_%H%M%S).hdf5 \
    --disable_left_hand --control_device gamepad --grip_report_every_n 360 --ik_gain 1.0 --retargeter dex \
    --idle_pause --auto_start --zest_upper \
    --loco_policy zest --static_blend_time 0.5 \
    --sim_dt_denominator 240 --walk_policy_decimation 1 --spawn_height 0.88 --gc_self_collisions \
    --loco_render_interval 16 --static_render_interval 4 --xr_view_forward 0.1
