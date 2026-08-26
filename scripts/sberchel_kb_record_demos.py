"""Record Sberchel XR demos with WASD keyboard locomotion.

Upper-body teleop is XR hand tracking; legs use the zest locomotion policy
driven by keyboard velocity commands (like ``demo_sberchel_kb.py``). Press **C**
to toggle locomotion (WASD + soft frozen upper body) vs static teleop (station-keep
at the latch pose + live XR hands + headset neck look). Press **T** to teleport the
XR anchor onto the robot (same pose relative to the robot as at episode start).

OpenXR often steals window focus (carb WASD drops). This script polls X11 keymap
globally so hold-to-move works while the headset is focused. During locomotion,
arm PD is lowered to policy-scale gains — stiff XR teleop PD (~3000) on frozen
arms fights balance and makes walking shuffle or stall.

Episodes export on RESET/R. Press **X** to discard the current episode and reset
without writing to HDF5.

With ``--video``, also records the viewport (``rgb_array``) via Gymnasium
``RecordVideo``; each RESET/R saves an MP4 (discard/X skips the video write).

``--control_device gamepad`` swaps the keyboard for a PS5 (DualSense) pad read
from ``/dev/input/js*`` (focus-independent, like the X11 poll): left stick = WASD,
Create = C, L1 = T, R1 = V, D-pad up = R, D-pad down = X, D-pad left = V,
Options = START/STOP.
See ``dsynth_isaac/teleop/gamepad.py``.

Usage:
    python scripts/sberchel_kb_record_demos.py \\
        --task Isaac-Demo-Sberchel-XR-v0 \\
        --teleop_device handtracking \\
        --enable_pinocchio \\
        --dataset_file ./datasets/sberchel_kb_demos.hdf5 \\
        --control_device gamepad \\
        --video
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from isaaclab.app import AppLauncher

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from dsynth_isaac.green_challenge_path import ensure_green_challenge_importable  # noqa: E402
from dsynth_isaac.sim_defaults import apply_sberchel_cli_defaults  # noqa: E402

ensure_green_challenge_importable()

parser = argparse.ArgumentParser(
    description="Record Sberchel XR demos with WASD keyboard locomotion."
)
parser.add_argument("--task", type=str, default="Isaac-Demo-Sberchel-XR-v0", help="Gym task id.")
parser.add_argument(
    "--dataset_file",
    type=str,
    default="./datasets/sberchel_kb_demos.hdf5",
    help="File path to export recorded demos.",
)
parser.add_argument(
    "--teleop_device",
    type=str,
    default="handtracking",
    help="Teleop device key from env teleop_devices.",
)
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz (non-XR only).")
parser.add_argument(
    "--num_demos",
    type=int,
    default=0,
    help="Number of demonstrations to record. Set to 0 for unlimited.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--no_product_jitter",
    action="store_true",
    default=False,
    help="Disable the per-reset pose jitter of first-row shelf products.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Session seed. Defaults to a fresh entropy-based seed each run (printed at startup). "
    "Each start pose gets its own seed derived from (session seed, pose index); the env is "
    "reseeded with it on every reset and it is stored per-demo as HDF5 attr 'seed', so any "
    "trajectory's start configuration can be reproduced with env.reset(seed=<that value>).",
)
parser.add_argument(
    "--enhanced_determinism",
    action="store_true",
    default=False,
    help=(
        "Enable PhysX enhanced determinism (PhysxCfg.enable_enhanced_determinism): "
        "simulation results become independent of CPU worker-thread count and island "
        "batching, at some performance cost. Improves the odds that "
        "sberchel_replay_demos.py reproduces the recorded trajectory bit-for-bit "
        "(replay auto-matches this setting from dataset provenance). Not a full "
        "guarantee on the GPU pipeline. Recorded as sberchel_enhanced_determinism."
    ),
)
parser.add_argument(
    "--solver_iters",
    type=str,
    default=None,
    metavar="POS,VEL",
    help=(
        "Override the robot articulation's PhysX solver iteration counts, e.g. '32,16'. "
        "Default is green_challenge's serialized 4/4, which is adequate under the PGS "
        "solver they select (and that we mirror). Raising them only matters if you also "
        "switch to TGS, where 4/4 leaves the eight loop closures compliant and the zest "
        "policy limit-cycles. Recorded in the dataset provenance either way."
    ),
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio (required for Pink IK / hand tracking).",
)
parser.add_argument(
    "--speed",
    type=float,
    default=1.0,
    help="Forward/back speed for W/S (m/s).",
)
parser.add_argument(
    "--turn",
    type=float,
    default=1.5,
    help="Turn rate for A/D (rad/s).",
)
parser.add_argument(
    "--strafe",
    type=float,
    default=0.6,
    help="Strafe speed for Q/E (m/s).",
)
parser.add_argument(
    "--static_blend_time",
    type=float,
    default=0.5,
    help=(
        "Seconds (sim time) to ramp the arm + neck commands from the held pose "
        "into the live XR targets after every entry into static mode. Without it "
        "the wrist/neck commands snap to wherever the hand/headset currently is "
        "in a single step. 0 disables the ramp."
    ),
)
parser.add_argument(
    "--station_lin_gain",
    type=float,
    default=1.5,
    help="P-gain for static-mode XY station-keep (m/s per meter).",
)
parser.add_argument(
    "--station_ang_gain",
    type=float,
    default=2.5,
    help="P-gain for static-mode yaw station-keep (rad/s per rad).",
)
parser.add_argument(
    "--station_max_lin",
    type=float,
    default=0.8,
    help="Max linear velocity for static-mode station-keep (m/s).",
)
parser.add_argument(
    "--station_max_ang",
    type=float,
    default=1.5,
    help="Max angular velocity for static-mode station-keep (rad/s).",
)
parser.add_argument(
    "--station_pos_deadzone",
    type=float,
    default=0.06,
    help=(
        "XY engage threshold for static-mode station-keep (m). Below it the legs get "
        "an exact zero command and the robot stands still, like green_challenge's own "
        "stack during manipulation. Once the base drifts past it, a correction runs "
        "until the error is back under half this value (hysteresis) -- a single "
        "threshold parks the base exactly on the boundary and chatters forever."
    ),
)
parser.add_argument(
    "--station_yaw_deadzone",
    type=float,
    default=0.10,
    help="Yaw engage threshold for static-mode station-keep (rad). See --station_pos_deadzone.",
)
parser.add_argument(
    "--station_min_lin",
    type=float,
    default=0.15,
    help=(
        "Minimum linear speed while a station-keep correction is engaged (m/s). "
        "Boundary-sized commands (~0.09 m/s) are below what the zest policy turns "
        "into actual motion, so without a floor the correction stalls and the error "
        "never closes."
    ),
)
parser.add_argument(
    "--station_min_ang",
    type=float,
    default=0.3,
    help="Minimum yaw rate while a station-keep correction is engaged (rad/s).",
)
parser.add_argument(
    "--ik_gain",
    type=float,
    default=None,
    help=(
        "Override Pink FrameTask gain (fraction of task error corrected per sim step). "
        "Default: whatever the env cfg ships (0.85). Raise toward 1.0 for snappier "
        "tracking of the hand markers; lower if the arms feel twitchy."
    ),
)
parser.add_argument(
    "--ik_lm_damping",
    type=float,
    default=None,
    help=(
        "Override Pink FrameTask lm_damping. Scales with squared task error, so high "
        "values brake hardest exactly when the target is far. Default: env cfg (1.0)."
    ),
)
parser.add_argument(
    "--ik_damping_cost",
    type=float,
    default=None,
    help="Override the Pink DampingTask cost (resists joint motion). Default: env cfg (0.1).",
)
parser.add_argument(
    "--gravity_compensation",
    action="store_true",
    default=False,
    help=(
        "Feed forward gravity torques on the Pink-controlled joints. With the soft "
        "green_challenge arm PD (42/28 Nm/rad) gravity droop is a constant offset the "
        "PD cannot close, so the hands sit below the markers; this removes it without "
        "touching the gains. NOTE: it is still a plant change (extra effort feedforward) "
        "-- recorded into dataset provenance as sberchel_gravity_compensation. During "
        "WASD locomotion the Pink hold bypasses the feedforward update, so the last "
        "static-mode torques persist on the frozen arms."
    ),
)
parser.add_argument(
    "--legacy_teleop_gains",
    action="store_true",
    default=False,
    help=(
        "Run the upper body on the legacy stiff teleop plant: the pre-split "
        "ImplicitActuator XR PD (arms/wrists 3000/100, fingers 400/40, neck "
        "200/20) instead of green_challenge's native soft gains, for the whole "
        "session and both modes. Much easier to teleoperate, but it is NOT the "
        "plant green_challenge evaluates on -- demos recorded with it track the "
        "operator far better than their VLA stack could reproduce. Legs/torso "
        "still take the active policy's PD, so locomotion is unaffected. "
        "Recorded in provenance (sberchel_legacy_stiff_teleop) and per step in "
        "teleop_state[:,2]. Makes --gravity_compensation redundant (no droop at "
        "3000 Nm/rad), though combining them is harmless."
    ),
)
parser.add_argument(
    "--legacy_gains_keep_actuator",
    action="store_true",
    default=False,
    help=(
        "EXPERIMENT, only with --legacy_teleop_gains: keep green_challenge's "
        "FrictionalActuator on xr_upper and write the stiff legacy gains into its "
        "live tensors, instead of swapping the group to a PhysX-implicit drive. "
        "Isolates 'same gains, different actuator model': friction and armature "
        "stay active, but the PD becomes explicit software torque at sim dt "
        "1/150, which is numerically unstable at 3000/100 -- expect the arms to "
        "chatter/oscillate, saturating at the effort limits. Recorded in "
        "provenance as sberchel_legacy_gains_actuator='frictional'."
    ),
)
parser.add_argument(
    "--zest_upper",
    action="store_true",
    default=False,
    help=(
        "Canonical green_challenge zest tract, one zest policy for both modes. "
        "STATIC mode (run_with_vla.py parity): arms track XR live -- the Pink "
        "IK arm solution feeds zest's commander reference "
        "(set_joint_overrides) and the arm/torso/neck joints execute zest's "
        "output (ref + residual); wrists/fingers stay direct from the "
        "retargeter, so squeeze grip is unchanged. LOCOMOTION mode: the "
        "--loco_policy net runs through the classic machinery. With zest "
        "(run_kb/run.py parity) it contributes only the legs -- the whole "
        "upper body holds its commanded targets (grip preserved, IK skipped "
        "for speed), neck 0, torso held, and the frozen arm pose is fed as "
        "zest's reference so its observations stay consistent (its arm output "
        "is discarded: the residual wobbles arms with the gait). With "
        "--loco_policy walk (the organizers' declared walk+zest split) walk "
        "drives the legs and, per --walk_arm_drive, the arm swing; gain swap "
        "and reset-on-swap apply as in any walk session; the recorded ref is "
        "a no-op for walk's commander. "
        "--static_blend_time ramps back into live tracking on the return to "
        "static. Incompatible with --legacy_teleop_gains. The per-step "
        "13-value reference is recorded as 'zest_upper_ref' so policy_loco "
        "replay re-runs zest bit-exact."
    ),
)
parser.add_argument(
    "--spawn_height",
    type=float,
    default=0.0,
    help=(
        "Override the robot root spawn/reset height in metres (0 = keep the "
        "task cfg's SBERCHEL_STAND_HEIGHT = 0.9). green_challenge initialises "
        "at 0.88 (their json init_state and validation_scene 'center'), so our "
        "default drops the robot 2 cm onto the floor at EVERY reset -- a "
        "systematic landing jolt their plant never has, and part of what "
        "decides which contact equilibrium the in-palm thumbs settle into. "
        "Pass 0.88 for their exact start. Recorded in provenance as "
        "sberchel_spawn_height (replay needs no matching: it restores the "
        "recorded initial state verbatim)."
    ),
)
parser.add_argument(
    "--sim_dt_denominator",
    type=int,
    default=0,
    help=(
        "Override the physics step to 1/N s (0 = keep the task cfg's 1/150). "
        "250 is the measured stability point for the thumb explicit-PD limit "
        "cycle (kd=2.0, armature=0.005 -> critical step ~5 ms; 1/150 = 6.7 ms "
        "rings, 1/250 = 4 ms does not) -- the ring green_challenge's own "
        "run.sh shows too. Invisible in the dataset artifact (achieved states, "
        "resampled to 30 fps on export). Pair with --walk_policy_decimation 5 "
        "so walk keeps its trained 50 Hz. Recorded in provenance as "
        "sberchel_sim_dt; replay auto-matches it."
    ),
)
parser.add_argument(
    "--thumb_probe",
    action="store_true",
    default=False,
    help=(
        "Print per-thumb-joint oscillation stats (RMS joint velocity, "
        "peak-to-peak position) every sim second, plus the resident "
        "stiffness/damping/armature at startup. Same metric as "
        "scripts/thumb_probe_socket.py prints against green_challenge's "
        "run.sh socket, so the two plants compare in joint units instead of "
        "screencast pixels."
    ),
)
parser.add_argument(
    "--gc_self_collisions",
    action="store_true",
    default=False,
    help=(
        "green_challenge-parity self-collisions: spawn the robot WITHOUT "
        "restoring the MJCF contype/conaffinity masks (their importer drops "
        "them, so PhysX collides every non-parent/child link pair -- including "
        "thumb-vs-palm pairs MuJoCo, the policies' training plant, never "
        "collides). Matches the plant their run.sh/eval actually runs: expect "
        "a thumb to sit jammed in the palm at the open-hand pose, and in-hand "
        "contact during grasps to include extra palm/proximal pairs. Also "
        "skips the post-reset settle re-latch of the locomotion hold, so the "
        "init-pose finger command persists exactly like their constant "
        "default-pose commands (their quiet thumb is quiet because the command "
        "wedges it into the palm). Default (off) keeps the MuJoCo-faithful "
        "filter from sberchel_self_collision. Recorded in provenance as "
        "sberchel_mjcf_collision_filters; replay auto-matches it."
    ),
)
parser.add_argument(
    "--print_every_n",
    type=int,
    default=30,
    help="Print debug line every N steps (0 to disable periodic prints).",
)
parser.add_argument(
    "--auto_start",
    action="store_true",
    default=False,
    help="Start recording immediately (skip XR START). Useful to debug WASD locomotion.",
)
parser.add_argument(
    "--loco_render_interval",
    type=int,
    default=5,
    help=(
        "Physics steps per render while in locomotion mode (1 = every step). "
        "Higher skips Pink-free frames' XR/GUI presents and raises wall-clock Hz. "
        "Static teleop renders every step by default (--static_render_interval)."
    ),
)
parser.add_argument(
    "--static_render_interval",
    type=int,
    default=1,
    help=(
        "Physics steps per render while in static teleop (1 = every step). "
        "Unlike the loco interval this is not free: sim.render() is the only Kit "
        "app tick in the loop, and the OpenXR input snapshot _get_raw_data() "
        "reads refreshes on that tick, so raising this makes hand tracking "
        "staler in the same proportion. Raise it either to stop submitting "
        "frames the headset cannot show (printed hz above ~90 refresh -- free), "
        "or to buy wall-clock back for physics when hz is low (costs hand-pose "
        "freshness)."
    ),
)
parser.add_argument(
    "--idle_pause",
    action="store_true",
    default=False,
    help=(
        "Freeze the sim (no physics, no recording) whenever locomotion mode has a "
        "zero stick/WASD command. Stepping resumes the instant a command arrives or "
        "static mode is toggled. Idle standing between mode switches never reaches "
        "the dataset. The view keeps rendering while frozen."
    ),
)
parser.add_argument(
    "--idle_pause_settle",
    type=float,
    default=1.5,
    help=(
        "Seconds to keep stepping after the loco command returns to zero before "
        "--idle_pause freezes the sim. Lets the policy damp walking momentum into a "
        "stand so the frozen state (and any later static-mode hold pose) is calm."
    ),
)
parser.add_argument(
    "--control_device",
    type=str,
    default="keyboard",
    choices=["keyboard", "gamepad", "both"],
    help=(
        "Locomotion/episode input: WASD keyboard, PS5 (DualSense) gamepad, or both. "
        "Gamepad: left stick = WASD, Create = toggle locomotion, L1 = teleport, "
        "D-pad right = save+reset, D-pad left = discard+reset, Options = START/STOP."
    ),
)
parser.add_argument(
    "--gamepad_dev",
    type=str,
    default="",
    help="Joystick device for --control_device gamepad (default: auto-detect DualSense).",
)
parser.add_argument(
    "--gamepad_deadzone",
    type=float,
    default=0.15,
    help="Analog stick deadzone (fraction of full travel).",
)
parser.add_argument(
    "--gamepad_no_sticks",
    action="store_true",
    default=False,
    help="Ignore the analog sticks; drive locomotion from the D-pad (digital WASD) only. "
    "D-pad left/right then turn instead of discarding/saving episodes.",
)
parser.add_argument(
    "--gamepad_debug",
    action="store_true",
    default=False,
    help="Print every gamepad button/axis event (to verify the button mapping).",
)
parser.add_argument(
    "--loco_policy",
    type=str,
    default="zest",
    choices=["zest", "walk"],
    help=(
        "green_challenge policy driving the legs in LOCOMOTION mode; static teleop "
        "always uses zest. 'walk' (policy_4k_tuni) is their dedicated locomotion "
        "policy -- trained on real velocity ranges with its own arm swing, where "
        "zest is a manipulation-first whole-body residual controller. The 25 body "
        "joints take their PD from whichever policy is loaded -- that is how "
        "green_challenge works too -- so 'walk' also changes the leg/arm gains "
        "while walking. Default 'zest' reproduces the previous behaviour exactly."
    ),
)
parser.add_argument(
    "--walk_arm_drive",
    type=str,
    default="when_free",
    choices=["when_free", "always", "never"],
    help=(
        "Who moves the arms during --loco_policy walk locomotion. 'when_free' "
        "(default) hands them to the policy only while nothing is grasped and holds "
        "them frozen once the operator squeezes -- swinging the shoulders while "
        "carrying a product drags the hand through space and can break the grasp. "
        "'always' is unconditional (closest to how walk was trained: it observes all "
        "25 dofs and learned balance while executing its own arm swing). 'never' "
        "keeps today's frozen hold. The gate reads the squeeze channel already "
        "recorded as teleop_state[:,1], so replay reproduces it from the dataset."
    ),
)
parser.add_argument(
    "--walk_policy_decimation",
    type=int,
    default=1,
    help=(
        "Env steps per walk-policy inference (its output is held in between). The env "
        "runs sim.dt=1/150 with decimation=1 while both policies were trained at 50 Hz. "
        "zest absorbs that -- it is a residual on a time-independent commander reference "
        "-- but walk is absolute with a 15-frame observation history, so its only clock "
        "IS that history: at 150 Hz the window spans 0.1 s instead of 0.3 s. 3 restores "
        "the trained 50 Hz. Default is 1 because a headless bench showed 1/3/5 give "
        "identical forward progress (0.355/0.358/0.363 m in 10 s at vx=0.5) while the "
        "held, piecewise-constant target measurably increased base sway (path length "
        "0.80/1.22/1.53 m). Try 3 if the gait looks slow-motion in the real rig."
    ),
)
parser.add_argument(
    "--loco_blend_time",
    type=float,
    default=0.5,
    help=(
        "Seconds to ramp the arms from the held pose into the walk policy's targets "
        "on every locomotion entry (mirror of --static_blend_time). walk's nominal "
        "arm pose is arms-at-sides while the operator enters locomotion from a reach, "
        "often radians away; without the ramp that whole delta is commanded in one "
        "step. 0 snaps."
    ),
)
parser.add_argument(
    "--retargeter",
    type=str,
    default="vrd",
    choices=["vrd", "dex"],
    help=(
        "Finger mapping. 'vrd' = per-digit curl heuristic (default). 'dex' = DexPilot "
        "optimizer matching inter-fingertip vectors, as IsaacLab ships for GR1T2; better "
        "in principle for pinch/precision grasps. Affects only how your hand becomes "
        "finger commands -- the robot, its gains and the recorded action layout are "
        "identical either way. Validate first: scripts/sberchel_check_dex_retargeting.py"
    ),
)
parser.add_argument(
    "--squeeze_scale",
    type=float,
    default=1.0,
    help=(
        "Multiplier on the operator squeeze input (gamepad L2 / keyboard G) before it "
        "is clamped to 0..1. Squeeze commands finger closure past where your own hand "
        "stopped, which is what generates grip force on a position-controlled hand. "
        "Raise above 1.0 to reach full closure at partial trigger travel."
    ),
)
parser.add_argument(
    "--squeeze_binary",
    action="store_true",
    default=False,
    help=(
        "Make grasp squeeze 0/1 instead of analog: L2 travel is latched through a "
        "Schmitt trigger (engage above 0.6, release below 0.4 -- hysteresis so a "
        "half-pulled trigger cannot flutter the grip open mid-carry) into either 0 or "
        "--squeeze_on_value. Matches green_challenge's own convention (their keyboard/"
        "socket paths always push finger targets to full closure, never modulate) and "
        "makes every recorded grasp identical."
    ),
)
parser.add_argument(
    "--squeeze_on_value",
    type=float,
    default=1.0,
    help=(
        "Squeeze level commanded while the binary latch is engaged (0..1; applies to "
        "keyboard G too). Lower toward ~0.6 for gentler contact on small rigid objects. "
        "Only used with --squeeze_binary; --squeeze_scale still multiplies on top."
    ),
)
parser.add_argument(
    "--disable_left_hand",
    action="store_true",
    default=False,
    help=(
        "The left arm/hand never follows the operator: it holds the measured pose "
        "(wrist pelvis-relative + fingers) latched shortly after each episode reset, "
        "re-latched on every static-mode entry, in both locomotion and static modes; "
        "squeeze does not affect the left fingers."
    ),
)
parser.add_argument(
    "--grip_report_every_n",
    type=int,
    default=0,
    help=(
        "Print a per-hand grip readout (commanded vs achieved finger angle, error, "
        "estimated torque vs the 8.5 Nm effort limit) every N steps. 0 disables. "
        "Use this to confirm grasps actually generate force instead of assuming it."
    ),
)
parser.add_argument(
    "--video",
    action="store_true",
    default=False,
    help="Record viewport MP4s via Gymnasium RecordVideo (saved on RESET/R).",
)
parser.add_argument(
    "--video_dir",
    type=str,
    default="",
    help="Directory for viewport MP4s. Default: <dataset_dir>/videos.",
)
parser.add_argument(
    "--cameras",
    type=str,
    default="left_head",
    help=(
        "Comma-separated robot cameras to render and record into the HDF5 when "
        "--enable_cameras is set: left_head,right_head,left_wrist,right_wrist "
        "(or 'all'/'none'). Default keeps only the left head camera; extra "
        "cameras slow teleop rendering. Prefer recording without cameras and "
        "adding RGB later via sberchel_replay_demos.py --output_dataset_file."
    ),
)
parser.add_argument(
    "--camera_size",
    type=str,
    default="448x448",
    help=(
        "Robot camera resolution: 'WxH', a single square size, or 'native' (960x600). "
        "Default 448x448 matches the LeRobot sample layout (center-crop-resize "
        "rescaled intrinsics)."
    ),
)
parser.add_argument(
    "--camera_hz",
    type=float,
    default=30.0,
    help=(
        "Robot camera refresh rate during teleop (sensor update_period = 1/hz; "
        "0 = re-render every env step). The recorder still writes an image row "
        "per 150 Hz env step, so rows repeat the latest frame between refreshes; "
        "30 Hz matches the LeRobot target rate (green_challenge's default is 15)."
    ),
)
parser.add_argument(
    "--target_items",
    type=str,
    default="Energy Drink,Juice,Tomato Paste,Shampoo",
    help=(
        "Comma-separated pick-target names. The session's first start pose gets "
        "a random target (derived from the episode seed); every later pose "
        "cycles through this list in order. A retried attempt keeps its "
        "target either way. Shown as a small text panel at the top-right of "
        "the VR view."
    ),
)
parser.add_argument(
    "--no_target_hud",
    action="store_true",
    default=False,
    help="Disable the in-headset target-item text panel.",
)
parser.add_argument(
    "--target_hud_offset",
    type=float,
    nargs=3,
    default=(0.9, 0.45, 1.8),
    metavar=("RIGHT", "UP", "FWD"),
    help=(
        "Target panel placement relative to the headset view, in metres: right, "
        "up, and forward. Snapshotted at every recording start / reset / "
        "T-teleport / static-mode entry, so the panel lands at the top-right of "
        "wherever you are looking and then stays put."
    ),
)
parser.add_argument(
    "--target_hud_width",
    type=float,
    default=1.6,
    help="Maximum width of the target task panel in metres (text wraps to fit).",
)
AppLauncher.add_app_launcher_args(parser)
# Sberchel runs on green_challenge's CPU physics by default; --device overrides.
apply_sberchel_cli_defaults(parser)
args_cli = parser.parse_args()

if args_cli.seed is None:
    args_cli.seed = int.from_bytes(os.urandom(4), "little") % (2**31)
    print(f"[INFO] No --seed given; using seed {args_cli.seed} (pass --seed {args_cli.seed} to reproduce this session)")

app_launcher_args = vars(args_cli)
_use_handtracking = "handtracking" in args_cli.teleop_device.lower()
if args_cli.enable_pinocchio or _use_handtracking:
    import pinocchio  # noqa: F401

if _use_handtracking:
    try:
        import dex_retargeting  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Sberchel hand-tracking teleop requires dex-retargeting. "
            "Install with: pip install 'dex-retargeting==0.4.6'"
        ) from exc

if _use_handtracking:
    app_launcher_args["xr"] = True
# Viewport rgb_array / RecordVideo needs camera rendering enabled.
if args_cli.video:
    app_launcher_args["enable_cameras"] = True

app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

from isaaclab.devices.device_base import DeviceBase
from dsynth_isaac.robots.sberchel_camera_cfg import (
    SBERCHEL_CAMERA_OBS_KEYS,
    apply_sberchel_camera_size,
    parse_camera_names,
    parse_camera_size,
    strip_camera_configs,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import DatasetExportMode
from isaaclab_tasks.utils import parse_env_cfg

import dsynth_isaac.tasks.manager_based  # noqa: F401

from dsynth_isaac.recorders import (
    SberchelXRRecorderManagerCfg,
    enable_async_episode_export,
    stash_episode_extra_attrs,
    stash_locomotion_command,
    stash_teleop_state,
    stash_zest_upper_ref,
)

from dsynth_isaac.teleop.sberchel_head_follow import (
    FrozenUpperSnapshot,
    UpperBodyMode,
    blend_upper_command,
    build_idle_upper_from_robot,
    build_upper_from_frozen_snapshot,
    capture_frozen_upper,
    smoothstep,
    create_sberchel_xr_teleop_interface,
    headset_to_neck_targets,
    pack_sberchel_xr_action,
    robot_root_state,
    yaw_from_quat_wxyz,
    StationKeeper,
    sync_retargeter_idle_from_robot,
)
from dsynth_isaac.teleop.gamepad import RecordGamepad
from dsynth_isaac.teleop.xr_anchor import (
    XrAnchorTeleporter,
    end_xr_display,
    xr_display_active,
)
from dsynth_isaac.teleop.xr_target_hud import XrTargetHud
from dsynth_isaac.robots.sberchel_cfg import (
    GREEN_CHALLENGE_SPAWN_FUNC,
    reassert_legacy_teleop_gain_tensors,
    sberchel_upper_pd_gain_table,
    xr_upper_legacy_teleop_actuator,
)
from dsynth_isaac.tasks.manager_based.sberchel_upperbody_xr_env_cfg import (
    SBERCHEL_HAND_JOINT_NAMES,
    build_sberchel_teleop_devices,
)
from dsynth_isaac.tasks.manager_based.sberchel_base_env_cfg import (
    LEGS_ACTION_DIM,
    NECK_ACTION_DIM,
    NECK_JOINT_NAMES,
    NECK_PITCH_LIMIT,
    NECK_YAW_LIMIT,
    POLICY_ARM_JOINT_NAMES,
    POLICY_MODEL_ARM_SLICE,
    POLICY_MODEL_NECK_SLICE,
    POLICY_TORSO_MODEL_INDEX,
    SBERCHEL_STAND_HEIGHT,
    LocoMode,
    LocoPolicySet,
)

logger = logging.getLogger(__name__)

# --walk_arm_drive when_free: below this squeeze the hands count as empty and the
# walk policy is allowed to swing the arms. Anything above is treated as "holding
# something", where an arm swing would drag the grasped product around.
_ARM_DRIVE_SQUEEZE_EPS = 1e-3

# Left-hand slots in the 26D upper command (--disable_left_hand): wrist pose7 at
# [0:7], plus the left entries of the 12D finger block at [14:26].
_LEFT_WRIST_SLICE = slice(0, 7)
_LEFT_FINGER_DIMS = tuple(
    14 + i for i, name in enumerate(SBERCHEL_HAND_JOINT_NAMES) if name.startswith("left_")
)

# The reset pose writes left thumb yaw at -0.1, inside the palm collision hull
# (PhysX self-collisions ignore the MJCF contype/conaffinity filters), so PhysX
# kicks the thumb out and it settles onto a contact equilibrium near 0.24 rad.
# Commanding the unreachable -0.1 back at it saturates the 8.5 Nm finger drive
# against that contact and the joint chatters at ~37 Hz. Latch the
# --disable_left_hand hold from the *measured* pose only after this many env
# steps (~0.2 s at 150 Hz) so it records the reachable equilibrium, not the
# in-penetration reset value.
_LEFT_HOLD_SETTLE_STEPS = 30

_HELP_KEYBOARD = """
==================== KB RECORD CONTROLS ====================
  W forward    S backward    A turn left    D turn right
  Q strafe L   E strafe R    (release = stop legs in loco mode)
  C  toggle locomotion (WASD + Pink hold-off) / static teleop
     (static: XR hands + headset neck; Pink IK on, slower)
  G  hold to squeeze (close fingers past your hand -> real grip force)
  T  teleport XR view onto the robot (same offset as at episode start)
  V / Left   restore last checkpoint (auto-taken at each C into static
             mode: rolls sim + recording back — retry a grasp seamlessly)
  R / Up     save episode + reset, new random start pose
  X / Down   discard episode + reset, SAME start pose (retry attempt)
  Right      discard episode + reset, new random start pose
  XR START / STOP  begin or pause recording (or pass --auto_start)
  Keys: X11 global poll (works while headset has focus) + Isaac window
  Tip: if loco hz still low, raise --loco_render_interval (e.g. 10)
===========================================================
"""

_HELP_GAMEPAD = """
================= PS5 PAD RECORD CONTROLS =================
  Left stick        forward/back + turn  (= WASD)
  Right stick X     analog strafe
  D-pad             digital WASD fallback (with --gamepad_no_sticks)
  Create   toggle locomotion / static teleop   (= C)
  L2       analog grasp squeeze (0..1)         (= G)
  L1       teleport XR view onto the robot     (= T)
  R1       restore loco->static checkpoint     (= V / Left arrow)
  D-pad ^  save episode + reset, new pose      (= R / Up arrow)
  D-pad v  discard + reset, SAME pose (retry)  (= X / Down arrow)
  D-pad >  discard + reset, new pose           (= Right arrow)
  D-pad <  restore loco->static checkpoint     (= V / Left arrow)
  Options  start / stop recording        (= XR START / STOP)
  Pad is read from /dev/input/js* — works while the headset has focus
  Check the mapping any time: python -m dsynth_isaac.teleop.gamepad
===========================================================
"""

_MOVE_KEYS = ("W", "A", "S", "D", "Q", "E")
_EDGE_KEYS = ("C", "R", "X", "T", "V", "UP", "DOWN", "LEFT", "RIGHT")


class PinkJointHold:
    """Bypass Pink IK solves by holding arm/hand joint targets.

    During WASD locomotion the upper body is frozen anyway; re-solving Pink every
    physics step (~10 Hz wall-clock with XR) is pure overhead. Holding joint-space
    targets keeps arms still relative to the torso at policy rate.

    A second, mutually exclusive wrapper mode exists for ``--zest_upper``
    (``enable_zest_passthrough``): Pink IK keeps running and its wrist/hand
    targets apply untouched, but the 10 arm joints are re-commanded from zest's
    output, and the raw IK arm solution is captured each step to serve as the
    *next* control tick's zest commander reference. One tick of reference
    staleness (~20 ms) is far below human teleop bandwidth, and it is what makes
    the loop causal: the reference for step t must exist before zest runs, while
    the IK solve for step t happens inside ``env.step``.
    """

    def __init__(self, env) -> None:
        self._env = env
        self._term = env.action_manager.get_term("upper_body_ik")
        self._orig_apply = self._term.apply_actions
        self._targets: torch.Tensor | None = None
        self._active = False
        self._targets_valid = False
        self._zest_passthrough = False
        self._zest_arm_targets: torch.Tensor | None = None
        self._last_ik_arm: torch.Tensor | None = None
        self._arm_slots, self._arm_joint_ids = self._resolve_arm_slots()

    @property
    def arm_joint_ids(self) -> list[int]:
        """Articulation joint ids of the 10 policy arm joints (model order)."""
        return list(self._arm_joint_ids)

    def _resolve_arm_slots(self) -> tuple[list[int], list[int]]:
        """Locate the 10 policy arm joints among the term's controlled joints.

        ``_controlled_joint_names`` is post-remap (wrist pitch/roll are written
        to the wrist crank dofs), but the arm joints are never remapped, so they
        match the policy's names directly. Resolved once and checked loudly:
        both vectors are length 10, so a reordered articulation would silently
        land shoulder commands on elbows.
        """
        names = list(self._term._controlled_joint_names)
        ids = [int(i) for i in self._term._controlled_joint_ids]
        slot_by_name = {name: slot for slot, name in enumerate(names)}
        missing = [n for n in POLICY_ARM_JOINT_NAMES if n not in slot_by_name]
        if missing:
            raise RuntimeError(
                f"[PinkJointHold] upper_body_ik does not control {missing}; the "
                "walk policy's arm targets have nowhere to go. Did the action "
                "layout or the robot model change?"
            )
        slots = [slot_by_name[n] for n in POLICY_ARM_JOINT_NAMES]
        return slots, [ids[s] for s in slots]

    def set_arm_targets(
        self,
        raw_arm: torch.Tensor,
        *,
        blend_from: torch.Tensor | None = None,
        alpha: float = 1.0,
    ) -> None:
        """Drive the 10 arm slots from the policy; wrists and fingers keep their hold.

        ``raw_arm`` is ``Network.__call__`` output, which already has the USD
        default subtracted for the offset-based ``JointPositionActionCfg`` terms.
        This writes *absolute* joint targets, so the default goes back on.

        Leaving the wrist and finger slots alone is what preserves grasp force:
        the hold captured them as *commanded* targets, which sit deeper than the
        measured pose by exactly the PD error holding the object.
        """
        if self._targets is None:
            return
        # default_joint_pos is a live articulation buffer; read it rather than
        # caching, even though the arm defaults happen to be zero today.
        default = self._term._asset.data.default_joint_pos[:, self._arm_joint_ids]
        target = raw_arm.to(self._targets) + default
        if blend_from is not None and alpha < 1.0:
            # Ramp in absolute joint space, so the caller never has to know about
            # the default offset.
            target = blend_from.to(target) + alpha * (target - blend_from.to(target))
        self._targets[:, self._arm_slots] = target

    def arm_targets(self) -> torch.Tensor | None:
        """Current absolute hold targets for the 10 arm slots (blend source)."""
        if self._targets is None:
            return None
        return self._targets[:, self._arm_slots].clone()

    def set_zest_arm_targets(self, raw_arm: torch.Tensor) -> None:
        """Arm targets zest computed for this step, to be applied over the IK's.

        ``raw_arm`` is ``Network.__call__`` output (USD default already
        subtracted); stored absolute, same convention as ``_processed_actions``.
        """
        default = self._term._asset.data.default_joint_pos[:, self._arm_joint_ids]
        self._zest_arm_targets = raw_arm.to(default) + default

    def last_ik_arm_solution(self) -> torch.Tensor | None:
        """Freshest raw Pink IK arm solution (absolute), or None right after reset."""
        return self._last_ik_arm

    def enable_zest_passthrough(self) -> None:
        """Install the wrapper in zest-passthrough form (see class docstring)."""
        self._zest_passthrough = True
        self._targets = None
        if not self._active:
            self._term.apply_actions = self._apply_hold  # type: ignore[method-assign]
            self._active = True

    def invalidate(self) -> None:
        """Mark commanded targets as stale — call right after ``env.reset()``.

        ``joint_pos_target`` is allocated as zeros and is *not* restored to the
        default pose on reset (``articulation.py:1641``), so after a reset it
        still holds the previous episode's commands while the robot has been
        teleported back to its init pose. In zest-passthrough form this also
        drops the captured IK solution and pending zest targets: both describe
        the pre-reset pose, and the next reference falls back to the measured
        arms instead.
        """
        self._targets_valid = False
        self._last_ik_arm = None
        self._zest_arm_targets = None

    def mark_stepped(self) -> None:
        """Commanded targets are meaningful once a step has written them."""
        self._targets_valid = True

    def _capture(self, robot) -> torch.Tensor:
        """Snapshot the *commanded* joint targets, not the measured pose.

        This distinction is the whole grasp. A hand squeezing an object sits at a
        measured angle the object allows, while the command sits deeper; the
        difference is exactly what the PD converts into grip force. Capturing
        ``joint_pos`` would set target := measured, zeroing that error and
        dropping whatever is being held the instant the hold engages.

        Before the first step of an episode there is no meaningful command yet,
        so fall back to the measured pose — otherwise the arms would be driven to
        the zero-initialised target and snap out of their init pose.
        """
        ids = self._term._controlled_joint_ids
        source = robot.data.joint_pos_target if self._targets_valid else robot.data.joint_pos
        return source[:, ids].detach().clone()

    def enable(self, robot) -> None:
        # Mutually exclusive with zest passthrough: --zest_upper locomotion
        # swaps the wrapper back to the classic hold (wrists/fingers frozen at
        # commanded targets, IK skipped), with set_arm_targets layering zest's
        # arm output on top.
        self._zest_passthrough = False
        self._targets = self._capture(robot)
        if not self._active:
            self._term.apply_actions = self._apply_hold  # type: ignore[method-assign]
            self._active = True

    def refresh(self, robot, *, measured: bool = False) -> None:
        """Re-capture hold targets (call after reset / mode entry).

        ``measured=True`` forces the capture from the measured pose even after
        steps have validated ``joint_pos_target``. Needed for the post-reset
        settle re-latch: by then the commanded targets ARE the problem (the
        in-penetration reset pose), and capturing them would re-latch it.
        """
        if measured:
            source_valid = self._targets_valid
            self._targets_valid = False
            self._targets = self._capture(robot)
            self._targets_valid = source_valid
        else:
            self._targets = self._capture(robot)

    def disable(self) -> None:
        if self._active:
            self._term.apply_actions = self._orig_apply  # type: ignore[method-assign]
            self._active = False

    def _apply_hold(self) -> None:
        if self._zest_passthrough:
            # Live IK first: solves and applies targets for every controlled
            # joint (wrists/hands keep these), rebinding _processed_actions.
            self._orig_apply()
            processed = self._term._processed_actions
            # Raw IK arm solution BEFORE the zest overwrite: next tick's ref.
            # apply_actions runs once per physics step, so with decimation the
            # last solve of the env step is the one that survives (freshest).
            self._last_ik_arm = processed[:, self._arm_slots].detach().clone()
            targets = self._zest_arm_targets
            if targets is not None:
                targets = targets.to(processed)
                # Last write wins in the PhysX target buffer: the arm joints
                # execute zest, wrists/hands keep the IK write from above.
                self._term._asset.set_joint_position_target(targets, self._arm_joint_ids)
                # Dataset truth: processed_actions must describe what ran.
                processed[:, self._arm_slots] = targets
            return
        if self._targets is None:
            self._orig_apply()
            return
        self._term._asset.set_joint_position_target(self._targets, self._term._controlled_joint_ids)
        # Keep ``processed_actions`` in step with what we actually applied.
        # Bypassing apply_actions also bypasses the assignment inside it, so
        # IsaacLab's PostStepProcessedActionsRecorder would otherwise log a stale
        # vector from the last real IK solve for the whole locomotion segment --
        # i.e. the dataset would disagree with the simulator.
        processed = self._term._processed_actions
        if processed.shape != self._targets.shape:
            raise RuntimeError(
                "PinkJointHold: processed_actions shape "
                f"{tuple(processed.shape)} != held targets {tuple(self._targets.shape)}; "
                "the upper_body_ik action layout changed and the hold would record garbage."
            )
        processed[:] = self._targets


def set_render_interval(env, interval: int) -> None:
    """Set physics-steps-per-render (read every ManagerBasedEnv.step)."""
    env.cfg.sim.render_interval = max(1, int(interval))


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz: int):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env: gym.Env) -> None:
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()
        self.last_time = self.last_time + self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


class _X11KeyPoller:
    """Poll physical key state via XQueryKeymap (no window-focus requirement)."""

    # X11 keysyms (lowercase letter keys).
    _KEYSYMS = {
        "W": 0x0077,
        "A": 0x0061,
        "S": 0x0073,
        "D": 0x0064,
        "Q": 0x0071,
        "E": 0x0065,
        "C": 0x0063,
        "R": 0x0072,
        "X": 0x0078,
        "T": 0x0074,
        "G": 0x0067,
        "V": 0x0076,
        # Arrow keys (X11 cursor keysyms).
        "UP": 0xFF52,
        "DOWN": 0xFF54,
        "LEFT": 0xFF51,
        "RIGHT": 0xFF53,
    }

    def __init__(self) -> None:
        import ctypes
        import ctypes.util

        self._ctypes = ctypes
        lib_name = ctypes.util.find_library("X11")
        if not lib_name:
            raise RuntimeError("libX11 not found")
        self._x11 = ctypes.cdll.LoadLibrary(lib_name)
        self._x11.XOpenDisplay.restype = ctypes.c_void_p
        self._x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        self._x11.XKeysymToKeycode.restype = ctypes.c_uint
        self._x11.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        self._x11.XQueryKeymap.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
        self._display = self._x11.XOpenDisplay(None)
        if not self._display:
            raise RuntimeError("XOpenDisplay failed")
        self._keycodes = {
            name: int(self._x11.XKeysymToKeycode(self._display, keysym))
            for name, keysym in self._KEYSYMS.items()
        }

    def pressed(self) -> set[str]:
        ctypes = self._ctypes
        keymap = (ctypes.c_char * 32)()
        self._x11.XQueryKeymap(self._display, keymap)
        down: set[str] = set()
        for name, keycode in self._keycodes.items():
            if keycode <= 0:
                continue
            byte = keymap[keycode // 8]
            bit = ord(byte) if isinstance(byte, (bytes, bytearray)) else byte
            if bit & (1 << (keycode % 8)):
                down.add(name)
        return down

    def close(self) -> None:
        if getattr(self, "_display", None):
            try:
                self._x11.XCloseDisplay(self._display)
            except Exception:
                pass
            self._display = None


class RecordKeyboard:
    """WASDQE hold-to-move plus edge-triggered C / R / X / T / V and arrows.

    Arrow aliases: Up = save+reset (R), Down = discard+reset (X),
    Left = restore checkpoint (V).

    Uses X11 keymap polling (global, works under OpenXR focus steal) unioned with
    Isaac carb press/release events as a fallback.
    """

    def __init__(
        self,
        speed: float,
        turn: float,
        strafe: float,
        *,
        on_toggle_mode: Callable[[], None] | None = None,
        on_save_reset: Callable[[], None] | None = None,
        on_discard_reset: Callable[[], None] | None = None,
        on_next_pose_reset: Callable[[], None] | None = None,
        on_teleport_view: Callable[[], None] | None = None,
        on_restore_checkpoint: Callable[[], None] | None = None,
    ) -> None:
        import carb
        import omni

        self._carb = carb
        self._speed = speed
        self._turn = turn
        self._strafe = strafe
        self._on_toggle_mode = on_toggle_mode
        self._on_save_reset = on_save_reset
        self._on_discard_reset = on_discard_reset
        self._on_next_pose_reset = on_next_pose_reset
        self._on_teleport_view = on_teleport_view
        self._on_restore_checkpoint = on_restore_checkpoint
        self._carb_pressed: set[str] = set()
        self._edge_prev: set[str] = set()
        self._ok = False
        self._x11: _X11KeyPoller | None = None
        try:
            self._x11 = _X11KeyPoller()
            print("[KB] X11 global key poll active (WASD works without Isaac focus)", flush=True)
        except Exception as exc:
            print(
                f"[KB] X11 key poll unavailable ({exc!r}); "
                f"click the Isaac window so carb keyboard works.",
                flush=True,
            )
        try:
            self._appwindow = omni.appwindow.get_default_app_window()
            self._input = carb.input.acquire_input_interface()
            self._keyboard = self._appwindow.get_keyboard()
            self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_event)
            self._ok = True
        except Exception as exc:  # headless / no window -> no carb keyboard
            print(
                f"[KB] carb keyboard unavailable ({exc!r}).",
                flush=True,
            )

    def _on_event(self, event, *args, **kwargs):
        carb = self._carb
        try:
            et = event.type
            if et not in (
                carb.input.KeyboardEventType.KEY_PRESS,
                carb.input.KeyboardEventType.KEY_RELEASE,
            ):
                return True
            raw = event.input
            name = raw if isinstance(raw, str) else getattr(raw, "name", "")
            name = name.rsplit(".", 1)[-1].upper()
            if et == carb.input.KeyboardEventType.KEY_PRESS:
                self._carb_pressed.add(name)
            else:
                self._carb_pressed.discard(name)
        except Exception:
            pass
        return True

    def _pressed_now(self) -> set[str]:
        pressed = set(self._carb_pressed)
        if self._x11 is not None:
            try:
                pressed |= self._x11.pressed()
            except Exception:
                pass
        return pressed

    def poll(self) -> None:
        """Process edge-triggered C / R / X / T / V / arrows; call every loop iteration."""
        pressed = self._pressed_now()
        edge_now = pressed & set(_EDGE_KEYS)
        newly = edge_now - self._edge_prev
        if "C" in newly and self._on_toggle_mode is not None:
            self._on_toggle_mode()
        if newly & {"R", "UP"} and self._on_save_reset is not None:
            self._on_save_reset()
        if newly & {"X", "DOWN"} and self._on_discard_reset is not None:
            self._on_discard_reset()
        if "RIGHT" in newly and self._on_next_pose_reset is not None:
            self._on_next_pose_reset()
        if "T" in newly and self._on_teleport_view is not None:
            self._on_teleport_view()
        if newly & {"V", "LEFT"} and self._on_restore_checkpoint is not None:
            self._on_restore_checkpoint()
        self._edge_prev = edge_now

    def squeeze(self) -> float:
        """Hold G for full grasp squeeze (keyboard has no analog axis)."""
        return 1.0 if "G" in self._pressed_now() else 0.0

    def command(self) -> tuple[float, float, float]:
        """Return (lin_vel_x, lin_vel_y, ang_vel_z) from held WASDQE keys."""
        p = self._pressed_now()
        vx = (self._speed if "W" in p else 0.0) - (self._speed if "S" in p else 0.0)
        wz = (self._turn if "A" in p else 0.0) - (self._turn if "D" in p else 0.0)
        vy = (self._strafe if "Q" in p else 0.0) - (self._strafe if "E" in p else 0.0)
        return vx, vy, wz

    def pressed_keys(self) -> str:
        """Compact debug string of currently held move keys."""
        p = self._pressed_now()
        return "".join(k for k in _MOVE_KEYS if k in p) or "-"

    def close(self) -> None:
        if self._ok:
            try:
                self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
            except Exception:
                pass
        if self._x11 is not None:
            self._x11.close()
            self._x11 = None


def setup_output_directories() -> tuple[str, str]:
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    # The HDF5 handler opens the dataset with mode "w" (truncate), so reusing a
    # filename would wipe previously recorded demos — auto-suffix instead.
    base_name = output_file_name
    suffix = 0
    while os.path.exists(os.path.join(output_dir, output_file_name + ".hdf5")):
        suffix += 1
        output_file_name = f"{base_name}_{suffix:03d}"
    if suffix:
        print(f"Dataset file {base_name}.hdf5 already exists; writing to {output_file_name}.hdf5 instead")
    return output_dir, output_file_name


def repo_git_commit() -> str:
    """Current commit hash of this repo, with a ``-dirty`` marker for uncommitted changes."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = (
            subprocess.run(
                ["git", "diff", "--quiet", "HEAD"],
                cwd=_repo_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            != 0
        )
        return f"{commit}-dirty" if dirty else commit
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def derive_episode_seed(session_seed: int, pose_index: int) -> int:
    """Per-start-pose seed, well-mixed from (session seed, pose index).

    Hash-based rather than ``session_seed + pose_index`` so that seeds from
    different sessions never form overlapping arithmetic runs, and an eval
    seed range can be kept disjoint from every recorded demo by construction.
    """
    digest = hashlib.blake2b(f"{session_seed}:{pose_index}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**31)


def derive_target_index(episode_seed: int, num_items: int) -> int:
    """Pick-target index for the session's FIRST start pose only.

    Hash-derived (not ``episode_seed % num_items``) so it is independent of the
    spawn randomization, and reproducible offline from the ``seed`` attr every
    demo already carries. Later poses ignore this and cycle through
    ``--target_items`` in list order (see ``roll_target_hud``).
    """
    digest = hashlib.blake2b(f"{episode_seed}:target".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % num_items


def target_task_instruction(item: str) -> str:
    """Natural-language pick task for a target item.

    Indefinite article on purpose: shelves hold several identical instances of
    each product and any one of them satisfies the task.
    """
    article = "an" if item.strip()[:1].lower() in "aeiou" else "a"
    return f"Pick {article} {item.strip()} and place it on the table."


def target_subtask_instructions(item: str) -> list[str]:
    """Five subtask prompts a demo decomposes into, in execution order.

    Boundaries are data-derived, not stamped here: subtasks 1-4 split at the
    three mode switches (``mode_switch_steps`` attr / ``teleop_state[:,0]``
    flips); the place->retract boundary is the step after which the carried
    product stays at its final rest pose in ``store_products`` — placed in the
    container and no longer moved by the hand, only the retract motion remains.
    (Squeeze is deliberately not used: hand-tracking demos can grip with finger
    curl alone and record zero squeeze throughout.)
    """
    noun = item.strip()
    article = "an" if noun[:1].lower() in "aeiou" else "a"
    return [
        f"Go to the shelf with the {noun}.",
        f"Pick {article} {noun} from the shelf.",
        f"Carry the {noun} to the table.",
        f"Put the {noun} into the container.",
        "Move the hand out of the container and return it to the default position.",
    ]


def exported_demo_count(env) -> int:
    """Total episodes written to HDF5."""
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    rm = base.recorder_manager
    return rm.exported_successful_episode_count + rm.exported_failed_episode_count


def discard_viewport_video_frames(env: gym.Env) -> None:
    """Drop in-progress RecordVideo frames so RESET discard does not write an MP4."""
    if isinstance(env, gym.wrappers.RecordVideo) and env.recording:
        env.recorded_frames.clear()


def viewport_video_frame_count(env: gym.Env) -> int:
    """In-progress RecordVideo frame count (0 when video is off)."""
    if isinstance(env, gym.wrappers.RecordVideo) and env.recording:
        return len(env.recorded_frames)
    return 0


def truncate_viewport_video_frames(env: gym.Env, keep: int) -> None:
    """Roll the in-progress RecordVideo buffer back to ``keep`` frames."""
    if isinstance(env, gym.wrappers.RecordVideo) and env.recording:
        del env.recorded_frames[keep:]


def episode_leaf_lengths(data: dict, prefix: str = "") -> dict[str, int]:
    """Per-leaf frame counts of an in-memory ``EpisodeData.data`` dict.

    Leaves are lists of per-frame tensors (``EpisodeData.add`` appends), keyed
    here by their "a/b/c" path. Counting every leaf separately keeps rollback
    correct even for keys recorded at a different cadence (e.g. the once-per-
    episode ``initial_state``).
    """
    lengths: dict[str, int] = {}
    for key, value in data.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            lengths.update(episode_leaf_lengths(value, path))
        elif isinstance(value, list):
            lengths[path] = len(value)
    return lengths


def truncate_episode_data(data: dict, lengths: dict[str, int], prefix: str = "") -> None:
    """Drop frames appended after ``lengths`` was captured (checkpoint rollback).

    A leaf that did not exist at capture time is truncated to zero frames.
    """
    for key, value in data.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            truncate_episode_data(value, lengths, path)
        elif isinstance(value, list):
            del value[lengths.get(path, 0):]


def create_environment_config(output_dir: str, output_file_name: str) -> ManagerBasedRLEnvCfg:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError(f"Expected ManagerBasedRLEnvCfg, got {type(env_cfg).__name__}")

    env_cfg.env_name = args_cli.task.split(":")[-1]
    env_cfg.seed = args_cli.seed
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    # green_challenge re-enables translucency on top of the "balanced" preset
    # (which disables it); match so recorded camera frames look the same.
    env_cfg.sim.render.enable_translucency = True

    if args_cli.enhanced_determinism:
        env_cfg.sim.physx.enable_enhanced_determinism = True
        print(">>> PhysX enhanced determinism: ON (recorded in dataset provenance)", flush=True)

    if args_cli.solver_iters:
        pos, vel = (int(v) for v in args_cli.solver_iters.split(","))
        env_cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count = pos
        env_cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = vel
    _props = env_cfg.scene.robot.spawn.articulation_props
    print(
        f">>> loop-closure solver: {_props.solver_position_iteration_count} position / "
        f"{_props.solver_velocity_iteration_count} velocity iterations "
        f"(green_challenge ships 4/4; recorded in dataset provenance)",
        flush=True,
    )

    if args_cli.enable_cameras:
        # Keep only the requested robot cameras: extras slow teleop rendering
        # and bloat the HDF5 via the RGB recorder term.
        selected_cameras = parse_camera_names(args_cli.cameras)
        env_cfg = strip_camera_configs(env_cfg, keep=selected_cameras)
        apply_sberchel_camera_size(env_cfg, parse_camera_size(args_cli.camera_size), selected_cameras)
        update_period = (1.0 / args_cli.camera_hz) if args_cli.camera_hz > 0 else 0.0
        for camera in selected_cameras:
            scene_attr, _ = SBERCHEL_CAMERA_OBS_KEYS[camera]
            cam_cfg = getattr(env_cfg.scene, scene_attr, None)
            if cam_cfg is not None:
                cam_cfg.update_period = update_period
        if selected_cameras:
            print(
                f">>> robot cameras: {', '.join(selected_cameras)} "
                f"({args_cli.camera_size} @ {args_cli.camera_hz:g} Hz refresh)",
                flush=True,
            )
    else:
        env_cfg = strip_camera_configs(env_cfg)
    if args_cli.xr:
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    if args_cli.retargeter != "vrd":
        # Swap the finger mapping only; the Pink export done in the env cfg's
        # __post_init__ still stands, so no need to redo it.
        build_sberchel_teleop_devices(env_cfg, args_cli.retargeter)
        print(f">>> retargeter: {args_cli.retargeter}", flush=True)

    # Optional Pink IK task overrides for in-headset tuning. These shape only how
    # commanded joint targets chase the operator's hand -- the demonstrator, not
    # the plant -- so they never invalidate demos for green_challenge.
    from pink.tasks import DampingTask as _PinkDampingTask
    from pink.tasks import FrameTask as _PinkFrameTask

    ik_tasks = env_cfg.actions.upper_body_ik.controller.variable_input_tasks
    for task in ik_tasks:
        if isinstance(task, _PinkFrameTask):
            if args_cli.ik_gain is not None:
                task.gain = float(args_cli.ik_gain)
            if args_cli.ik_lm_damping is not None:
                task.lm_damping = float(args_cli.ik_lm_damping)
        elif isinstance(task, _PinkDampingTask) and args_cli.ik_damping_cost is not None:
            task.cost = float(args_cli.ik_damping_cost)
    _frame_tasks = [t for t in ik_tasks if isinstance(t, _PinkFrameTask)]
    if _frame_tasks:
        print(
            f">>> Pink IK: gain={_frame_tasks[0].gain:g} "
            f"lm_damping={_frame_tasks[0].lm_damping:g}",
            flush=True,
        )

    if args_cli.gravity_compensation:
        env_cfg.actions.upper_body_ik.enable_gravity_compensation = True
        print(">>> gravity compensation: ON (recorded in dataset provenance)", flush=True)

    if args_cli.zest_upper:
        if args_cli.legacy_teleop_gains:
            raise SystemExit(
                "--zest_upper and --legacy_teleop_gains contradict each other: "
                "the point of the zest tract is to track the operator on the "
                "native (eval) plant via the residual, not on a stiff implicit PD."
            )
        if args_cli.gravity_compensation:
            print(
                ">>> WARNING: --gravity_compensation with --zest_upper double-"
                "compensates (the zest residual already lifts against gravity, and "
                "the eval plant runs without gravcomp). Recorded in provenance.",
                flush=True,
            )
    if args_cli.legacy_gains_keep_actuator and not args_cli.legacy_teleop_gains:
        raise SystemExit(
            "--legacy_gains_keep_actuator is a modifier for --legacy_teleop_gains; "
            "pass both."
        )
    if args_cli.legacy_teleop_gains:
        if args_cli.legacy_gains_keep_actuator:
            # No cfg change: xr_upper stays the FrictionalActuator group, and the
            # stiff gains are written into its live tensors at runtime by the
            # LocoPolicySet gain_reassert hook (after every policy gain write).
            # Explicit PD at these gains is unstable by construction -- this mode
            # exists to observe exactly that, not to record usable demos.
            print(
                ">>> GAINS: LEGACY stiff gains on the UNCHANGED FrictionalActuator "
                "(EXPERIMENT) -- explicit PD at 3000/100, dt 1/150: expect arm "
                "chatter at the effort limits. Written live via the gain hook.",
                flush=True,
            )
        else:
            # Swap the whole group cfg, not the gain tensors: the stiff values
            # are only stable as PhysX-implicit drives, and on the
            # FrictionalActuator group runtime tensor writes would be both live
            # and explicit. Rebind a fresh dict -- cfg .replace() is shallow, so
            # mutating in place would leak into the module-level robot cfg.
            actuators = dict(env_cfg.scene.robot.actuators)
            actuators["xr_upper"] = xr_upper_legacy_teleop_actuator()
            env_cfg.scene.robot.actuators = actuators
            print(
                ">>> GAINS: LEGACY stiff teleop plant on xr_upper (implicit PD, arms "
                "3000/100, fingers 400/40, neck 200/20) -- easier to teleoperate, but "
                "NOT the plant green_challenge evaluates on. Recorded in provenance.",
                flush=True,
            )

    if args_cli.spawn_height > 0.0:
        # Rebind a fresh init_state on this cfg copy (shallow-shared with the
        # module-level robot cfg). The reset event reads default_root_state, so
        # the override applies to every episode's teleport, not just spawn.
        _p = env_cfg.scene.robot.init_state.pos
        env_cfg.scene.robot.init_state = env_cfg.scene.robot.init_state.replace(
            pos=(_p[0], _p[1], float(args_cli.spawn_height))
        )
        print(
            f">>> SPAWN HEIGHT: z={args_cli.spawn_height:g} m (task cfg {_p[2]:g}; "
            "green_challenge initialises at 0.88). Recorded in provenance.",
            flush=True,
        )
    if args_cli.sim_dt_denominator > 0:
        env_cfg.sim.dt = 1.0 / float(args_cli.sim_dt_denominator)
        print(
            f">>> SIM DT: 1/{args_cli.sim_dt_denominator} s (task cfg overridden; "
            "recorded in provenance, replay auto-matches). Control still runs "
            "once per env step -- retune --walk_policy_decimation for 50 Hz.",
            flush=True,
        )
    if args_cli.gc_self_collisions:
        # Spawn through green_challenge's raw spawner instead of the
        # mask-restoring wrapper: the MJCF contype/conaffinity masks stay lost,
        # exactly like their run.sh/eval plant. The filter is applied to the
        # live stage at spawn time (never baked into the USD cache), so the
        # unwrapped func is a clean opt-out. Rebind spawn on this cfg copy --
        # .replace() is shallow and mutating in place would leak into the
        # module-level robot cfg.
        env_cfg.scene.robot.spawn = env_cfg.scene.robot.spawn.replace(
            func=GREEN_CHALLENGE_SPAWN_FUNC
        )
        print(
            ">>> SELF-COLLISIONS: green_challenge parity (MJCF masks NOT "
            "restored) -- PhysX collides every non-adjacent link pair. Expect "
            "run.sh thumb behaviour (one jammed in the palm) and extra in-hand "
            "contact pairs during grasps. Recorded in provenance.",
            flush=True,
        )
    env_cfg.recorders = SberchelXRRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
    env_cfg.recorders.export_in_record_pre_reset = True
    # Avoid exporting a stub episode (initial_state only) when closing after RESET.
    env_cfg.recorders.export_in_close = False
    return env_cfg


def capture_hold_pose_from_robot(robot) -> tuple[float, float, float]:
    """Latch current robot world XY and yaw for static-mode station-keeping."""
    (x, y, _), yaw = robot_root_state(robot)
    return x, y, yaw


def measured_neck_targets(robot) -> tuple[float, float]:
    """Neck action values that reproduce the measured neck pose (offset-corrected)."""
    name_to_id = {name: idx for idx, name in enumerate(robot.joint_names)}
    joint_pos = robot.data.joint_pos[0]
    default_pos = robot.data.default_joint_pos[0]
    yaw_id = name_to_id["neck_yaw_joint"]
    pitch_id = name_to_id["neck_pitch_joint"]
    return (
        float(joint_pos[yaw_id] - default_pos[yaw_id]),
        float(joint_pos[pitch_id] - default_pos[pitch_id]),
    )


def measured_torso_target(robot) -> float:
    """torso_yaw action value that reproduces the measured pose (offset-corrected)."""
    name_to_id = {name: idx for idx, name in enumerate(robot.joint_names)}
    torso_id = name_to_id["torso_yaw_joint"]
    return float(
        robot.data.joint_pos[0, torso_id] - robot.data.default_joint_pos[0, torso_id]
    )


def reset_teleop_state(
    *,
    env,
    policies: "LocoPolicySet",
    robot,
    teleop_interface,
    loco_mode: LocoMode = "locomotion",
) -> tuple[FrozenUpperSnapshot | None, torch.Tensor | None, torch.Tensor]:
    """Reset policy and teleop state after env.reset()."""
    # Every net, not just the live one: the idle policy would otherwise resume the
    # previous episode's action history on the next mode toggle.
    policies.reset_all()
    # One regime, always: the upper-body plant must not depend on the mode, or the
    # dataset becomes non-Markovian (same observation, different dynamics).
    # --loco_policy walk deliberately breaks that for the leg group only, which is
    # why the active policy is re-asserted below and recorded in provenance.
    # The caller enters its mode *before* this runs, so the gain write above may
    # have just landed on top of the policy's leg table; put it back.
    policies.reassert(loco_mode)
    teleop_interface.reset()
    sync_retargeter_idle_from_robot(teleop_interface, robot)
    # The red hand-joint markers are only redrawn by retarget() in static mode;
    # hide them so they don't float at the previous episode's location.
    for retargeter in getattr(teleop_interface, "_retargeters", []):
        hide = getattr(retargeter, "hide_visualization", None)
        if callable(hide):
            hide()
    idle_upper = build_idle_upper_from_robot(robot, env.device)
    return None, None, idle_upper


def main() -> None:
    output_dir, output_file_name = setup_output_directories()
    env_cfg = create_environment_config(output_dir, output_file_name)
    env_cfg.viewer.eye = (4, 1.0, 2.5)
    env_cfg.viewer.lookat = (9.5, 6.0, 0.0)

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    enable_async_episode_export(env)
    if args_cli.video:
        video_folder = Path(args_cli.video_dir) if args_cli.video_dir else Path(output_dir) / "videos"
        video_folder.mkdir(parents=True, exist_ok=True)
        name_prefix = time.strftime("%Y%m%d_%H%M%S")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_folder),
            episode_trigger=lambda episode_id: True,
            video_length=0,  # entire episode; MP4 written on reset
            disable_logger=True,
            name_prefix=name_prefix,
        )
        print(f"Viewport video recording enabled → {video_folder}", flush=True)

    base_env = env.unwrapped
    if args_cli.no_product_jitter and hasattr(base_env.scene, "first_row_jitter"):
        base_env.scene.first_row_jitter.enabled = False
    base_env.sim.reset()
    # Per-trajectory seeding: every new start pose reseeds the env with a seed
    # derived from (session seed, pose index), so each demo's start
    # configuration is reproducible on its own via env.reset(seed=...) without
    # replaying the whole session's save/discard/retry sequence.
    pose_index = 0
    episode_seed = derive_episode_seed(args_cli.seed, pose_index)
    env.reset(seed=episode_seed)

    def stamp_episode_seed() -> None:
        """Attach the current start-pose seed to the in-progress episode.

        The HDF5 handler exports ``EpisodeData.seed`` as the demo group's
        ``seed`` attribute, so every trajectory records the seed that
        generated its spawn pose and product jitter.
        """
        rm = base_env.recorder_manager
        episode = rm.get_episode(0)
        episode.seed = int(episode_seed)
        rm._episodes[0] = episode

    stamp_episode_seed()
    print(f">>> start-pose seed {episode_seed} (pose 0 of session seed {args_cli.seed})", flush=True)

    robot = base_env.scene["robot"]

    policies = LocoPolicySet(
        env,
        {"locomotion": args_cli.loco_policy, "static": "zest"},
        decimation={"walk": args_cli.walk_policy_decimation},
        device="cpu",
        # Re-write the legacy gains after every Network construction and policy
        # swap (each of those writes policy values into the xr_upper tensors).
        # Default legacy mode: the group is implicit, tensors are inert, and this
        # only keeps the gain dump/provenance truthful. --legacy_gains_keep_actuator:
        # the group is still the FrictionalActuator, its tensors ARE the plant,
        # and this hook is the mechanism that holds the stiff gains resident.
        gain_reassert=(
            (lambda: reassert_legacy_teleop_gain_tensors(robot))
            if args_cli.legacy_teleop_gains
            else None
        ),
    )
    policies.reset_all()
    walk_arm_drive = (
        args_cli.walk_arm_drive
        if policies.type_for("locomotion") == "walk"
        else "never"
    )
    # walk commands the whole body (run_kb parity): its neck targets are applied
    # whenever it drives the legs, and its torso_yaw target shares the arm-drive
    # gate below -- a torso swing drags a grasped hand through space exactly like
    # an arm swing does. zest keeps the previous behaviour: neck 0, torso held.
    loco_is_walk = policies.type_for("locomotion") == "walk"

    # --- per-trajectory pick target -----------------------------------------
    # Headset-only XR widget (XRSceneView UI, not a stage prim), placed
    # top-right of the operator's view in the robot's yaw frame — robot camera
    # sensors and viewport render products can never capture it. The item is
    # derived from the episode seed: retries keep their target, and each demo's
    # target is stamped into its HDF5 attrs.
    target_items = [s.strip() for s in args_cli.target_items.split(",") if s.strip()]
    target_hud: XrTargetHud | None = None
    if not args_cli.no_target_hud and target_items:
        target_hud = XrTargetHud(
            [target_task_instruction(item) for item in target_items],
            offset=tuple(args_cli.target_hud_offset),
            width=args_cli.target_hud_width,
        )

    def roll_target_hud() -> None:
        """Choose this episode's target; show it and stamp the demo.

        Only the session's first start pose gets a random (seed-derived) target;
        every later pose walks ``--target_items`` in list order, cycling. Both
        are functions of ``pose_index``, so a retried pose keeps its target.

        ``target_item`` and ``task_instruction`` become HDF5 attrs on the demo
        group at export (see ``stash_episode_extra_attrs``), alongside the
        ``seed`` attr the episode already carries.
        """
        if not target_items:
            return
        if pose_index == 0:
            idx = derive_target_index(episode_seed, len(target_items))
        else:
            idx = (pose_index - 1) % len(target_items)
        item = target_items[idx]
        if target_hud is not None:
            shown = target_hud.show(idx)
            if shown is None:
                # Attrs below are still stamped, but the operator cannot see the
                # task — without this warning demos get collected blind.
                print(
                    "[target-hud] WARNING: HUD is not displaying — the operator "
                    "will NOT see the pick target in the headset!",
                    flush=True,
                )
        rm = base_env.recorder_manager
        episode = rm.get_episode(0)
        stash_episode_extra_attrs(
            episode,
            target_item=item,
            task_instruction=target_task_instruction(item),
            # JSON string, not a list attr: h5py reads vlen-str attrs back as
            # object arrays, which the replay script's attr re-stamping cannot
            # write again; a plain string survives the whole pipeline.
            subtask_instructions=json.dumps(target_subtask_instructions(item)),
        )
        rm._episodes[0] = episode
        print(
            f">>> TASK: {target_task_instruction(item)} (target '{item}', seed {episode_seed})",
            flush=True,
        )

    roll_target_hud()

    def apply_native_gains() -> None:
        """green_challenge's effective PD -- the plant their VLA is evaluated on."""

    apply_native_gains()
    # Land on the starting mode's policy before anything reads the gain table, so
    # the provenance dump below describes the plant step 0 actually runs on.
    policies.reassert("locomotion")
    if policies.multi_policy:
        loco_ranges = policies.command_ranges("locomotion")
        print(
            f">>> loco policy: locomotion={policies.type_for('locomotion')} "
            f"static={policies.type_for('static')} "
            f"(body PD follows the active policy; decimation="
            f"{args_cli.walk_policy_decimation}, arms={walk_arm_drive})",
            flush=True,
        )
        print(f">>> {policies.type_for('locomotion')} command clip: {loco_ranges}", flush=True)
        for axis, flag, requested in (
            ("lin_vel_x", "speed", args_cli.speed),
            ("lin_vel_y", "strafe", args_cli.strafe),
            ("ang_vel_z", "turn", args_cli.turn),
        ):
            bounds = loco_ranges.get(axis)
            if bounds is None:
                continue
            low, high = bounds
            if requested > high or -requested < low:
                print(
                    f"[loco policy] --{flag} +/-{requested} exceeds {axis} "
                    f"[{low}, {high}]; the commander clips silently, so stick "
                    "travel is asymmetric and recorded loco_commands stay pre-clip.",
                    flush=True,
                )
    station_keeper = StationKeeper(
        lin_gain=args_cli.station_lin_gain,
        ang_gain=args_cli.station_ang_gain,
        max_lin=args_cli.station_max_lin,
        max_ang=args_cli.station_max_ang,
        pos_engage=args_cli.station_pos_deadzone,
        yaw_engage=args_cli.station_yaw_deadzone,
        min_lin=args_cli.station_min_lin,
        min_ang=args_cli.station_min_ang,
    )
    pink_hold = PinkJointHold(base_env)
    # --zest_upper starts in the classic hold too (startup mode is locomotion);
    # entering static swaps the wrapper to zest passthrough, entering locomotion
    # swaps it back.
    pink_hold.enable(robot)
    # Articulation ids for assembling the absolute 13-value zest reference
    # (torso_yaw + arms + neck). Resolved unconditionally: cheap, and loud if
    # the model ever renames a joint.
    _joint_id_by_name = {name: idx for idx, name in enumerate(robot.joint_names)}
    zest_ref_torso_id = _joint_id_by_name["torso_yaw_joint"]
    zest_ref_neck_ids = [_joint_id_by_name[n] for n in NECK_JOINT_NAMES]
    if args_cli.zest_upper:
        _loco_desc = (
            "walk drives the legs (+ arm swing per --walk_arm_drive), gain "
            "swap and reset on every C"
            if policies.type_for("locomotion") == "walk"
            else "upper body frozen on the classic hold, zest drives ONLY the "
            "legs (fixed arm reference keeps its obs consistent)"
        )
        print(
            ">>> ZEST-UPPER tract. STATIC (run_with_vla parity): arms track XR "
            "live (Pink IK solution -> zest reference), torso/arm/neck execute "
            f"zest's output. LOCOMOTION ({policies.type_for('locomotion')}): "
            f"{_loco_desc}. Per-step ref recorded as 'zest_upper_ref'.",
            flush=True,
        )
    set_render_interval(base_env, args_cli.loco_render_interval)
    torso_action_dim = (
        base_env.action_manager.get_term("torso_yaw_joint_pos").action_dim
        if "torso_yaw_joint_pos" in base_env.action_manager.active_terms
        else 0
    )
    upper_body_dim = (
        base_env.action_manager.total_action_dim
        - LEGS_ACTION_DIM
        - NECK_ACTION_DIM
        - torso_action_dim
    )
    print(
        f">>> action dim = {base_env.action_manager.total_action_dim} "
        f"(legs {LEGS_ACTION_DIM} + neck {NECK_ACTION_DIM} + upper_body {upper_body_dim} "
        f"+ torso_yaw {torso_action_dim})",
        flush=True,
    )
    print(
        f">>> loco fast-path: Pink IK hold-on + render_interval={args_cli.loco_render_interval} "
        f"(static render_interval={args_cli.static_render_interval})",
        flush=True,
    )
    print(
        f">>> GAINS: leg/torso PD comes from the loaded policy ({policies.active_type}); "
        "arm/hand/neck PD comes from the robot cfg. NOTE the table below is the "
        "python-side actuator tensor -- for ImplicitActuator groups that is the "
        "cfg value pushed to PhysX at spawn, and runtime writes to it do NOT reach "
        "the simulation. Check the actuator class before trusting these as the plant.",
        flush=True,
    )
    _gain_table = sberchel_upper_pd_gain_table(robot)
    if args_cli.legacy_teleop_gains:
        _gain_label = (
            "LEGACY stiff gains on FrictionalActuator (EXPERIMENT) + policy legs"
            if args_cli.legacy_gains_keep_actuator
            else "LEGACY stiff teleop upper body + policy legs"
        )
    else:
        _gain_label = "green_challenge-native"
    print(
        f">>> PD gains ({_gain_label}, whole robot): "
        + ", ".join(f"{n}={k:g}/{d:g}" for n, (k, d) in sorted(_gain_table.items())),
        flush=True,
    )

    # --- dataset provenance ---------------------------------------------------
    # Record the joint ordering and the plant the demos were produced on. Every
    # joint-space consumer (green_challenge's VLA groups in particular) needs the
    # name order to slice these vectors correctly, and `find_joints` does NOT
    # preserve the order given in the cfg -- so it must be read at runtime rather
    # than assumed. Without this a converter is guessing, and a wrong guess is
    # silent: our 12 finger values and theirs are both length 12 but ordered
    # differently, so thumb commands would quietly land on the pinky.
    def _write_dataset_provenance() -> None:
        handler = getattr(base_env.recorder_manager, "_dataset_file_handler", None)
        if handler is None or not hasattr(handler, "add_env_args"):
            print("[provenance] no dataset file handler; joint order NOT recorded", flush=True)
            return
        upper_term = base_env.action_manager.get_term("upper_body_ik")
        # store_products rows are indexed by this prim order; without it,
        # identifying which product a row belongs to requires re-instantiating
        # the scene. The full list is too large for a 64 KB HDF5 attr, so it
        # goes to a sidecar JSON bound to the file by the digest below.
        _store_snapshot = getattr(base_env.scene, "_store_rigid_snapshot", None)
        _store_prim_paths = list(_store_snapshot.prim_paths) if _store_snapshot is not None else []
        try:
            handler.add_env_args(
                {
                    # Session provenance: the session seed and the code version that
                    # produced these demos. Each demo additionally carries its own
                    # start-pose seed in the demo group's "seed" attr, derived as
                    # described below; reproducing a specific start configuration
                    # needs only that per-demo seed (plus the same code version).
                    "seed": int(args_cli.seed),
                    "sberchel_episode_seed_scheme": (
                        "blake2b(f'{session_seed}:{pose_index}', digest_size=8) % 2**31; "
                        "reseeded via env.reset(seed=...) at every new start pose; "
                        "per-demo value in demo attrs['seed']"
                    ),
                    "git_commit": repo_git_commit(),
                    "first_row_jitter": (
                        vars(base_env.scene.first_row_jitter)
                        if hasattr(base_env.scene, "first_row_jitter")
                        else None
                    ),
                    "sberchel_joint_names": list(robot.data.joint_names),
                    "sberchel_upper_controlled_joint_names": list(
                        upper_term._controlled_joint_names
                    ),
                    "sberchel_hand_joint_names": list(upper_term._hand_joint_names),
                    "sberchel_upper_pd_gains": {n: list(v) for n, v in _gain_table.items()},
                    # True when --legacy_teleop_gains ran the session on the old
                    # stiff implicit XR plant (single regime, both modes) instead
                    # of the green_challenge-native soft PD. Unlike the retired
                    # two-regime flag of the same name, the plant never switches
                    # mid-session.
                    # True (default): the MuJoCo-faithful contype/conaffinity
                    # filter from sberchel_self_collision was applied at spawn.
                    # False (--gc_self_collisions): green_challenge parity, the
                    # importer's masks-lost collision set -- every non-adjacent
                    # link pair collides, thumb-vs-palm included. Replay
                    # auto-matches this from provenance.
                    "sberchel_mjcf_collision_filters": not args_cli.gc_self_collisions,
                    # Physics step the session actually ran (s). Replay matches
                    # it; the dataset artifact itself never exposes it (achieved
                    # states, 30 fps export).
                    "sberchel_sim_dt": float(base_env.physics_dt),
                    # Root z the robot spawns/resets at. Replay restores the
                    # recorded initial state verbatim, so no matching needed.
                    "sberchel_spawn_height": (
                        float(args_cli.spawn_height)
                        if args_cli.spawn_height > 0.0
                        else float(SBERCHEL_STAND_HEIGHT)
                    ),
                    "sberchel_legacy_stiff_teleop": bool(args_cli.legacy_teleop_gains),
                    # Which actuator model carried the stiff gains: 'implicit'
                    # (the faithful legacy plant: PhysX drives, no friction) or
                    # 'frictional' (--legacy_gains_keep_actuator experiment:
                    # same gains live on green_challenge's explicit actuator).
                    # null when the session ran the native soft plant.
                    "sberchel_legacy_gains_actuator": (
                        ("frictional" if args_cli.legacy_gains_keep_actuator else "implicit")
                        if args_cli.legacy_teleop_gains
                        else None
                    ),
                    "sberchel_squeeze_scale": float(args_cli.squeeze_scale),
                    "sberchel_squeeze_binary": bool(args_cli.squeeze_binary),
                    "sberchel_squeeze_on_value": float(args_cli.squeeze_on_value),
                    # Which finger mapping produced these demos. Not a plant
                    # difference (the robot is identical), but it is a
                    # demonstrator-style difference, so keep episodes separable.
                    "sberchel_retargeter": str(args_cli.retargeter),
                    "sberchel_gravity_compensation": bool(
                        getattr(upper_term.cfg, "enable_gravity_compensation", False)
                    ),
                    # Plant-level PhysX setting: replay must run with the same
                    # value or the simulation itself differs from the recording.
                    "sberchel_enhanced_determinism": bool(args_cli.enhanced_determinism),
                    # Also plant-level, and together they decide how rigid the
                    # eight knee/ankle/wrist loop closures are: they are solved
                    # as maximal-coordinate constraints, so the solver type and
                    # its iteration budget change the leg dynamics. PGS (0) at
                    # 4/4 is what green_challenge evaluates on and what we
                    # mirror; TGS (1) at 4/4 makes the zest policy limit-cycle.
                    "sberchel_solver_position_iterations": int(
                        base_env.scene.cfg.robot.spawn.articulation_props.solver_position_iteration_count
                    ),
                    "sberchel_solver_velocity_iterations": int(
                        base_env.scene.cfg.robot.spawn.articulation_props.solver_velocity_iteration_count
                    ),
                    "sberchel_physx_solver_type": int(base_env.cfg.sim.physx.solver_type),
                    # CPU vs GPU PhysX are different implementations, so this is
                    # part of the plant too. green_challenge evaluates on "cpu".
                    "sberchel_sim_device": str(base_env.cfg.sim.device),
                    # Demonstrator-side Pink IK shaping (not a plant difference,
                    # but affects how closely demos track the operator's hand).
                    "sberchel_pink_ik_tasks": [
                        {
                            "type": type(t).__name__,
                            "gain": float(t.gain),
                            "lm_damping": float(t.lm_damping),
                        }
                        if hasattr(t, "frame")  # FrameTask and subclasses
                        else {"type": type(t).__name__, "cost": float(getattr(t, "cost", 0.0) or 0.0)}
                        for t in upper_term.cfg.controller.variable_input_tasks
                    ],
                    "sberchel_action_layout": (
                        "legs(12) + neck(2) + upper_body_ik(26) + torso_yaw(1)"
                    ),
                    "sberchel_teleop_state_layout": (
                        "is_locomotion, squeeze, legacy_gains (1 iff the session "
                        "ran with --legacy_teleop_gains; constant per session)"
                    ),
                    # Per-demo attrs stamped at save: mode_switch_steps = step
                    # indices where the operator's C / Create mode toggle took
                    # effect (the flips of teleop_state[:,0] that survived
                    # checkpoint rollbacks); mode_switch_to = the mode entered
                    # at each (1 locomotion, 0 static). Segment k of the demo is
                    # [step_{k-1}, step_k); a standard 3-press demo splits into
                    # walk-to-shelf / take-object / walk-to-table / place-object.
                    "sberchel_mode_switch_attrs": (
                        "mode_switch_steps: first step of each new mode segment "
                        "(= teleop_state[:,0] flip indices, i.e. surviving C / "
                        "Create presses); mode_switch_to: mode entered at that "
                        "step (1 locomotion, 0 static). Segment boundaries: "
                        "step 0 (always locomotion), each mode_switch_steps "
                        "entry, episode end — a 3-switch demo yields 4 subtasks"
                    ),
                    # Language decomposition of each demo. Boundaries are NOT
                    # stamped: subtasks 1-4 split at mode_switch_steps; the
                    # place->retract boundary (no button press exists there) is
                    # derived offline from object motion alone — the step after
                    # which the carried product stays at its final rest pose in
                    # store_products (see sberchel_hdf5_to_lerobot.py). Squeeze
                    # is deliberately not part of the rule: hand-tracking demos
                    # can record zero squeeze for the whole session.
                    "sberchel_subtask_scheme": (
                        "per-demo attr 'subtask_instructions' (JSON list of 5 "
                        "strings): walk-to-shelf / pick / carry-to-table / "
                        "place-in-container / retract-hand prompts. Boundaries: "
                        "subtasks 1-4 at mode_switch_steps; place->retract at "
                        "the step after which the carried product (max-"
                        "displacement store product) rests at its final pose in "
                        "store_products for the remainder of the episode"
                    ),
                    # Which policy produced each segment. teleop_state[:,0] is the
                    # per-step selector into this map, so no extra channel is needed.
                    "sberchel_loco_policy_by_mode": {
                        "locomotion": policies.type_for("locomotion"),
                        "static": policies.type_for("static"),
                    },
                    "sberchel_loco_policy_weights": policies.weights_by_type(),
                    # The body-joint PD arrives with the policy, so a per-mode
                    # policy also means a per-mode leg/arm plant; teleop_state[:,0]
                    # is what segments a dataset back apart.
                    "sberchel_loco_policy_gain_switch": policies.multi_policy,
                    # Entering policy is reset at every swap; replay applies the same
                    # rule at the same indices, derived from teleop_state[:,0].
                    "sberchel_loco_policy_reset_on_switch": True,
                    "sberchel_walk_arm_drive": walk_arm_drive,
                    "sberchel_walk_policy_decimation": int(args_cli.walk_policy_decimation),
                    # loco_commands are recorded PRE-clip: the commander clamps to
                    # these ranges without warning, so a consumer reading the recorded
                    # command as executed truth is wrong wherever it saturates.
                    "sberchel_loco_command_clip": policies.command_ranges("locomotion"),
                    # walk drives all 25 body joints: neck whenever it drives the
                    # legs, torso_yaw under the same engage gate as the arms. zest
                    # keeps the old behaviour (neck 0, torso_yaw held at default)
                    # -- unless --zest_upper, where zest's full torso block
                    # (torso_yaw + arms + neck) is what the robot executes.
                    "sberchel_loco_policy_unmapped_outputs": (
                        []
                        if (loco_is_walk or args_cli.zest_upper)
                        else [
                            "torso_yaw_joint",
                            "neck_yaw_joint",
                            "neck_pitch_joint",
                        ]
                    ),
                    # True when the walk policy's neck targets are applied during
                    # locomotion (unconditional under walk).
                    "sberchel_walk_neck_drive": loco_is_walk,
                    # Gate for the walk policy's torso_yaw target: same value and
                    # semantics as sberchel_walk_arm_drive ('never' under zest).
                    "sberchel_walk_torso_drive": walk_arm_drive if loco_is_walk else "never",
                    # --zest_upper: split-parity zest tract. STATIC segments
                    # (teleop_state[:,0]==0, run_with_vla parity): arms/torso/
                    # neck execute zest(ref + residual), ref = previous step's
                    # raw Pink IK arm solution + headset neck (live tracking).
                    # LOCOMOTION segments: the --loco_policy net runs through
                    # the classic machinery -- upper body PD-held at commanded
                    # targets (hold, IK skipped) unless walk's arm drive is
                    # engaged (sberchel_walk_arm_drive); the frozen arm pose is
                    # still fed as the zest ref (no-op for walk's commander) so
                    # zest's observations stay consistent across swaps. The
                    # ref is recorded per step as 'zest_upper_ref' (layout
                    # below, absolute radians). Raw 'actions' upper 26D =
                    # operator IK intent (frozen snapshot during locomotion);
                    # processed_actions/joint_pos_target = what ran.
                    "sberchel_zest_upper": bool(args_cli.zest_upper),
                    "sberchel_zest_upper_ref_layout": (
                        "torso_yaw + POLICY_ARM_JOINT_NAMES(10) + neck_yaw + "
                        "neck_pitch; absolute rad; fed to zest via "
                        "Network.set_joint_overrides as 'torso_joint_pos'"
                    ),
                    # Raw 'actions' still encode the frozen-snapshot IK targets while
                    # the arms were actually driven by the policy (and under
                    # --zest_upper the operator's IK intent while zest ran the
                    # arms); only processed_actions describe what ran.
                    "sberchel_actions_valid_replay_sources": (
                        ["policy_loco", "processed_joint_targets"]
                        if (walk_arm_drive != "never" or args_cli.zest_upper)
                        else ["policy_loco", "processed_joint_targets", "actions"]
                    ),
                    # Pick target shown to the operator. Each demo group carries
                    # 'target_item' and 'task_instruction' attrs (authoritative);
                    # the scheme below documents how they were assigned.
                    "sberchel_target_items": target_items,
                    "sberchel_target_scheme": (
                        "pose 0: items[int.from_bytes(blake2b(f'{seed}:target', "
                        "digest_size=8).digest(), 'little') % len(items)] "
                        "with seed = demo attrs['seed']; "
                        "pose n>=1: items[(n - 1) % len(items)] (list order, cycling); "
                        "per-demo attrs 'target_item'/'task_instruction' are authoritative"
                    ),
                    "sberchel_task_template": "Pick a/an {target_item} and place it on the table.",
                    "sberchel_store_prim_count": len(_store_prim_paths),
                    "sberchel_store_prim_paths_blake2b": hashlib.blake2b(
                        "\n".join(_store_prim_paths).encode(), digest_size=8
                    ).hexdigest(),
                }
            )
            if _store_prim_paths:
                _sidecar = Path(args_cli.dataset_file).with_suffix(".store_prims.json")
                _sidecar.write_text(json.dumps(_store_prim_paths))
                print(f">>> store product prim order -> {_sidecar}", flush=True)
            print(">>> dataset provenance written (joint order + gains)", flush=True)
        except Exception as exc:  # never lose a recording session over metadata
            print(f"[provenance] could not write env args: {exc!r}", flush=True)

    _write_dataset_provenance()

    should_reset = False
    should_discard_reset = False
    should_next_reset = False
    # Start state of the episode in progress; discard-reset (D-pad down / X) reapplies
    # it after env.reset() so a failed attempt can be retried from the same spawn
    # with the same (jittered) product placement.
    episode_start_root_pose = robot.data.root_state_w[:, :7].clone()
    episode_start_products = (
        base_env.scene.capture_store_rigid_body_state(base_env.sim)
        if hasattr(base_env.scene, "capture_store_rigid_body_state")
        else None
    )
    teleoperation_active = (not args_cli.xr) or args_cli.auto_start
    running_recording_instance = (not args_cli.xr) or args_cli.auto_start
    step_count = 0
    first_active_logged = False
    loco_mode: LocoMode = "locomotion"
    hold_xy_yaw: tuple[float, float, float] | None = None
    frozen_snapshot: FrozenUpperSnapshot | None = None
    left_hold_snapshot: FrozenUpperSnapshot | None = None
    last_teleop_upper: torch.Tensor | None = None
    last_neck_targets: tuple[float, float] = (0.0, 0.0)
    # Last commanded torso_yaw action (offset from default, i.e. action units).
    # Walk locomotion writes the policy's target here; static mode ramps it back
    # to 0 so the torso ends where every pre-torso-term demo held it.
    last_torso_target = 0.0
    # --zest_upper: the arm reference fed to zest during locomotion, captured
    # (absolute, [num_envs, 10]) on every locomotion entry -- the last IK solve,
    # or the measured pose before any static teleop. The reference keeps zest's
    # observations consistent with the PD-held arms; its arm OUTPUT is discarded
    # for the whole locomotion segment (run_kb/run.py parity -- see the apply
    # site below).
    loco_zest_ref_arms: torch.Tensor | None = None
    # Post-reset settle re-latch for the locomotion hold, same physics as
    # _LEFT_HOLD_SETTLE_STEPS: the reset pose writes both thumb yaws at -0.1,
    # inside the palm collision hull, and the hold captures BEFORE PhysX kicks
    # them out to the ~0.24 contact equilibrium. Left as-is, the whole
    # locomotion segment commands the unreachable -0.1 into the palm (kp*0.34 >
    # the 8.5 Nm finger limit) and both thumbs chatter from step 0. Once the
    # pose has settled, re-capture the hold from the MEASURED pose. Cleared on
    # static entry: from there the operator owns the hand and the
    # grip-preserving commanded-target capture is the right one again.
    pink_hold_settle_pending = True
    last_hz_t = time.time()
    hz_window_steps = 0
    effective_hz = 0.0
    # --idle_pause: wall-clock of the last nonzero loco command. None means "no
    # command since entering locomotion mode" -> frozen immediately, no settle.
    last_loco_cmd_t: float | None = None
    idle_paused_prev = False
    # Auto-checkpoint taken at every loco->static switch: full scene physics
    # state + policy carry-over + episode/video frame counts for rollback.
    sim_checkpoint: dict | None = None
    # --static_blend_time: on every static-mode entry, ramp the upper/neck
    # commands from this held pose into the live XR targets instead of snapping.
    static_blend_from: FrozenUpperSnapshot | None = None
    static_blend_neck_from: tuple[float, float] = (0.0, 0.0)
    static_blend_torso_from = 0.0
    static_blend_steps_done = 0
    static_blend_total = 0
    # --loco_blend_time: mirror of the above for the arms when the walk policy
    # drives them. Its nominal pose is arms-at-sides while the operator enters
    # locomotion from a reach, so the raw delta can be radians.
    loco_arm_blend_from: torch.Tensor | None = None
    loco_torso_blend_from = 0.0
    loco_arm_blend_steps_done = 0
    loco_arm_blend_total = 0
    # Whether the policy currently owns the arms (--walk_arm_drive when_free
    # hands them back the moment the operator squeezes).
    arm_drive_engaged = False
    # True once the walk policy has moved the arms this locomotion segment, so
    # the static ramp knows the frozen snapshot no longer describes them.
    arms_moved_by_policy = False

    def seed_loco_arm_blend() -> None:
        """Start the arm/torso ramp from whatever is being commanded right now."""
        nonlocal loco_arm_blend_from, loco_torso_blend_from
        nonlocal loco_arm_blend_steps_done, loco_arm_blend_total
        # torso_yaw rides the same ramp as the arms (same gate, same alpha).
        loco_torso_blend_from = last_torso_target
        if walk_arm_drive == "never" or args_cli.loco_blend_time <= 0.0:
            loco_arm_blend_from = None
            return
        loco_arm_blend_from = pink_hold.arm_targets()
        loco_arm_blend_steps_done = 0
        step_dt = float(getattr(base_env, "step_dt", 1.0 / 50.0))
        loco_arm_blend_total = max(1, round(args_cli.loco_blend_time / step_dt))

    def enter_locomotion_mode(*, capture_frozen: bool) -> None:
        nonlocal loco_mode, hold_xy_yaw, frozen_snapshot, last_neck_targets
        nonlocal last_loco_cmd_t, arm_drive_engaged, arms_moved_by_policy
        nonlocal loco_zest_ref_arms
        # First: the leg PD that follows the active policy has to be resident
        # before the policy it belongs to produces a target.
        policies.activate("locomotion")
        loco_mode = "locomotion"
        hold_xy_yaw = None
        last_loco_cmd_t = None
        arm_drive_engaged = False
        arms_moved_by_policy = False
        last_neck_targets = (0.0, 0.0)
        if args_cli.zest_upper:
            # Freeze the zest arm reference where the operator's IK last left
            # it, ON EVERY locomotion entry. Capturing lazily in the loop (only
            # when frozen_snapshot was None) kept a stale ref across C-presses:
            # the C entry pre-fills frozen_snapshot, so the ref stayed at its
            # episode-start capture -- the default pose -- and zest's leg
            # inference saw arms it did not have.
            _ref = pink_hold.last_ik_arm_solution()
            loco_zest_ref_arms = (
                _ref.clone()
                if _ref is not None
                else robot.data.joint_pos[:, pink_hold.arm_joint_ids].clone()
            )
        if capture_frozen:
            frozen_snapshot = capture_frozen_upper(robot, last_teleop_upper)
        # The upper plant never moves with the mode, so C changes what is being
        # commanded, not what the robot physically is. --loco_policy walk breaks
        # that for the leg group alone.
        pink_hold.enable(robot)
        seed_loco_arm_blend()
        set_render_interval(base_env, args_cli.loco_render_interval)

    def enter_static_mode() -> None:
        nonlocal loco_mode, hold_xy_yaw, frozen_snapshot, last_neck_targets
        nonlocal left_hold_snapshot, last_torso_target
        nonlocal static_blend_from, static_blend_neck_from, static_blend_torso_from
        nonlocal static_blend_steps_done, static_blend_total
        nonlocal loco_arm_blend_from, arm_drive_engaged
        nonlocal pink_hold_settle_pending
        # The operator owns the hand from here; later holds must capture the
        # grip-preserving commanded targets, not the measured pose.
        pink_hold_settle_pending = False
        policies.activate("static")
        loco_mode = "static"
        loco_arm_blend_from = None
        arm_drive_engaged = False
        hold_xy_yaw = capture_hold_pose_from_robot(robot)
        station_keeper.reset()
        sync_retargeter_idle_from_robot(teleop_interface, robot)
        if args_cli.disable_left_hand:
            # The left hand is never used, so the measured pose is the hold target.
            left_hold_snapshot = capture_frozen_upper(robot)
        if args_cli.static_blend_time > 0.0:
            # Ramp from what is being commanded right now (the loco hold pose if
            # there is one, else the measured pose) into the live XR targets.
            static_blend_from = (
                # The walk policy has been moving the arms, so frozen_snapshot
                # describes where they were at locomotion *entry*, not where they
                # are now -- ramping from it would snap, which is the one thing
                # the ramp exists to prevent. Measured pose is the truth here.
                capture_frozen_upper(robot)
                if arms_moved_by_policy
                else frozen_snapshot
                if frozen_snapshot is not None
                else capture_frozen_upper(robot, last_teleop_upper)
            )
            static_blend_neck_from = last_neck_targets
            # Torso ramps back to its default (0 offset) alongside the arm/neck
            # blend, from wherever the walk policy last commanded it.
            static_blend_torso_from = last_torso_target
            static_blend_steps_done = 0
            step_dt = float(getattr(base_env, "step_dt", 1.0 / 50.0))
            static_blend_total = max(1, round(args_cli.static_blend_time / step_dt))
        else:
            static_blend_from = None
            # No ramp: snap the torso home, same as the neck/arms snap.
            last_torso_target = 0.0
        frozen_snapshot = None
        last_neck_targets = (0.0, 0.0)
        if args_cli.zest_upper:
            # Static teleop runs the zest passthrough: live Pink IK (its raw arm
            # solution feeds zest's reference), arms execute zest's output,
            # wrists/fingers take the IK directly.
            pink_hold.enable_zest_passthrough()
        else:
            pink_hold.disable()
        set_render_interval(base_env, args_cli.static_render_interval)

    def apply_left_hold(upper_body: torch.Tensor) -> torch.Tensor:
        """Overwrite left wrist/finger slots with the --disable_left_hand hold."""
        if not args_cli.disable_left_hand or left_hold_snapshot is None:
            return upper_body
        left_hold = build_upper_from_frozen_snapshot(
            robot, left_hold_snapshot, upper_body.device
        ).to(upper_body.dtype)
        upper_body[..., _LEFT_WRIST_SLICE] = left_hold[_LEFT_WRIST_SLICE]
        for dim in _LEFT_FINGER_DIMS:
            upper_body[..., dim] = left_hold[dim]
        return upper_body

    def reset_env() -> None:
        nonlocal should_reset
        should_reset = True
        print("Reset requested — episode will be saved and environment reset on next step")

    def discard_reset_env() -> None:
        nonlocal should_discard_reset
        should_discard_reset = True
        print(
            "Discard reset requested — episode will NOT be saved; retrying the SAME start pose"
        )

    def next_pose_reset_env() -> None:
        nonlocal should_next_reset
        should_next_reset = True
        print(
            "Next-pose reset requested — episode will NOT be saved; advancing to a NEW start pose"
        )

    def capture_checkpoint() -> None:
        """Snapshot everything needed to roll the episode back to this instant.

        The scene state covers the robot (root + joints + velocities) and every
        rigid object, so a dropped item is teleported back too. Policy carry-over
        is delegated to ``LocoPolicySet.capture`` -- for zest that is just
        ``controller.actions``, but the walk policy also keeps a 15-frame
        observation history, and rolling back one without the other leaves it on
        a timeline the restored scene does not match. Episode
        and video buffers are rolled back by frame count, which is exact because
        restore reproduces the checkpoint state bit-for-bit — the stitched
        recording stays continuous.
        """
        nonlocal sim_checkpoint
        episode = base_env.recorder_manager.get_episode(0)
        sim_checkpoint = {
            "scene_state": base_env.scene.get_state(is_relative=False),
            # Store products are plain USD rigid bodies, not registered scene
            # assets, so scene.get_state() misses them — snapshot via PhysX view.
            "store_state": (
                base_env.scene.capture_store_rigid_body_state(base_env.sim)
                if hasattr(base_env.scene, "capture_store_rigid_body_state")
                else None
            ),
            "policy_state": policies.capture(),
            # The 'actions' obs term reads action_manager.action; without this
            # rollback the first post-restore obs row records the abandoned
            # timeline's last action.
            "last_action": base_env.action_manager.action.clone(),
            "prev_action": base_env.action_manager.prev_action.clone(),
            "episode_lengths": episode_leaf_lengths(episode.data),
            "video_frames": viewport_video_frame_count(env),
        }
        print("[checkpoint] captured (V / R1 rolls sim + recording back here)", flush=True)

    def restore_checkpoint() -> None:
        nonlocal last_teleop_upper, last_neck_targets, last_torso_target, frozen_snapshot
        if not teleoperation_active:
            print("[checkpoint] ignored — recording is stopped", flush=True)
            return
        if sim_checkpoint is None:
            print("[checkpoint] none captured yet this episode (press C to enter static mode first)", flush=True)
            return
        base_env.scene.reset_to(sim_checkpoint["scene_state"], None, is_relative=False)
        if sim_checkpoint["store_state"] is not None:
            restored = base_env.scene.restore_store_rigid_body_state(
                base_env.sim, sim_checkpoint["store_state"]
            )
            print(f"[checkpoint] store products restored: {restored}", flush=True)
        base_env.action_manager._action.copy_(sim_checkpoint["last_action"])
        base_env.action_manager._prev_action.copy_(sim_checkpoint["prev_action"])
        # scene.reset_to only writes sim state; env.obs_buf is recomputed inside
        # env.step/reset, so the next pre-step recorder tick would log the
        # abandoned timeline's observation (one stale obs row per restore).
        # Mirror env.reset_to: forward kinematics, then refresh the buffer.
        base_env.sim.forward()
        base_env.obs_buf = base_env.observation_manager.compute()
        truncate_episode_data(
            base_env.recorder_manager.get_episode(0).data,
            sim_checkpoint["episode_lengths"],
        )
        truncate_viewport_video_frames(env, sim_checkpoint["video_frames"])
        # Commanded targets were overwritten by scene.reset_to; treat them as
        # stale until the next stepped frame, same as after env.reset().
        pink_hold.invalidate()
        last_teleop_upper = None
        # Blend seeds: the stale pre-restore commands no longer match the scene,
        # so restart the static ramp from the restored measured pose.
        frozen_snapshot = None
        last_neck_targets = measured_neck_targets(robot)
        # Static blend (inside enter_static_mode) ramps the torso home from here.
        last_torso_target = measured_torso_target(robot)
        enter_static_mode()
        # After the mode entry, never before: entering static activates its policy,
        # and a policy swap resets the net -- which would wipe the state we are
        # restoring. Rollback has to stay bit-exact or the frame-count truncation
        # above stops lining up with the scene.
        policies.restore(sim_checkpoint["policy_state"])
        if anchor_teleporter is not None:
            anchor_teleporter.teleport_to_robot(robot)
        if target_hud is not None:
            target_hud.refresh_soon()
        print(
            "[checkpoint] restored — static teleop re-latched at the checkpoint pose "
            "(checkpoint kept; V / R1 works again)",
            flush=True,
        )

    def toggle_loco_mode() -> None:
        if not teleoperation_active:
            return
        if loco_mode == "locomotion":
            enter_static_mode()
            capture_checkpoint()
            if target_hud is not None:
                # Re-snapshot the panel into the operator's current view.
                target_hud.refresh_soon()
            print(
                f"Mode: STATIC teleop (station-keep at "
                f"xy=({hold_xy_yaw[0]:.3f}, {hold_xy_yaw[1]:.3f}) yaw={hold_xy_yaw[2]:.3f}; "
                f"headset drives neck; Pink IK on)",
                flush=True,
            )
        else:
            enter_locomotion_mode(capture_frozen=True)
            if args_cli.zest_upper:
                print(
                    f"Mode: LOCOMOTION (WASD velocity to "
                    f"{policies.type_for('locomotion')}; hands are ignored until "
                    "you return to static)",
                    flush=True,
                )
            else:
                print("Mode: LOCOMOTION (WASD + Pink IK hold-off + soft arm PD)", flush=True)

    def start_recording_instance() -> None:
        nonlocal running_recording_instance, teleoperation_active, first_active_logged
        nonlocal frozen_snapshot, last_teleop_upper
        running_recording_instance = True
        teleoperation_active = True
        first_active_logged = False
        frozen_snapshot = None
        last_teleop_upper = None
        enter_locomotion_mode(capture_frozen=False)
        sync_retargeter_idle_from_robot(teleop_interface, robot)
        if target_hud is not None:
            # First event that is guaranteed to come from a live XR session.
            target_hud.refresh_soon()
        if args_cli.zest_upper:
            print(
                "Recording started (zest-upper: WASD locomotion with frozen upper "
                "body; press C for static teleop — arms live through zest)"
            )
        else:
            print("Recording started (WASD locomotion fast-path; press C for static teleop)")

    def stop_recording_instance() -> None:
        nonlocal running_recording_instance, teleoperation_active
        nonlocal hold_xy_yaw, frozen_snapshot, last_teleop_upper, last_neck_targets
        running_recording_instance = False
        teleoperation_active = False
        hold_xy_yaw = None
        frozen_snapshot = None
        last_teleop_upper = None
        last_neck_targets = (0.0, 0.0)
        enter_locomotion_mode(capture_frozen=False)
        print("Recording paused")

    def toggle_recording_instance() -> None:
        """Gamepad stand-in for the XR START / STOP buttons."""
        if running_recording_instance:
            stop_recording_instance()
        else:
            start_recording_instance()

    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_env,
        "START": start_recording_instance,
        "STOP": stop_recording_instance,
        "RESET": reset_env,
    }

    # Head tracking required for static-mode neck look.
    teleop_interface = create_sberchel_xr_teleop_interface(
        env_cfg, args_cli.teleop_device, teleoperation_callbacks, require_head_tracking=True
    )
    teleop_interface.reset()
    sync_retargeter_idle_from_robot(teleop_interface, robot)
    idle_upper = build_idle_upper_from_robot(robot, base_env.device)

    # Anchor prim only exists once the OpenXR device has been created above.
    # Latch the anchor offset against the robot's NOMINAL init pose (what
    # ``xr.anchor_pos`` was tuned for), not the live pose: the spawn is
    # randomized per episode, so the live pose would bake a random offset
    # into every teleport.
    nominal_init = env_cfg.scene.robot.init_state
    anchor_teleporter = (
        XrAnchorTeleporter(
            env_cfg.xr,
            robot,
            ref_pose=(
                tuple(float(v) for v in nominal_init.pos),
                yaw_from_quat_wxyz(nominal_init.rot),
            ),
        )
        if args_cli.xr
        else None
    )
    if anchor_teleporter is not None:
        # The OpenXR device placed the anchor at the static cfg pose; move it
        # onto the randomized spawn so the operator starts at the robot.
        anchor_teleporter.teleport_to_robot(robot)

    def teleport_view_to_robot() -> None:
        """Snap the XR view back onto the robot at the episode-start offset."""
        if anchor_teleporter is None:
            print("[XR] Teleport ignored: no XR anchor (run with hand tracking).", flush=True)
            return
        pose = anchor_teleporter.teleport_to_robot(robot)
        if pose is None:
            print("[XR] Teleport failed: could not write the XR anchor prim.", flush=True)
            return
        if target_hud is not None:
            target_hud.refresh_soon()
        (ax, ay, az), ayaw = pose
        print(
            f"[XR] View teleported to robot: anchor xy=({ax:.3f}, {ay:.3f}) "
            f"z={az:.3f} yaw={ayaw:.3f}",
            flush=True,
        )

    # Keyboard and gamepad expose the same poll/command/pressed_keys surface, so the
    # loop below is agnostic to which one (or both) is driving.
    controllers: list[RecordKeyboard | RecordGamepad] = []
    use_keyboard = args_cli.control_device in ("keyboard", "both")
    use_gamepad = args_cli.control_device in ("gamepad", "both")
    if use_keyboard:
        controllers.append(
            RecordKeyboard(
                args_cli.speed,
                args_cli.turn,
                args_cli.strafe,
                on_toggle_mode=toggle_loco_mode,
                on_save_reset=reset_env,
                on_discard_reset=discard_reset_env,
                on_next_pose_reset=next_pose_reset_env,
                on_teleport_view=teleport_view_to_robot,
                on_restore_checkpoint=restore_checkpoint,
            )
        )
    if use_gamepad:
        controllers.append(
            RecordGamepad(
                args_cli.speed,
                args_cli.turn,
                args_cli.strafe,
                device=args_cli.gamepad_dev,
                deadzone=args_cli.gamepad_deadzone,
                use_sticks=not args_cli.gamepad_no_sticks,
                debug=args_cli.gamepad_debug,
                on_toggle_mode=toggle_loco_mode,
                on_save_reset=reset_env,
                on_discard_reset=discard_reset_env,
                on_next_pose_reset=next_pose_reset_env,
                on_teleport_view=teleport_view_to_robot,
                on_toggle_recording=toggle_recording_instance,
                on_restore_checkpoint=restore_checkpoint,
            )
        )

    def poll_controllers() -> None:
        for controller in controllers:
            controller.poll()

    def controller_command() -> tuple[float, float, float]:
        """Largest-magnitude command per axis across the active input devices."""
        vx = vy = wz = 0.0
        for controller in controllers:
            cx, cy, cz = controller.command()
            vx = cx if abs(cx) > abs(vx) else vx
            vy = cy if abs(cy) > abs(vy) else vy
            wz = cz if abs(cz) > abs(wz) else wz
        return vx, vy, wz

    def controller_pressed() -> str:
        active = [c.pressed_keys() for c in controllers]
        return " ".join(a for a in active if a != "-") or "-"

    def controller_squeeze() -> float:
        """Strongest squeeze command across the active input devices."""
        value = 0.0
        for controller in controllers:
            getter = getattr(controller, "squeeze", None)
            if getter is not None:
                value = max(value, float(getter()))
        return value

    # --squeeze_binary: Schmitt-trigger the raw trigger travel into 0 / on_value.
    # Hysteresis (0.6 up, 0.4 down), not a single threshold: a trigger held near
    # the switching point would otherwise flutter the grip open mid-carry.
    _SQUEEZE_ENGAGE = 0.6
    _SQUEEZE_RELEASE = 0.4
    _squeeze_latched = False

    def shape_squeeze(raw: float) -> float:
        nonlocal _squeeze_latched
        if not args_cli.squeeze_binary:
            return raw
        if _squeeze_latched:
            if raw < _SQUEEZE_RELEASE:
                _squeeze_latched = False
        elif raw > _SQUEEZE_ENGAGE:
            _squeeze_latched = True
        return float(args_cli.squeeze_on_value) if _squeeze_latched else 0.0

    _squeeze_sinks = [
        r for r in getattr(teleop_interface, "_retargeters", []) if hasattr(r, "set_squeeze")
    ]
    for _sink in _squeeze_sinks:
        _sink._squeeze_scale = float(args_cli.squeeze_scale)
    if not _squeeze_sinks:
        print(
            "[squeeze] no retargeter accepts a squeeze command — grip force will come "
            "only from your own hand curl, which is usually too shallow to hold anything.",
            flush=True,
        )

    def push_squeeze(value: float) -> None:
        for retargeter in _squeeze_sinks:
            retargeter.set_squeeze(value)

    # --- grip instrumentation -------------------------------------------------
    # Grip force on a position-controlled hand is clip(k_p*err - k_d*qdot, +-limit)
    # where err = commanded - achieved. Reporting it turns "does the grasp hold?"
    # into a measurement during the session instead of a guess afterwards.
    _FINGER_EFFORT_LIMIT = 8.5  # N.m, per green_challenge's actuator config
    _upper_term = base_env.action_manager.get_term("upper_body_ik")
    _finger_ids = list(_upper_term._hand_joint_ids)
    _finger_names = list(_upper_term._hand_joint_names)
    _finger_gains: list[tuple[float, float]] = []
    _xr_actuator = robot.actuators.get("xr_upper")
    for _name in _finger_names:
        if _xr_actuator is not None and _name in _xr_actuator.joint_names:
            _li = list(_xr_actuator.joint_names).index(_name)
            _finger_gains.append(
                (
                    float(_xr_actuator.stiffness[0, _li].item()),
                    float(_xr_actuator.damping[0, _li].item()),
                )
            )
        else:
            _finger_gains.append((0.0, 0.0))

    # --- thumb oscillation probe (--thumb_probe) ---------------------------
    # Same metric as scripts/thumb_probe_socket.py reads off green_challenge's
    # run.sh socket: per-joint RMS velocity + peak-to-peak position over ~1 sim
    # second. Includes the passive intermediate/distal dofs -- they are the
    # visibly rattling segments and carry no PD of their own.
    _thumb_ids = [i for i, n in enumerate(robot.joint_names) if "thumb" in n]
    _thumb_names = [robot.joint_names[i].replace("_joint", "") for i in _thumb_ids]
    _thumb_pos_buf: list[torch.Tensor] = []
    _thumb_vel_buf: list[torch.Tensor] = []
    if args_cli.thumb_probe:
        _gain_rows = []
        for _gname, _gact in robot.actuators.items():
            for _li, _jn in enumerate(_gact.joint_names):
                if "thumb" not in _jn:
                    continue
                _arm = (
                    float(_gact.armature[0, _li].item())
                    if getattr(_gact, "armature", None) is not None
                    else float("nan")
                )
                _gain_rows.append(
                    f"{_jn}[{_gname}] kp={float(_gact.stiffness[0, _li]):g} "
                    f"kd={float(_gact.damping[0, _li]):g} armature={_arm:g}"
                )
        print(">>> [thumb-probe] resident params: " + "; ".join(_gain_rows), flush=True)

    def thumb_probe_tick() -> None:
        if not args_cli.thumb_probe:
            return
        _thumb_pos_buf.append(robot.data.joint_pos[0, _thumb_ids].detach().cpu().clone())
        _thumb_vel_buf.append(robot.data.joint_vel[0, _thumb_ids].detach().cpu().clone())
        if len(_thumb_pos_buf) < 150:
            return
        pos = torch.stack(_thumb_pos_buf)
        vel = torch.stack(_thumb_vel_buf)
        _thumb_pos_buf.clear()
        _thumb_vel_buf.clear()
        stats = "  ".join(
            f"{name}: rms_v={vel[:, k].pow(2).mean().sqrt():.4f} "
            f"p2p={(pos[:, k].max() - pos[:, k].min()):.4f} "
            f"mean={pos[:, k].mean():.3f}"
            for k, name in enumerate(_thumb_names)
        )
        print(f"[thumb-probe] {stats}", flush=True)

    def grip_report() -> str:
        """Per-hand peak *closing* torque, and how many digits are saturated.

        Only positive error (commanded closed of achieved) counts. All finger
        joints run 0/open -> positive/closed, so a negative error means the hand
        is being driven open, which is not grip. That distinction matters here:
        the open-hand command sits at the joint's lower limit while the thumb
        physically rests above it, so an idle hand shows a permanent -8.5 Nm
        opening torque that would otherwise mask every real grasp.
        """
        target = robot.data.joint_pos_target[0, _finger_ids]
        actual = robot.data.joint_pos[0, _finger_ids]
        vel = robot.data.joint_vel[0, _finger_ids]
        lines = []
        for side in ("left", "right"):
            peak_tau = 0.0
            peak_name = ""
            peak_err = 0.0
            saturated = 0
            count = 0
            for idx, name in enumerate(_finger_names):
                if not name.startswith(side):
                    continue
                count += 1
                k_p, k_d = _finger_gains[idx]
                err = float(target[idx].item() - actual[idx].item())
                if err <= 0.0:  # opening, not gripping
                    continue
                tau = min(_FINGER_EFFORT_LIMIT, k_p * err - k_d * float(vel[idx].item()))
                if tau >= _FINGER_EFFORT_LIMIT - 1e-3:
                    saturated += 1
                if tau > peak_tau:
                    peak_tau, peak_err, peak_name = tau, err, name.replace(f"{side}_", "")
            lines.append(
                f"{side[0].upper()}: grip {peak_tau:+.2f}Nm on {peak_name or '-'} "
                f"(err {peak_err:+.3f} rad) sat {saturated}/{count}"
            )
        return " | ".join(lines)

    if args_cli.xr:
        rate_limiter = None
    else:
        rate_limiter = RateLimiter(args_cli.step_hz)

    print(f"Using teleop device: {teleop_interface}")
    if use_keyboard:
        print(_HELP_KEYBOARD, flush=True)
    if use_gamepad:
        print(_HELP_GAMEPAD, flush=True)
    if args_cli.idle_pause:
        print(
            f"Idle-pause ON: sim freezes whenever the loco command is zero "
            f"(settle {args_cli.idle_pause_settle:.1f}s; stick/WASD or C resumes).",
            flush=True,
        )
    if args_cli.xr and not args_cli.auto_start:
        print("Press START in XR to begin recording (or relaunch with --auto_start).")
    elif args_cli.auto_start:
        print("Auto-start enabled: recording/WASD active immediately.", flush=True)

    # Anchor rotation written before the headset session is live does not
    # survive session start (position does), so the very first trajectory
    # would face the static-anchor direction instead of the robot's. Re-assert
    # the anchor on every False -> True edge of the live-session signal so the
    # operator always starts looking where the robot looks.
    xr_session_was_live = False

    # Window X / File->Exit posts a Kit quit request; with a live CloudXR
    # session Kit's own shutdown wedges inside xrWaitFrame before the loop's
    # is_running() check ever sees it. Catch the quit request ourselves, end
    # the XR session synchronously (before Kit continues shutting down), and
    # let the loop fall through to the normal save/close teardown.
    quit_requested = [False]
    _quit_sub = None
    try:
        import omni.kit.app

        def _on_quit_request(event) -> None:
            if not quit_requested[0]:
                quit_requested[0] = True
                print("[exit] quit requested — ending XR session", flush=True)
                end_xr_display()

        _quit_sub = (
            omni.kit.app.get_app()
            .get_shutdown_event_stream()
            .create_subscription_to_pop(_on_quit_request, name="sberchel_record_quit")
        )
    except Exception as exc:
        logger.debug("quit-request hook unavailable: %s", exc)

    while simulation_app.is_running() and not quit_requested[0]:
        try:
            with torch.inference_mode():
                poll_controllers()
                if anchor_teleporter is not None:
                    xr_session_live = xr_display_active()
                    if xr_session_live and not xr_session_was_live:
                        anchor_teleporter.teleport_to_robot(robot)
                        if target_hud is not None:
                            target_hud.refresh_soon()
                        print(
                            "[XR] session live — anchor re-asserted on the robot",
                            flush=True,
                        )
                    xr_session_was_live = xr_session_live
                if target_hud is not None:
                    target_hud.tick()
                idle_paused = False
                if (
                    args_cli.idle_pause
                    and running_recording_instance
                    and teleoperation_active
                    and loco_mode == "locomotion"
                ):
                    if any(abs(c) > 1e-9 for c in controller_command()):
                        last_loco_cmd_t = time.time()
                    idle_paused = (
                        last_loco_cmd_t is None
                        or time.time() - last_loco_cmd_t >= args_cli.idle_pause_settle
                    )
                if idle_paused != idle_paused_prev:
                    idle_paused_prev = idle_paused
                    if idle_paused:
                        print(
                            "[idle-pause] loco command zero — sim frozen "
                            "(stick/WASD resumes, C/Create for static teleop)",
                            flush=True,
                        )
                    elif running_recording_instance:
                        # Drop the frozen span from the Hz window so the first
                        # post-resume rate print is not diluted by wall-clock pause.
                        last_hz_t = time.time()
                        hz_window_steps = 0
                        print("[idle-pause] sim resumed", flush=True)
                if running_recording_instance and not idle_paused:
                    step_count += 1
                    hz_window_steps += 1
                    now = time.time()
                    if now - last_hz_t >= 1.0:
                        effective_hz = hz_window_steps / (now - last_hz_t)
                        hz_window_steps = 0
                        last_hz_t = now
                    robot_pos, robot_yaw = robot_root_state(robot)
                    # Latch the left-hand hold from the settled measured pose so
                    # it applies from (nearly) step 0 of every episode, not only
                    # after the first static-mode entry.
                    if (
                        args_cli.disable_left_hand
                        and left_hold_snapshot is None
                        and step_count >= _LEFT_HOLD_SETTLE_STEPS
                    ):
                        left_hold_snapshot = capture_frozen_upper(robot)
                        print(
                            f"[left-hold] latched at step {step_count} from measured pose",
                            flush=True,
                        )
                    if (
                        pink_hold_settle_pending
                        and not args_cli.gc_self_collisions
                        and loco_mode == "locomotion"
                        and step_count >= _LEFT_HOLD_SETTLE_STEPS
                    ):
                        # Re-latch the locomotion hold from the settled measured
                        # pose (see pink_hold_settle_pending above): drops the
                        # in-penetration reset thumb command that chatters both
                        # thumbs against the palm from step 0.
                        #
                        # Skipped under --gc_self_collisions: run.sh parity
                        # means commanding the init pose forever, exactly like
                        # their constant default-pose finger commands. Their
                        # quiet thumb is quiet BECAUSE the command wedges it
                        # into the palm (a static jam a solver constraint
                        # holds); re-latching to the settled pose un-jams it
                        # and frees it to ring.
                        pink_hold.refresh(robot, measured=True)
                        pink_hold_settle_pending = False
                        print(
                            f"[loco-hold] re-latched at step {step_count} from "
                            "settled measured pose (thumb de-penetration)",
                            flush=True,
                        )
                    vx = vy = wz = 0.0
                    upper_body_mode: UpperBodyMode | None = None
                    neck_yaw_cmd, neck_pitch_cmd = 0.0, 0.0
                    # Freeze-by-default: the walk engage block below overwrites
                    # this with the policy's target; static mode ramps it to 0.
                    torso_yaw_cmd = last_torso_target
                    blend_alpha: float | None = None
                    squeeze_cmd = shape_squeeze(controller_squeeze())
                    push_squeeze(squeeze_cmd)

                    if teleoperation_active:
                        if loco_mode == "locomotion":
                            vx, vy, wz = controller_command()
                            if frozen_snapshot is None:
                                frozen_snapshot = capture_frozen_upper(robot, last_teleop_upper)
                                pink_hold.refresh(robot)
                                # (--zest_upper: the frozen arm reference was
                                # already captured in enter_locomotion_mode --
                                # every entry path runs it, C-press included.)
                            # Wrist targets still recorded for replay; Pink solve is skipped.
                            upper_body = apply_left_hold(
                                build_upper_from_frozen_snapshot(
                                    robot, frozen_snapshot, base_env.device
                                )
                            )
                            upper_body_mode = "frozen"
                            last_neck_targets = (0.0, 0.0)
                        else:
                            raw = teleop_interface._get_raw_data()
                            if hold_xy_yaw is None:
                                hold_xy_yaw = capture_hold_pose_from_robot(robot)
                            vx, vy, wz = station_keeper.command(robot, hold_xy_yaw)
                            upper_body = torch.cat(
                                [r.retarget(raw) for r in teleop_interface._retargeters],
                                dim=-1,
                            )
                            upper_body = apply_left_hold(upper_body)
                            if static_blend_from is not None:
                                static_blend_steps_done += 1
                                blend_alpha = smoothstep(
                                    static_blend_steps_done / static_blend_total
                                )
                                from_upper = apply_left_hold(
                                    build_upper_from_frozen_snapshot(
                                        robot, static_blend_from, upper_body.device
                                    )
                                )
                                upper_body = blend_upper_command(
                                    from_upper, upper_body, blend_alpha
                                )
                                if static_blend_steps_done >= static_blend_total:
                                    static_blend_from = None
                            last_teleop_upper = (
                                upper_body.squeeze(0).clone()
                                if upper_body.ndim > 1
                                else upper_body.clone()
                            )
                            upper_body_mode = "teleop"
                            head_pose = raw.get(DeviceBase.TrackingTarget.HEAD)
                            if head_pose is not None:
                                last_neck_targets = headset_to_neck_targets(
                                    head_pose,
                                    robot_yaw,
                                    yaw_limit=NECK_YAW_LIMIT,
                                    pitch_limit=NECK_PITCH_LIMIT,
                                )
                            # Missing HEAD: hold last neck targets from static mode.

                        neck_yaw_cmd, neck_pitch_cmd = last_neck_targets
                        if loco_mode == "static" and not args_cli.zest_upper:
                            # Torso home is the default pose (0 offset); ramp there
                            # from wherever the walk policy left it. Under
                            # --zest_upper the torso executes zest's output in
                            # both modes (assigned after the policy step below).
                            torso_yaw_cmd = 0.0
                            if blend_alpha is not None:
                                torso_yaw_cmd = static_blend_torso_from + blend_alpha * (
                                    0.0 - static_blend_torso_from
                                )
                            last_torso_target = torso_yaw_cmd
                        if blend_alpha is not None:
                            neck_yaw_cmd = static_blend_neck_from[0] + blend_alpha * (
                                neck_yaw_cmd - static_blend_neck_from[0]
                            )
                            neck_pitch_cmd = static_blend_neck_from[1] + blend_alpha * (
                                neck_pitch_cmd - static_blend_neck_from[1]
                            )

                        if upper_body.ndim == 1:
                            upper_body = upper_body.unsqueeze(0)
                        upper_body = upper_body.to(base_env.device).repeat(base_env.num_envs, 1)
                    else:
                        upper_body = idle_upper.unsqueeze(0).repeat(base_env.num_envs, 1)
                        upper_body_mode = "idle"
                        last_neck_targets = (0.0, 0.0)
                        neck_yaw_cmd, neck_pitch_cmd = last_neck_targets

                    zest_ref: list[float] | None = None
                    if args_cli.zest_upper:
                        # Absolute 13-value commander reference: torso_yaw at its
                        # default (the operator has no torso input; zest's
                        # residual owns the counter-rotation), neck from the
                        # headset (zeroed in locomotion). Arms: in static, the
                        # previous tick's raw Pink IK solution (live tracking);
                        # in locomotion, the reference FROZEN at loco entry --
                        # the operator walks away from the robot, so live hand
                        # tracking stops describing an intent. First tick after
                        # a reset has no IK solve yet; measured pose stands in.
                        if loco_mode == "locomotion" and loco_zest_ref_arms is not None:
                            ref_arm = loco_zest_ref_arms
                        else:
                            ref_arm = pink_hold.last_ik_arm_solution()
                            if ref_arm is None:
                                ref_arm = robot.data.joint_pos[:, pink_hold.arm_joint_ids]
                        defaults = robot.data.default_joint_pos[0]
                        zest_ref = [
                            float(defaults[zest_ref_torso_id]),
                            *[float(v) for v in ref_arm[0].tolist()],
                            float(neck_yaw_cmd) + float(defaults[zest_ref_neck_ids[0]]),
                            float(neck_pitch_cmd) + float(defaults[zest_ref_neck_ids[1]]),
                        ]
                    body_targets = policies.step(loco_mode, vx, vy, wz, upper_ref=zest_ref)
                    legs = body_targets[:LEGS_ACTION_DIM]
                    if args_cli.zest_upper:
                        if loco_mode == "static":
                            # run_with_vla parity for manipulation: everything
                            # in the torso block that reaches the robot is
                            # zest's output (ref + residual). Values are
                            # default-subtracted (Network.__call__), same units
                            # as the offset-based action terms.
                            # last_neck_targets deliberately NOT updated here:
                            # under --zest_upper it holds the *headset
                            # reference* (reused when HEAD tracking drops out);
                            # overwriting it with the policy output would feed
                            # zest its own neck back.
                            policy_neck = body_targets[POLICY_MODEL_NECK_SLICE]
                            neck_yaw_cmd = float(policy_neck[0])
                            neck_pitch_cmd = float(policy_neck[1])
                            torso_yaw_cmd = float(body_targets[POLICY_TORSO_MODEL_INDEX])
                            last_torso_target = torso_yaw_cmd
                            arm_raw = torch.as_tensor(
                                body_targets[POLICY_MODEL_ARM_SLICE],
                                dtype=torch.float32,
                                device=base_env.device,
                            ).unsqueeze(0).repeat(base_env.num_envs, 1)
                            pink_hold.set_zest_arm_targets(arm_raw)
                        # locomotion: zest contributes ONLY the legs, exactly
                        # like run_kb/run.py (their loops PD-hold the raw torso
                        # block and discard zest's). Its residual wobbles the
                        # arms with the gait -- the classic hold carries an
                        # object far more steadily -- and at an empty ref its
                        # equilibrium parks the hands behind the hips. The hold
                        # keeps arms/wrists/fingers PD-pinned; neck stays 0 and
                        # torso_yaw holds its last value via the defaults above.
                        stash_zest_upper_ref(base_env, zest_ref)
                    if loco_mode == "locomotion" and loco_is_walk:
                        # run_kb parity: walk's neck targets apply whenever it
                        # drives the legs. Values are default-subtracted, same
                        # units as the neck action term.
                        policy_neck = body_targets[POLICY_MODEL_NECK_SLICE]
                        neck_yaw_cmd = float(policy_neck[0])
                        neck_pitch_cmd = float(policy_neck[1])
                        last_neck_targets = (neck_yaw_cmd, neck_pitch_cmd)
                    if loco_mode == "locomotion" and walk_arm_drive != "never":
                        # when_free hands the arms back the instant the operator
                        # squeezes: swinging the shoulders while carrying a product
                        # drags the hand through space and can break the grasp. The
                        # gate reads the squeeze already recorded as teleop_state[:,1],
                        # so replay reproduces it from the dataset.
                        engage = walk_arm_drive == "always" or squeeze_cmd <= _ARM_DRIVE_SQUEEZE_EPS
                        if engage and not arm_drive_engaged:
                            # Re-seed on every re-engage, not just mode entry: the
                            # arms have been frozen wherever the grasp left them.
                            seed_loco_arm_blend()
                        arm_drive_engaged = engage
                        if engage:
                            arm_raw = torch.as_tensor(
                                body_targets[POLICY_MODEL_ARM_SLICE],
                                dtype=torch.float32,
                                device=base_env.device,
                            ).unsqueeze(0).repeat(base_env.num_envs, 1)
                            arm_alpha = 1.0
                            if loco_arm_blend_from is not None:
                                loco_arm_blend_steps_done += 1
                                arm_alpha = smoothstep(
                                    loco_arm_blend_steps_done / loco_arm_blend_total
                                )
                                if loco_arm_blend_steps_done >= loco_arm_blend_total:
                                    loco_arm_blend_from = None
                            pink_hold.set_arm_targets(
                                arm_raw, blend_from=loco_arm_blend_from, alpha=arm_alpha
                            )
                            # torso_yaw shares the arm gate and ramp: swinging the
                            # torso while carrying drags the grasped hand exactly
                            # like an arm swing does.
                            policy_torso = float(body_targets[POLICY_TORSO_MODEL_INDEX])
                            torso_yaw_cmd = loco_torso_blend_from + arm_alpha * (
                                policy_torso - loco_torso_blend_from
                            )
                            last_torso_target = torso_yaw_cmd
                            arms_moved_by_policy = True
                    legs_tensor = torch.as_tensor(legs, dtype=torch.float32, device=base_env.device)
                    legs_tensor = legs_tensor.unsqueeze(0).repeat(base_env.num_envs, 1)
                    neck_tensor = torch.tensor(
                        [[neck_yaw_cmd, neck_pitch_cmd]],
                        dtype=torch.float32,
                        device=base_env.device,
                    ).repeat(base_env.num_envs, 1)
                    torso_tensor = torch.tensor(
                        [[torso_yaw_cmd]],
                        dtype=torch.float32,
                        device=base_env.device,
                    ).repeat(base_env.num_envs, 1)
                    actions = pack_sberchel_xr_action(
                        legs_tensor, upper_body, neck_tensor, torso_tensor
                    )
                    stash_locomotion_command(base_env, vx, vy, wz)
                    stash_teleop_state(
                        base_env,
                        loco_mode=loco_mode,
                        squeeze=squeeze_cmd,
                        legacy_gains=args_cli.legacy_teleop_gains,
                    )
                    env.step(actions)
                    # From here on joint_pos_target holds a real command, so the
                    # Pink hold can capture grasp intent instead of measured pose.
                    pink_hold.mark_stepped()
                    thumb_probe_tick()

                    should_print = False
                    if args_cli.print_every_n > 0:
                        if teleoperation_active and not first_active_logged:
                            should_print = True
                            first_active_logged = True
                        elif step_count % args_cli.print_every_n == 0:
                            should_print = True
                    if should_print:
                        hold_dbg = (
                            f" hold=({hold_xy_yaw[0]:.2f},{hold_xy_yaw[1]:.2f},{hold_xy_yaw[2]:.2f})"
                            if hold_xy_yaw is not None
                            else ""
                        )
                        print(
                            f"[step {step_count}] mode={loco_mode}"
                            f"{'/' + policies.active_type if policies.multi_policy else ''} "
                            f"upper={upper_body_mode} "
                            f"pos=({robot_pos[0]:.2f},{robot_pos[1]:.2f}) yaw={robot_yaw:.2f} "
                            f"cmd=({vx:.2f},{vy:.2f},{wz:.2f}) keys={controller_pressed()} "
                            f"hz={effective_hz:.1f} "
                            f"neck=({neck_yaw_cmd:.2f},{neck_pitch_cmd:.2f}) "
                            f"torso={torso_yaw_cmd:.2f}{hold_dbg}",
                            flush=True,
                        )
                        if effective_hz > 0.0 and effective_hz < 60.0 and loco_mode == "locomotion":
                            print(
                                f"[warn] loco wall-clock {effective_hz:.1f} Hz "
                                f"(want ~150). Try --loco_render_interval 10",
                                flush=True,
                            )
                        elif effective_hz > 0.0 and effective_hz < 60.0 and loco_mode == "static":
                            print(
                                f"[warn] static wall-clock {effective_hz:.1f} Hz "
                                f"(Pink IK + XR); expected slower than loco. "
                                f"--static_render_interval 2 trades hand-pose "
                                f"freshness for rate.",
                                flush=True,
                            )
                    if (
                        args_cli.grip_report_every_n > 0
                        and step_count % args_cli.grip_report_every_n == 0
                    ):
                        print(
                            f"[grip {step_count}] squeeze={squeeze_cmd:.2f} {grip_report()}",
                            flush=True,
                        )
                else:
                    base_env.sim.render()

                if should_reset or should_discard_reset or should_next_reset:
                    if should_reset and not (should_discard_reset or should_next_reset):
                        episode = base_env.recorder_manager.get_episode(0)
                        if len(episode.data.get("actions", [])) == 0:
                            # A double SAVE press right after a reset would export a
                            # num_samples=0 stub and advance the --num_demos counter.
                            print(
                                "[save] no steps recorded since the last reset — nothing to "
                                "save; retrying the same start pose",
                                flush=True,
                            )
                            should_reset = False
                            should_discard_reset = True
                        else:
                            # The operator only saves good episodes, so the SAVE press is
                            # the ground-truth success signal. Stash it as an extra HDF5
                            # attr rather than episode.success: record_pre_reset overwrites
                            # .success from the (absent) success termination term right
                            # before export, while stashed attrs are written after the
                            # stock writer and win.
                            #
                            # Mode-switch moments (C / Create presses) are read back from
                            # the recorded buffer, not live bookkeeping: a press flips the
                            # mode of the NEXT recorded step, so the presses that survived
                            # checkpoint rollbacks (which truncate the buffer) are exactly
                            # the flips of teleop_state[:,0]. Each index is the first step
                            # of its segment: [0, s0), [s0, s1), ... splits the demo into
                            # walk / manipulate subtasks.
                            is_loco = [
                                bool(float(row[0, 0]) > 0.5)
                                for row in episode.data.get("teleop_state", [])
                            ]
                            switch_steps = [
                                i
                                for i in range(1, len(is_loco))
                                if is_loco[i] != is_loco[i - 1]
                            ]
                            stash_episode_extra_attrs(
                                episode,
                                success=True,
                                mode_switch_steps=np.asarray(switch_steps, dtype=np.int64),
                                mode_switch_to=np.asarray(
                                    [int(is_loco[i]) for i in switch_steps],
                                    dtype=np.int64,
                                ),
                            )
                            switch_desc = ", ".join(
                                f"{i}->{'loco' if is_loco[i] else 'static'}"
                                for i in switch_steps
                            )
                            print(
                                f"[save] mode switches at steps [{switch_desc}] — "
                                f"{len(switch_steps) + 1} segment(s) over "
                                f"{len(is_loco)} steps",
                                flush=True,
                            )
                    discard = should_discard_reset or should_next_reset
                    retry_same_pose = should_discard_reset
                    if discard:
                        base_env.recorder_manager.cfg.export_in_record_pre_reset = False
                        discard_viewport_video_frames(env)
                    if not retry_same_pose:
                        pose_index += 1
                        episode_seed = derive_episode_seed(args_cli.seed, pose_index)
                    # Reseed on every reset: a retry re-derives the SAME spawn and
                    # jitter sample (the explicit pose restore below stays as
                    # belt-and-braces), a new pose gets its own recorded seed.
                    env.reset(seed=episode_seed)
                    if retry_same_pose:
                        # Overwrite the freshly sampled random spawn with the pose the
                        # discarded attempt started from, so it can be retried.
                        robot.write_root_pose_to_sim(episode_start_root_pose)
                        robot.write_root_velocity_to_sim(
                            torch.zeros_like(robot.data.root_state_w[:, 7:13])
                        )
                        if episode_start_products is not None:
                            # Put first-row products back to the discarded attempt's
                            # (jittered) placement. Tolerances sit above contact
                            # micro-jitter but below the jitter magnitudes, so
                            # re-rolled items get restored without waking the
                            # whole store.
                            base_env.scene.restore_store_rigid_body_state(
                                base_env.sim,
                                episode_start_products,
                                pos_tol=0.004,
                                ang_tol=5e-4,
                            )
                        # env.reset() already recorded initial_state with the discarded
                        # sample; recapture it so the episode replays from the true spawn.
                        # Pass explicit env ids: with None the stock InitialStateRecorder
                        # indexes value[None], which prepends a dim instead of selecting
                        # envs and breaks scene.reset_to on replay.
                        base_env.recorder_manager.reset()
                        base_env.recorder_manager.record_post_reset(
                            torch.arange(base_env.num_envs, dtype=torch.int64, device=base_env.device)
                        )
                    stamp_episode_seed()
                    roll_target_hud()
                    episode_start_root_pose = robot.data.root_state_w[:, :7].clone()
                    episode_start_products = (
                        base_env.scene.capture_store_rigid_body_state(base_env.sim)
                        if hasattr(base_env.scene, "capture_store_rigid_body_state")
                        else None
                    )
                    # Targets survive reset while the robot is teleported back to
                    # its init pose, so they no longer describe the robot.
                    pink_hold.invalidate()
                    # New episode, new in-penetration reset pose: re-arm the
                    # settle re-latch of the locomotion hold.
                    pink_hold_settle_pending = True
                    if discard:
                        base_env.recorder_manager.cfg.export_in_record_pre_reset = True
                    hold_xy_yaw = None
                    sim_checkpoint = None  # belongs to the finished episode
                    # Re-latch after the reset pose settles; the stale hold's
                    # pelvis-relative wrist belongs to the previous episode.
                    left_hold_snapshot = None
                    enter_locomotion_mode(capture_frozen=False)
                    last_neck_targets = (0.0, 0.0)
                    # Joints are back at defaults; the previous episode's torso
                    # command no longer describes anything.
                    last_torso_target = 0.0
                    (
                        frozen_snapshot,
                        last_teleop_upper,
                        idle_upper,
                    ) = reset_teleop_state(
                        env=base_env,
                        policies=policies,
                        robot=robot,
                        teleop_interface=teleop_interface,
                        loco_mode="locomotion",
                    )
                    pink_hold.refresh(robot)
                    if anchor_teleporter is not None:
                        # Robot is back at spawn; restore the start-of-episode XR view.
                        anchor_teleporter.teleport_to_robot(robot)
                    should_reset = False
                    should_discard_reset = False
                    should_next_reset = False
                    first_active_logged = False
                    step_count = 0
                    demo_count = exported_demo_count(base_env)
                    _writer = getattr(base_env.recorder_manager, "_dataset_file_handler", None)
                    if getattr(_writer, "failed_writes", 0):
                        print(
                            f"[WARNING] {_writer.failed_writes} episode write(s) FAILED in the "
                            f"background ({_writer.last_write_error}); the demo count below "
                            "overstates what is on disk. Check free space — saved demos are "
                            "being LOST until this is fixed.",
                            flush=True,
                        )
                    if discard:
                        pose_note = (
                            "same start pose retried" if retry_same_pose else "new start pose"
                        )
                        print(
                            f"Discard reset complete (episode not saved, {pose_note}). "
                            f"Recorded {demo_count} episode(s). "
                            f"Start-pose seed {episode_seed} (pose {pose_index}).",
                            flush=True,
                        )
                    else:
                        print(
                            f"Environment reset complete. Recorded {demo_count} episode(s). "
                            f"Start-pose seed {episode_seed} (pose {pose_index}).",
                            flush=True,
                        )

                if args_cli.num_demos > 0 and exported_demo_count(base_env) >= args_cli.num_demos:
                    print(f"Recorded {exported_demo_count(base_env)} demonstrations. Exiting.")
                    break

                if rate_limiter is not None and running_recording_instance:
                    rate_limiter.sleep(base_env)
        except KeyboardInterrupt:
            print("[exit] Ctrl+C — closing cleanly", flush=True)
            break
        except Exception as exc:
            logger.error("Error during simulation step: %s", exc)
            break

    _quit_sub = None
    # End the headset session before tearing anything down; closing the env or
    # app with a live CloudXR session hangs in xrWaitFrame. Skip the update
    # pump when Kit is already shutting down from a window-X quit.
    end_xr_display(simulation_app if not quit_requested[0] else None)
    for controller in controllers:
        controller.close()
    # Drain the async writer and reconcile the export counter against the file:
    # a failed background write loses a saved demo while the counter advances.
    _writer = getattr(base_env.recorder_manager, "_dataset_file_handler", None)
    if hasattr(_writer, "wait_idle"):
        _writer.wait_idle()
        _counted = exported_demo_count(base_env)
        _on_disk = _writer.demo_count
        if getattr(_writer, "failed_writes", 0) or _on_disk != _counted:
            print(
                f"[WARNING] session reconcile: {_on_disk} demo(s) on disk vs "
                f"{_counted} exported, {getattr(_writer, 'failed_writes', 0)} failed "
                f"write(s) ({getattr(_writer, 'last_write_error', None)}) — some saved "
                "demos are MISSING from the file!",
                flush=True,
            )
        else:
            print(f">>> session reconcile OK: {_on_disk} demo(s) on disk", flush=True)
    env.close()
    print(f"Recording session complete. Demonstrations saved to: {args_cli.dataset_file}")
    if args_cli.video:
        video_folder = Path(args_cli.video_dir) if args_cli.video_dir else Path(output_dir) / "videos"
        print(f"Viewport videos saved under: {video_folder}")

if __name__ == "__main__":
    main()
    simulation_app.close()
