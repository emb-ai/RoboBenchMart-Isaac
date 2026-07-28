"""Zest locomotion policy glue for open-kinematics Sberchel."""

from __future__ import annotations

import os
from typing import List

import numpy as np
import torch
import yaml
from green_challenge.configs.chel_2_0_rl import fingers as sim_fingers
from green_challenge.configs.chel_2_0_rl import model as sim_model
from green_challenge.policy import network as network_mod
from green_challenge.policy.network import (
    Network,
    _CONTROLLER_CLS_BY_TYPE,
    _POLICY_NAME_BY_TYPE,
    _load_policy_model,
)
from green_challenge.policy.rl_controller.rl_controller import RLControllerZestAP3Open

from dsynth_isaac.robots.sberchel_cfg import _POLICY_TO_OPEN_JOINT

LEGS_ACTION_DIM = 12

OPEN_LEG_JOINT_NAMES: List[str] = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]


def resolve_open_kinematics_joint_name(name: str, robot_joint_names: List[str]) -> str:
    """Map policy/crank joint names to open MJCF articulation names."""
    if name in robot_joint_names:
        return name
    mapped = _POLICY_TO_OPEN_JOINT.get(name)
    if mapped is not None and mapped in robot_joint_names:
        return mapped
    raise KeyError(f"Joint {name!r} not found on open-kinematics robot")


def _remap_pd_gain_keys(kp_by_name: dict, kd_by_name: dict) -> tuple[dict, dict]:
    """Remap crank-named policy gains onto open joint actuator names."""
    u = RLControllerZestAP3Open._ANKLE_U_RATIO
    b = RLControllerZestAP3Open._ANKLE_B_RATIO
    ankle_gain_scale = {
        f"{side}_ankle_u_crank_joint": 2.0 * u * u
        for side in ("left", "right")
    }
    ankle_gain_scale.update(
        {f"{side}_ankle_b_crank_joint": 2.0 * b * b for side in ("left", "right")}
    )
    kp_scaled = {n: v * ankle_gain_scale.get(n, 1.0) for n, v in kp_by_name.items()}
    kd_scaled = {n: v * ankle_gain_scale.get(n, 1.0) for n, v in kd_by_name.items()}

    kp_open: dict = {}
    kd_open: dict = {}
    for policy_name, kp in kp_scaled.items():
        open_name = _POLICY_TO_OPEN_JOINT.get(policy_name, policy_name)
        kp_open[open_name] = kp
        kd_open[open_name] = kd_scaled[policy_name]
    return kp_open, kd_open


def _restore_open_pd_gains(net: Network, env) -> None:
    """Apply policy PD gains to open-named locomotion actuators."""
    kp_by_name, kd_by_name = _remap_pd_gain_keys(
        net.cfg["control"]["stiffness"],
        net.cfg["control"]["damping"],
    )
    robot = env.unwrapped.scene["robot"]
    overridden: list[str] = []
    for actuator in robot.actuators.values():
        for local_idx, j_name in enumerate(actuator.joint_names):
            if j_name not in kp_by_name:
                continue
            actuator.stiffness[:, local_idx] = float(kp_by_name[j_name])
            actuator.damping[:, local_idx] = float(kd_by_name[j_name])
            overridden.append(j_name)

    policy_joint_order = net.policy_joint_order
    missing = [
        n for n in policy_joint_order if _POLICY_TO_OPEN_JOINT.get(n, n) not in overridden
    ]
    if missing:
        print(
            "[SberchelOpenLocomotionNetwork] WARNING: could not restore PD gains for: "
            f"{missing}"
        )


class SberchelOpenLocomotionNetwork(Network):
    """Zest policy with open-kinematics joint-name aliasing for sim IO and PD restore."""

    def __init__(
        self,
        env,
        type: str = "zest",
        device: str = "cpu",
        debug: bool = False,
    ) -> None:
        if type not in _POLICY_NAME_BY_TYPE:
            raise ValueError(
                f"Invalid network type: {type!r}. "
                f"Expected one of {list(_POLICY_NAME_BY_TYPE)}."
            )

        self.type = type
        self.env = env
        self.device = torch.device(device)
        self._debug = debug
        self._closed_kinematics = False
        self._first_call = True
        self._dump_target_once = True
        self._call_count = 0
        self._verbose_steps = 5

        policy_name = _POLICY_NAME_BY_TYPE[type]
        policy_dir = os.path.join(network_mod.POLICY_DIR, "weights", policy_name)
        config_path = os.path.join(policy_dir, "config.yaml")

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg

        policy_model = _load_policy_model(policy_dir, self.device)
        controller_cls = _CONTROLLER_CLS_BY_TYPE[type]
        self.controller = controller_cls(
            cfg, policy_model, self.device, closed_kinematics=False
        )

        self.policy_joint_order = list(cfg["joint_order"])
        self.model_joint_order = list(sim_model.joint_order)
        self._validate_joint_orders()

        self.model_to_policy_idx = [
            self.model_joint_order.index(name) for name in self.policy_joint_order
        ]
        self.policy_to_model_idx = [
            self.policy_joint_order.index(name) for name in self.model_joint_order
        ]

        robot = env.unwrapped.scene["robot"]
        robot_joint_names: List[str] = list(robot.data.joint_names)
        self._sim_indices = [
            robot_joint_names.index(resolve_open_kinematics_joint_name(name, robot_joint_names))
            for name in self.model_joint_order
        ]

        default_joint_pos_all = robot.data.default_joint_pos[0].detach().cpu().numpy()
        self._default_joint_pos_model_order = np.asarray(
            default_joint_pos_all[self._sim_indices], dtype=np.float32
        )

        cfg_defaults = cfg["init_state"]["default_joint_angles"]
        self._policy_default_pos = np.asarray(
            [cfg_defaults[name] for name in self.policy_joint_order], dtype=np.float32
        )

        self._n_fingers = len(sim_fingers.joint_order)
        _restore_open_pd_gains(self, env)
        self.reset()


def build_sberchel_locomotion_network(
    env,
    type: str = "zest",
    device: str = "cpu",
    debug: bool = False,
) -> SberchelOpenLocomotionNetwork:
    """Construct a zest policy wired for open-kinematics Sberchel (``closed_kinematics=False``)."""
    net = SberchelOpenLocomotionNetwork(env, type=type, device=device, debug=debug)
    net.reset()
    return net


def policy_leg_targets(net: Network, vx: float, vy: float, wz: float) -> list[float]:
    """Run locomotion policy; return 12 leg targets in ``OPEN_LEG_JOINT_NAMES`` order."""
    net.set_command_detailed(vx, vy, wz)
    net.set_joint_overrides({
        "torso_joint_pos": [0.0] * 13,
        "wrist_joint_pos": [0.0] * 4,
        "finger_joint_pos": [0.0] * 12,
    })
    loco = net(net.obs())
    return list(loco["joint_pos"][:LEGS_ACTION_DIM])
