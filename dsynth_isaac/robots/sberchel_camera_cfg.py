"""Sberchel head camera configs mirrored from green_challenge.

Params match ``camera_parameters.json`` preset ``head_left_rgb_fisheye_v1`` and
the ``create_camera_cfgs`` factory in green_challenge (RGB only, no depth).
"""

from __future__ import annotations

from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg

# green_challenge head_left_rgb_fisheye_v1
_WIDTH = 960
_HEIGHT = 600
_FX = 60.0
_FY = 60.0
_CX = 430.0
_CY = 300.0
_HORIZONTAL_APERTURE = 5.784
_DISTORTION = (0.0, 0.0038, 0.0, 0.0)
_FOV = 200.0
_CLIPPING_RANGE = (0.15, 1_000_000.0)
_UPDATE_PERIOD = 0.067
_OFFSET_POS = (0.0, 0.0, 0.0155)
# MJCF ``left_head_camera`` is already X-forward / Y-up (body quat +90° about X
# from the Z-up neck frame). green_challenge's (0.5, 0.5, -0.5, -0.5) OpenGL
# offset assumes a Z-up parent and rolls the image 90° here. Map OpenGL
# (-Z forward, +Y up) onto that body frame with a -90° rotation about Y.
_OFFSET_ROT = (0.70710678, 0.0, -0.70710678, 0.0)


def make_sberchel_left_head_camera_cfg() -> CameraCfg:
    """Left head fisheye RGB camera on the MJCF ``left_head_camera`` body."""
    focal_length = (_FX * _HORIZONTAL_APERTURE) / _WIDTH
    intrinsic_matrix = [
        _FX, 0.0, _CX,
        0.0, _FY, _CY,
        0.0, 0.0, 1.0,
    ]
    pinhole_cfg = sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
        intrinsic_matrix=intrinsic_matrix,
        width=_WIDTH,
        height=_HEIGHT,
        focal_length=focal_length,
        projection_type="pinhole",
    )
    k1, k2, k3, k4 = _DISTORTION
    fisheye_cfg = sim_utils.FisheyeCameraCfg(
        projection_type="fisheyePolynomial",
        fisheye_nominal_width=float(_WIDTH),
        fisheye_nominal_height=float(_HEIGHT),
        fisheye_optical_centre_x=_CX,
        fisheye_optical_centre_y=_CY,
        fisheye_polynomial_a=k1,
        fisheye_polynomial_b=k2,
        fisheye_polynomial_c=k3,
        fisheye_polynomial_d=k4,
        fisheye_polynomial_e=0.0,
        fisheye_polynomial_f=0.0,
        fisheye_max_fov=_FOV,
        clipping_range=_CLIPPING_RANGE,
        focal_length=pinhole_cfg.focal_length,
        focus_distance=pinhole_cfg.focus_distance,
        f_stop=pinhole_cfg.f_stop,
        horizontal_aperture=pinhole_cfg.horizontal_aperture,
        vertical_aperture=pinhole_cfg.vertical_aperture,
        horizontal_aperture_offset=pinhole_cfg.horizontal_aperture_offset,
        vertical_aperture_offset=pinhole_cfg.vertical_aperture_offset,
        lock_camera=pinhole_cfg.lock_camera,
    )
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/root/left_head_camera/left_head_camera_sensor",
        update_period=_UPDATE_PERIOD,
        height=_HEIGHT,
        width=_WIDTH,
        data_types=["rgb"],
        spawn=fisheye_cfg,
        offset=CameraCfg.OffsetCfg(
            pos=_OFFSET_POS,
            rot=_OFFSET_ROT,
            convention="opengl",
        ),
    )


def strip_camera_configs(env_cfg: Any) -> Any:
    """Remove scene cameras and ObsTerms that reference them.

    Isaac Lab's ``remove_camera_configs`` only clears ObsTerms under
    ``observations.policy``. Sberchel puts image terms under
    ``observations.rgb_camera``, so XR teleop/record scripts must use this
    helper (or also pass ``--enable_cameras``) to avoid dangling sensor refs.
    """
    cam_names: list[str] = []
    for attr_name in dir(env_cfg.scene):
        if attr_name.startswith("_"):
            continue
        attr = getattr(env_cfg.scene, attr_name, None)
        if isinstance(attr, CameraCfg):
            cam_names.append(attr_name)
            setattr(env_cfg.scene, attr_name, None)

    if not cam_names or not hasattr(env_cfg, "observations"):
        return env_cfg

    for group in env_cfg.observations.__dict__.values():
        if group is None or not hasattr(group, "__dict__"):
            continue
        for obs_name in list(group.__dict__.keys()):
            obsterm = getattr(group, obs_name, None)
            params = getattr(obsterm, "params", None)
            if not params:
                continue
            for param_value in params.values():
                if isinstance(param_value, SceneEntityCfg) and param_value.name in cam_names:
                    delattr(group, obs_name)
                    break
    return env_cfg
