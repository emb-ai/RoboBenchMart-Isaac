"""Sberchel robot camera configs mirrored from green_challenge.

Params match ``camera_parameters.json`` presets for robot ``v3_0a`` and the
``create_camera_cfgs`` factory in green_challenge (RGB only, no depth):

- ``left_head``  -> preset ``head_left_rgb_fisheye_v1``
- ``right_head`` -> preset ``head_right_rgb_fisheye_v1``
- ``left_wrist`` -> preset ``wrist_left_rgb_fisheye_v1``
- ``right_wrist``-> preset ``wrist_right_rgb_fisheye_v1``
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg

_HORIZONTAL_APERTURE = 5.784
_UPDATE_PERIOD = 0.067
# MJCF head camera bodies are X-forward / Y-up (+90 deg about X from the Z-up
# neck frame). green_challenge's (0.5, 0.5, -0.5, -0.5) OpenGL offset assumes a
# Z-up parent and rolls the image 90 deg here. Map OpenGL (-Z forward, +Y up)
# onto that body frame with a -90 deg rotation about Y.
_OFFSET_ROT = (0.70710678, 0.0, -0.70710678, 0.0)
# The wrist camera bodies are rolled 90 deg about the optical axis relative to
# the head bodies (verified visually on replay renders): compose the head
# offset with Rz(+90 deg) in the camera frame, rotating the image clockwise.
_WRIST_OFFSET_ROT = (0.5, -0.5, -0.5, 0.5)

# green_challenge camera_parameters.json (robot_v3_0a) presets.
_CAMERA_PARAMS: dict[str, dict[str, Any]] = {
    "left_head": {
        "body": "left_head_camera",
        "projection_type": "fisheyePolynomial",
        "fx": 60.0,
        "fy": 60.0,
        "cx": 430.0,
        "cy": 300.0,
        "distortion": (0.0, 0.0038, 0.0, 0.0),
        "fov": 200.0,
        "clipping_range": (0.15, 1_000_000.0),
        "offset_pos": (0.0, 0.0, 0.0155),
    },
    "right_head": {
        "body": "right_head_camera",
        "projection_type": "fisheyeKannalaBrandtK3",
        "fx": 307.6776698581577,
        "fy": 307.7025650237449,
        "cx": 477.6228792583235,
        "cy": 319.9706807837122,
        "distortion": (
            -0.008605678403163834,
            0.006550329526540998,
            -0.010705254411063993,
            0.003190446522626963,
        ),
        "fov": 180.0,
        "clipping_range": (0.08, 1_000_000.0),
        "offset_pos": (0.0, 0.0, 0.0155),
    },
    # Wrists: green_challenge specifies KannalaBrandtK3 with zero distortion,
    # i.e. the pure equidistant model r = fx*theta. Isaac's KB3 path renders
    # that far too narrow, so use the equivalent f-theta polynomial
    # theta(r) = r/fx (B = 1/fx) on the fisheyePolynomial path, which the
    # left head camera already validated visually.
    "left_wrist": {
        "body": "left_wrist_camera",
        "projection_type": "fisheyePolynomial",
        "fx": 270.0,
        "fy": 270.0,
        "cx": 480.0,
        "cy": 300.0,
        "distortion": (0.0, 1.0 / 270.0, 0.0, 0.0),
        "fov": 180.0,
        "clipping_range": (0.025, 1_000_000.0),
        "offset_pos": (0.04, 0.0, 0.0155),
        "offset_rot": _WRIST_OFFSET_ROT,
    },
    "right_wrist": {
        "body": "right_wrist_camera",
        "projection_type": "fisheyePolynomial",
        "fx": 270.0,
        "fy": 270.0,
        "cx": 480.0,
        "cy": 300.0,
        "distortion": (0.0, 1.0 / 270.0, 0.0, 0.0),
        "fov": 180.0,
        "clipping_range": (0.05, 1_000_000.0),
        "offset_pos": (0.04, 0.0, 0.0155),
        "offset_rot": _WRIST_OFFSET_ROT,
    },
}

_WIDTH = 960
_HEIGHT = 600

SBERCHEL_CAMERA_NAMES: tuple[str, ...] = tuple(_CAMERA_PARAMS.keys())

# camera name -> (scene attr, observations.rgb_camera term name). ``head_img``
# stays the legacy left-head key so existing datasets and hdf5_to_mp4 calls
# keep working.
SBERCHEL_CAMERA_OBS_KEYS: dict[str, tuple[str, str]] = {
    "left_head": ("left_head_camera", "head_img"),
    "right_head": ("right_head_camera", "right_head_img"),
    "left_wrist": ("left_wrist_camera", "left_wrist_img"),
    "right_wrist": ("right_wrist_camera", "right_wrist_img"),
}


# LeRobot export trio (sample datasets: cam_head / cam_left_wrist / cam_right_wrist).
SBERCHEL_LEROBOT_CAMERAS: tuple[str, ...] = ("left_head", "left_wrist", "right_wrist")

# green_challenge per-camera masks (manip_env.py CAMERA_MASKS), applied at
# native 960x600 before resize: fisheye valid-region oval + mount cutouts.
_CAMERA_MASK_FILES: dict[str, str] = {
    "left_head": "mask_head.npy",
    "right_head": "mask_head.npy",
    "left_wrist": "mask_left_arm.npy",
    "right_wrist": "mask_right_arm.npy",
}


def load_sberchel_camera_masks(cameras: str | Iterable[str]) -> dict[str, np.ndarray]:
    """Load green_challenge's native-resolution camera masks, keyed by camera name.

    The green_challenge checkout is resolved from ``GREEN_CHALLENGE_PATH`` or
    the in-repo ``green_challenge`` folder.
    """
    gc_root = Path(
        os.environ.get(
            "GREEN_CHALLENGE_PATH",
            Path(__file__).resolve().parents[2] / "green_challenge",
        )
    )
    mask_dir = gc_root / "assets" / "robot_v3_0a" / "dependencies" / "settings" / "camera_mask"
    masks: dict[str, np.ndarray] = {}
    for camera in parse_camera_names(cameras):
        mask_path = mask_dir / _CAMERA_MASK_FILES[camera]
        if not mask_path.exists():
            raise FileNotFoundError(
                f"Camera mask not found: {mask_path} (set GREEN_CHALLENGE_PATH to the "
                "green_challenge checkout or use --camera_pipeline center_crop)."
            )
        masks[camera] = np.load(mask_path)
    return masks


def parse_camera_size(spec: str) -> tuple[int, int] | None:
    """Parse a ``--camera_size`` value: ``448``, ``448x448``, or ``native``."""
    spec = spec.strip().lower()
    if spec in ("", "native"):
        return None
    parts = spec.split("x")
    if len(parts) == 1:
        return int(parts[0]), int(parts[0])
    if len(parts) == 2:
        return int(parts[0]), int(parts[1])
    raise ValueError(f"Invalid camera size {spec!r}; expected 'native', '448' or '448x448'.")


def parse_camera_names(spec: str | Iterable[str]) -> tuple[str, ...]:
    """Normalize a ``--cameras`` CLI value into validated camera names.

    Accepts an iterable of names or a comma-separated string; ``"all"`` selects
    every camera and ``"none"`` / ``""`` selects nothing.
    """
    if isinstance(spec, str):
        spec = [item.strip() for item in spec.split(",") if item.strip()]
    names = list(spec)
    if names == ["all"]:
        return SBERCHEL_CAMERA_NAMES
    if names in ([], ["none"]):
        return ()
    unknown = [name for name in names if name not in _CAMERA_PARAMS]
    if unknown:
        raise ValueError(
            f"Unknown camera name(s) {unknown}; valid names: {list(SBERCHEL_CAMERA_NAMES)} (or 'all'/'none')."
        )
    return tuple(dict.fromkeys(names))


def _center_crop_resize_intrinsics(
    params: dict[str, Any], new_width: int, new_height: int
) -> tuple[float, float, float, float]:
    """green_challenge ``rescale_camera_preset(mode="center_crop_resize")``.

    Crop the native 960x600 frame to its centered 600x600 square, then scale to
    the new resolution, adjusting fx/fy/cx/cy accordingly.
    """
    crop_size = min(_WIDTH, _HEIGHT)
    crop_x = (_WIDTH - crop_size) / 2.0
    crop_y = (_HEIGHT - crop_size) / 2.0
    sx = new_width / crop_size
    sy = new_height / crop_size
    fx = params["fx"] * sx
    fy = params["fy"] * sy
    cx = (params["cx"] - crop_x) * sx
    cy = (params["cy"] - crop_y) * sy
    return fx, fy, cx, cy


def make_sberchel_camera_cfg(camera: str, size: tuple[int, int] | None = None) -> CameraCfg:
    """Fisheye RGB camera on the MJCF ``*_camera`` body for ``camera``.

    ``size=(width, height)`` renders at that resolution with center-crop-resize
    rescaled intrinsics (e.g. ``(448, 448)`` for the LeRobot sample layout);
    ``None`` keeps the native 960x600.
    """
    params = _CAMERA_PARAMS[camera]
    if size is None:
        width, height = _WIDTH, _HEIGHT
        fx: float = params["fx"]
        fy: float = params["fy"]
        cx: float = params["cx"]
        cy: float = params["cy"]
    else:
        width, height = size
        fx, fy, cx, cy = _center_crop_resize_intrinsics(params, width, height)
    focal_length = (fx * _HORIZONTAL_APERTURE) / width
    intrinsic_matrix = [
        fx, 0.0, cx,
        0.0, fy, cy,
        0.0, 0.0, 1.0,
    ]
    pinhole_cfg = sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
        intrinsic_matrix=intrinsic_matrix,
        width=width,
        height=height,
        focal_length=focal_length,
        projection_type="pinhole",
    )
    k1, k2, k3, k4 = params["distortion"]
    fisheye_cfg = sim_utils.FisheyeCameraCfg(
        projection_type=params["projection_type"],
        fisheye_nominal_width=float(width),
        fisheye_nominal_height=float(height),
        fisheye_optical_centre_x=cx,
        fisheye_optical_centre_y=cy,
        fisheye_polynomial_a=k1,
        fisheye_polynomial_b=k2,
        fisheye_polynomial_c=k3,
        fisheye_polynomial_d=k4,
        fisheye_polynomial_e=0.0,
        fisheye_polynomial_f=0.0,
        fisheye_max_fov=params["fov"],
        clipping_range=params["clipping_range"],
        focal_length=pinhole_cfg.focal_length,
        focus_distance=pinhole_cfg.focus_distance,
        f_stop=pinhole_cfg.f_stop,
        horizontal_aperture=pinhole_cfg.horizontal_aperture,
        vertical_aperture=pinhole_cfg.vertical_aperture,
        horizontal_aperture_offset=pinhole_cfg.horizontal_aperture_offset,
        vertical_aperture_offset=pinhole_cfg.vertical_aperture_offset,
        lock_camera=pinhole_cfg.lock_camera,
    )
    body = params["body"]
    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/root/{body}/{body}_sensor",
        update_period=_UPDATE_PERIOD,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=fisheye_cfg,
        offset=CameraCfg.OffsetCfg(
            pos=params["offset_pos"],
            rot=params.get("offset_rot", _OFFSET_ROT),
            convention="opengl",
        ),
    )


def make_sberchel_left_head_camera_cfg() -> CameraCfg:
    """Left head fisheye RGB camera on the MJCF ``left_head_camera`` body."""
    return make_sberchel_camera_cfg("left_head")


def apply_sberchel_camera_size(
    env_cfg: Any, size: tuple[int, int] | None, cameras: Iterable[str] | None = None
) -> Any:
    """Rebuild existing scene camera cfgs at ``size`` via center-crop-resize.

    Only cameras already present on the scene are touched; ``cameras`` limits
    the rebuild further. ``size=None`` rebuilds at native 960x600.
    """
    selected = set(SBERCHEL_CAMERA_NAMES if cameras is None else parse_camera_names(cameras))
    for camera in selected:
        scene_attr, _ = SBERCHEL_CAMERA_OBS_KEYS[camera]
        if getattr(env_cfg.scene, scene_attr, None) is not None:
            setattr(env_cfg.scene, scene_attr, make_sberchel_camera_cfg(camera, size=size))
    return env_cfg


def add_sberchel_cameras(
    env_cfg: Any,
    cameras: Iterable[str] = SBERCHEL_CAMERA_NAMES,
    size: tuple[int, int] | None = None,
) -> Any:
    """Attach camera sensors and raw-RGB ObsTerms under ``observations.rgb_camera``.

    Raw RGB (``normalize=False``) because the default ``normalize=True``
    mean-centers floats, which breaks HDF5 demo video export.
    """
    for camera in parse_camera_names(cameras):
        scene_attr, obs_name = SBERCHEL_CAMERA_OBS_KEYS[camera]
        setattr(env_cfg.scene, scene_attr, make_sberchel_camera_cfg(camera, size=size))
        setattr(
            env_cfg.observations.rgb_camera,
            obs_name,
            ObsTerm(
                func=mdp.image,
                params={"sensor_cfg": SceneEntityCfg(scene_attr), "normalize": False},
            ),
        )
    return env_cfg


def strip_camera_configs(env_cfg: Any, keep: Iterable[str] = ()) -> Any:
    """Remove scene cameras (except ``keep``) and ObsTerms that reference them.

    ``keep`` holds camera names (``left_head``) or scene attrs (``left_head_camera``).

    Isaac Lab's ``remove_camera_configs`` only clears ObsTerms under
    ``observations.policy``. Sberchel puts image terms under
    ``observations.rgb_camera``, so XR teleop/record scripts must use this
    helper (or also pass ``--enable_cameras``) to avoid dangling sensor refs.
    """
    keep_attrs = {name if name.endswith("_camera") else f"{name}_camera" for name in keep}

    cam_names: list[str] = []
    for attr_name in dir(env_cfg.scene):
        if attr_name.startswith("_") or attr_name in keep_attrs:
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
