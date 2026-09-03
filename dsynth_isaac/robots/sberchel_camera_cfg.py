"""Sberchel robot camera configs, read live from green_challenge.

Camera params are NOT duplicated here. ``_CAMERA_PARAMS`` is parsed at import
time from the checkout's ``camera_parameters.json`` (resolved by
``dsynth_isaac.green_challenge_path``) and assembled exactly the way
green_challenge's own camera factory assembles it, so a preset revision on their
side lands here without an edit. Hardcoding the presets is what let us run the
retired ``_v1`` optics for two weeks after they moved to ``_poly200``.

That factory is now ``scenes/assets/camera_spawner.CameraSpawner`` (the
``green-challenge-sim`` branch; it replaced the free function
``sensors.create_camera_cfgs``, kept there as ``sensors_backup21aug2026.py``).
Two of its behaviours differ from the retired one: ``intrinsic.ftheta_width /
ftheta_height`` now feed ``fisheye_nominal_*`` instead of the render resolution
(mirrored in ``make_sberchel_camera_cfg``), and ``preset["mask_file"]`` is read
into a runtime mask (not mirrored -- see ``_CAMERA_MASK_FILES``, which follows
the masking their evaluation actually runs).

``camera_mapping`` decides which cameras exist. As of the ``_poly200`` revision
that is left head + both wrists; ``right_head_camera`` maps to ``null`` (their
``scenes.DefaultSceneCfg`` has it commented out too), so ``--cameras
right_head`` now fails instead of rendering a camera their stack does not have.

``extrinsic.rotation`` used to be overridden too, with a -90 deg rotation about
Y for the heads and that composed with Rz(+90 deg) for the wrists, on the theory
that green_challenge's ``(0.5, 0.5, -0.5, -0.5)`` assumed a Z-up parent while
the MJCF camera bodies are X-forward / Y-up. Measured against their value that
override rolled the head 90 deg and the wrists 180 deg (upside down) about the
optical axis, which is exactly the framing mismatch seen against the
``left_wrist_camera_sensor`` viewport of a live ``run.sh``. Both stacks spawn
the same MJCF through green_challenge's own ``spawn_from_mjcf``, so the parent
frames are identical and there is nothing to compensate for: the rotation now
comes from the file like everything else. The images the VLA trains on are
rotated instead, at LeRobot conversion time -- see
``sberchel_hdf5_to_lerobot.CAMERA_ROT90_CW``.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

import numpy as np

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg

from dsynth_isaac.green_challenge_path import green_challenge_settings_dir

_HORIZONTAL_APERTURE = 5.784
# Fallback for presets that do not state one. Note that the camera export in
# ``sberchel_replay_demos`` overrides update_period to 0.0 (one fresh frame per
# env step), so this only matters for teleop/record runs.
_UPDATE_PERIOD = 0.067
_WIDTH = 960
_HEIGHT = 600

# green_challenge camera key -> (our camera name, MJCF body). The body name and
# the camera key coincide; keep both so a rename on their side is a one-line
# change here.
_GC_CAMERAS: dict[str, tuple[str, str]] = {
    "left_head_camera": ("left_head", "left_head_camera"),
    "right_head_camera": ("right_head", "right_head_camera"),
    "left_wrist_camera": ("left_wrist", "left_wrist_camera"),
    "right_wrist_camera": ("right_wrist", "right_wrist_camera"),
}


def _load_green_challenge_camera_params() -> dict[str, dict[str, Any]]:
    """Parse ``camera_parameters.json`` the way ``CameraSpawner`` reads it.

    Only cameras that ``camera_mapping`` points at a preset are returned, so a
    camera green_challenge retires (mapped to ``null``) disappears here too.
    """
    path = green_challenge_settings_dir() / "camera_parameters.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"green_challenge camera_parameters.json not found: {path} "
            "(set GREEN_CHALLENGE_PATH to the checkout directory)."
        )
    config = json.loads(path.read_text())
    presets = config["presets"]
    mapping = config["camera_mapping"]

    unknown = set(mapping) - set(_GC_CAMERAS)
    if unknown:
        raise ValueError(
            f"{path} maps camera(s) {sorted(unknown)} that this module has no MJCF "
            "body for; add them to _GC_CAMERAS."
        )

    params: dict[str, dict[str, Any]] = {}
    for gc_name, (name, body) in _GC_CAMERAS.items():
        preset_name = mapping.get(gc_name)
        if preset_name is None:
            continue
        preset = presets[preset_name]
        intrinsic = preset["intrinsic"]
        extrinsic = preset["extrinsic"]

        resolution = tuple(intrinsic["resolution"])
        if resolution != (_WIDTH, _HEIGHT):
            raise ValueError(
                f"{path}: preset {preset_name!r} renders at {resolution}, but the "
                f"mask and center-crop-resize maths here assume {(_WIDTH, _HEIGHT)}."
            )
        # CameraSpawner takes the render raster from the preset's own
        # width/height and the intrinsics from intrinsic.resolution; nothing
        # checks that the two agree, and a preset where they disagree would put
        # the fisheye centre off by the difference.
        raster = (int(preset["width"]), int(preset["height"]))
        if raster != resolution:
            raise ValueError(
                f"{path}: preset {preset_name!r} renders at {raster} but states "
                f"intrinsic.resolution {resolution}; CameraSpawner would use both."
            )
        convention = extrinsic.get("convention", "opengl")
        if convention != "opengl":
            raise ValueError(
                f"{path}: preset {preset_name!r} uses extrinsic convention "
                f"{convention!r}; CameraCfg.OffsetCfg is wired up as OpenGL here."
            )
        if preset.get("data_types", ["rgb"]) != ["rgb"]:
            raise ValueError(
                f"{path}: preset {preset_name!r} asks for {preset['data_types']}; "
                "this module only wires up RGB."
            )
        projection_type = preset.get("projection_type", "fisheyeKannalaBrandtK3")
        if projection_type == "pinhole":
            # CameraSpawner spawns the PinholeCameraCfg for these and skips the
            # fisheye path entirely; make_sberchel_camera_cfg always builds the
            # fisheye cfg, so a pinhole mapping would silently render a lens
            # they do not use. The JSON carries four unmapped pinhole presets.
            raise ValueError(
                f"{path}: camera_mapping points at pinhole preset {preset_name!r}; "
                "make_sberchel_camera_cfg only builds fisheye cameras."
            )

        params[name] = {
            "preset": preset_name,
            "body": body,
            "projection_type": projection_type,
            "fx": float(intrinsic["fx"]),
            "fy": float(intrinsic["fy"]),
            "cx": float(intrinsic["cx"]),
            "cy": float(intrinsic["cy"]),
            "distortion": tuple(float(k) for k in intrinsic["distortion"]),
            "fov": float(intrinsic["fov"]),
            # The ftheta polynomial's reference raster. Absent on the pinhole
            # and _kb3 presets, where CameraSpawner falls back to resolution.
            "ftheta_width": float(intrinsic.get("ftheta_width", resolution[0])),
            "ftheta_height": float(intrinsic.get("ftheta_height", resolution[1])),
            "horizontal_aperture": float(
                intrinsic.get("horizontal_aperture", _HORIZONTAL_APERTURE)
            ),
            "clipping_range": tuple(float(v) for v in preset["clipping_range"]),
            "offset_pos": tuple(float(v) for v in extrinsic["position"]),
            "offset_rot": tuple(float(v) for v in extrinsic["rotation"]),
            "update_period": float(preset.get("update_period", _UPDATE_PERIOD)),
        }

    if not params:
        raise ValueError(f"{path}: camera_mapping enables no cameras.")
    return params


_CAMERA_PARAMS: dict[str, dict[str, Any]] = _load_green_challenge_camera_params()

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

# green_challenge per-camera masks, applied at native 960x600 before resize:
# fisheye valid-region oval + mount cutouts.
#
# Taken from ``manip_env.CAMERA_MASKS`` rather than from the preset's
# ``mask_file`` field, which names the arm masks crossed over: wrist_left ->
# mask_right_arm, wrist_right -> mask_left_arm. ``CameraSpawner._init_masks``
# does now load that field (the retired ``create_camera_cfgs`` ignored it
# entirely), but nothing calls ``CameraSpawner.apply_masks`` -- the one live
# call in ``manip_env.step`` is commented out -- so the masking that actually
# runs in their evaluation is still ``LatestFrameHandler(masks=CAMERA_MASKS)``,
# which pairs each wrist with its own side. Follow the code, not the field.
#
# ``mask_right_arm.npy`` was regenerated on the sim branch and is now the exact
# horizontal mirror of ``mask_left_arm.npy`` (measured: ``L[:, ::-1] == R``,
# valid region 780 px wide centred at x=476 vs the old 738 px at x=451). The
# masks are read live from the checkout, so this needs no edit -- but the
# right-wrist vignette in demos replayed after the switch does not match the
# one in demos replayed before it.
_CAMERA_MASK_FILES: dict[str, str] = {
    "left_head": "mask_head.npy",
    "right_head": "mask_head.npy",
    "left_wrist": "mask_left_arm.npy",
    "right_wrist": "mask_right_arm.npy",
}


def load_sberchel_camera_masks(cameras: str | Iterable[str]) -> dict[str, np.ndarray]:
    """Load green_challenge's native-resolution camera masks, keyed by camera name.

    The green_challenge checkout is resolved by ``dsynth_isaac.green_challenge_path``
    (``GREEN_CHALLENGE_PATH`` or the default checkout location).
    """
    mask_dir = green_challenge_settings_dir() / "camera_mask"
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


def camera_optics_provenance(cameras: str | Iterable[str]) -> dict[str, Any]:
    """What the cameras of a render actually were, for the dataset's ``env_args``.

    Everything here is read live from the green_challenge checkout, so two
    exports made from the same recording months apart can differ in optics with
    nothing in the file to say so. The sim-branch switch of 2026-08-28 is exactly
    that case: the ftheta raster went 960x600 -> 860x600 / 900x510 and
    ``mask_right_arm.npy`` was regenerated, both invisible in every other field.

    The mask digest is over the file bytes, so a regenerated mask of unchanged
    shape still shows up. Masks are recorded for every requested camera even
    under ``--camera_pipeline center_crop``, which does not apply them -- the
    pipeline is recorded next to this and says which happened.
    """
    import hashlib

    mask_dir = green_challenge_settings_dir() / "camera_mask"
    out: dict[str, Any] = {"green_challenge_root": str(green_challenge_settings_dir().parents[3])}
    for camera in parse_camera_names(cameras):
        params = _CAMERA_PARAMS[camera]
        mask_path = mask_dir / _CAMERA_MASK_FILES[camera]
        entry: dict[str, Any] = {
            "preset": params["preset"],
            "projection_type": params["projection_type"],
            "fx": params["fx"], "fy": params["fy"],
            "cx": params["cx"], "cy": params["cy"],
            "distortion": list(params["distortion"]),
            "fov": params["fov"],
            "ftheta_size": [params["ftheta_width"], params["ftheta_height"]],
            "clipping_range": list(params["clipping_range"]),
            "offset_pos": list(params["offset_pos"]),
            "offset_rot": list(params["offset_rot"]),
            "mask_file": _CAMERA_MASK_FILES[camera],
            "mask_sha1": (
                hashlib.sha1(mask_path.read_bytes()).hexdigest()
                if mask_path.exists()
                else None
            ),
        }
        out[camera] = entry
    return out


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
        known = {entry[0] for entry in _GC_CAMERAS.values()}
        retired = [name for name in unknown if name in known]
        hint = (
            f" {retired} exist on the robot but green_challenge's camera_mapping "
            "points them at null, so their evaluation stack does not render them."
            if retired
            else ""
        )
        raise ValueError(
            f"Unknown camera name(s) {unknown}; valid names: "
            f"{list(SBERCHEL_CAMERA_NAMES)} (or 'all'/'none').{hint}"
        )
    return tuple(dict.fromkeys(names))


def _center_crop_resize_intrinsics(
    params: dict[str, Any], new_width: int, new_height: int
) -> tuple[float, float, float, float, float, float]:
    """green_challenge ``rescale_camera_preset(mode="center_crop_resize")``.

    Crop the native 960x600 frame to its centered 600x600 square, then scale to
    the new resolution, adjusting fx/fy/cx/cy accordingly. Returns the scale
    factors too, for the ftheta raster (which their rescale leaves alone --
    ``CameraSpawner.rescale_camera_preset`` only touches the intrinsic matrix,
    so a rescaled ftheta preset comes out with a nominal size that no longer
    matches its own resolution).
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
    return fx, fy, cx, cy, sx, sy


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
        ftheta_width: float = params["ftheta_width"]
        ftheta_height: float = params["ftheta_height"]
    else:
        width, height = size
        fx, fy, cx, cy, sx, sy = _center_crop_resize_intrinsics(params, width, height)
        ftheta_width = params["ftheta_width"] * sx
        ftheta_height = params["ftheta_height"] * sy
    focal_length = (fx * params["horizontal_aperture"]) / width
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
        # Their factory used to pass the render resolution here and ignore the
        # preset's ftheta_* fields; CameraSpawner reads them, so the _poly200
        # presets now map their polynomial over an 860x600 head / 900x510 wrist
        # raster instead of 960x600 -- a different FoV per pixel, not a crop.
        fisheye_nominal_width=ftheta_width,
        fisheye_nominal_height=ftheta_height,
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
        update_period=params.get("update_period", _UPDATE_PERIOD),
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=fisheye_cfg,
        offset=CameraCfg.OffsetCfg(
            pos=params["offset_pos"],
            rot=params["offset_rot"],
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
