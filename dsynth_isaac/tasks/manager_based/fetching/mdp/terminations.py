
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from dsynth_isaac.planning.utils import get_item_bbox
from dsynth_isaac.scene.scene_builder import SceneBuilder

if TYPE_CHECKING:
    from dsynth_isaac.tasks.manager_based.fetching.fetching_env import FetchingBaseEnv


def point_inside_bbox(
    point: torch.Tensor,
    bbox: dict,
    device: torch.device,
    outside_distance: float = 0.0,
) -> bool:
    """Return True when a 3D point lies inside an axis-aligned bbox.

    The point is considered outside only after it is farther than
    ``outside_distance`` beyond any bbox face.
    """
    center = torch.as_tensor(bbox["center"], dtype=torch.float32, device=device)
    half_extents = 0.5 * torch.as_tensor(bbox["extents"], dtype=torch.float32, device=device)
    half_extents = half_extents + float(outside_distance)
    lower = center - half_extents
    upper = center + half_extents
    point = point.to(dtype=torch.float32, device=device)
    return bool(torch.all((point >= lower) & (point <= upper)))


def target_object_grasped(
    env: FetchingBaseEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
    outside_distance: float = 0.1,
    check_if_pulled: bool = False,
) -> torch.Tensor:
    """
    Check whether each environment currently grasps a target object.

    The function uses per-env `contact_grasp_[env_i]` sensors if available and
    requires both fingers to exceed `force_threshold`. For fetch tasks
    (`check_if_pulled=True`), the object must additionally move outside shelf
    bounds.
    """
    scene = env.scene
    scene_builder: SceneBuilder = scene.scene_builder
    target_item_ids = env.get_target_items_ids()

    grasped = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)
    for env_idx in range(env.num_envs):
        contact_sensor_name = f"contact_grasp_[env_{env_idx}]"
        if contact_sensor_name not in env.scene.keys():
            continue

        contact_sensor = env.scene[contact_sensor_name]
        contact_data = contact_sensor.data
        contact_force_grasp = contact_data.net_forces_w  # shape:(1, 2, 3) for two fingers
        contact_force_norm = torch.linalg.vector_norm(contact_force_grasp, dim=2)  # shape:(1, 2)
        both_fingers_force_ok = torch.all(contact_force_norm > force_threshold, dim=1)

        contact_pos_w = contact_data.contact_pos_w
        non_nan_ids_left = ~torch.isnan(contact_pos_w[0, 0, :, 0])
        non_nan_ids_right = ~torch.isnan(contact_pos_w[0, 1, :, 0])
        non_nan_ids = torch.where(non_nan_ids_left & non_nan_ids_right)[0]
        if len(non_nan_ids) == 0:
            continue

        obj_id = int(non_nan_ids[0].item())
        filtered_paths = list(contact_sensor.cfg.filter_prim_paths_expr or [])
        if obj_id >= len(filtered_paths):
            continue

        grasped_item_path = filtered_paths[obj_id]
        product_items = scene_builder.get_product_items(env_idx)
        item_path_to_id = {item_info["item_prim_path"]: item_info["item_id"] for item_info in product_items}
        grasped_item_id = item_path_to_id.get(grasped_item_path)
        if grasped_item_id is None:
            continue
        if grasped_item_id not in target_item_ids.get(env_idx, []):
            continue

        grasped[env_idx] = bool(both_fingers_force_ok[0].item())
        if check_if_pulled and grasped[env_idx]:
            grasped_object = scene[grasped_item_id]
            grasped_object_pos = grasped_object.data.root_pos_w[0]
            shelf = scene_builder.get_shelf_by_item_id(env_idx, grasped_item_id)
            if shelf is None:
                continue
            shelf_bbox = get_item_bbox(shelf.cfg.prim_path)
            if point_inside_bbox(grasped_object_pos, shelf_bbox, env.device, outside_distance=outside_distance):
                grasped[env_idx] = False

    return grasped
