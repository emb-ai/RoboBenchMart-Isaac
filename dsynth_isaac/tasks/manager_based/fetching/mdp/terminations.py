
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from dsynth_isaac.tasks.utils import object_grasped
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
) -> torch.Tensor: # finished here
    """
    Check if an object is grasped by the specified robot.
    Support both surface gripper and parallel gripper.
    If contact_grasp sensor is found, check if the contact force is greater than force_threshold.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    scene = env.scene
    scene_builder: SceneBuilder = scene.scene_builder
    target_item_ids = env.get_target_items_ids()

    grasped = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)
    for env_idx in range(env.num_envs):
        contact_sensor_name = f"contact_grasp_[env_{env_idx}]"
        if contact_sensor_name in env.scene.keys():
            contact_data = env.scene[contact_sensor_name].data
            contact_force_grasp = contact_data.net_forces_w # shape:(1, 2, 3) for two fingers
            contact_force_norm = torch.linalg.vector_norm(
                contact_force_grasp, dim=2
            )  # shape:(1, 2) - force magnitude per finger
            both_fingers_force_ok = torch.all(
                contact_force_norm > force_threshold, dim=1
            )  # both fingers must exceed threshold

            contact_pos_w = contact_data.contact_pos_w
            non_nan_ids_left = ~torch.isnan(contact_pos_w[0, 0, :, 0])
            non_nan_ids_right = ~torch.isnan(contact_pos_w[0, 1, :, 0])
            non_nan_ids = torch.where(non_nan_ids_left & non_nan_ids_right)[0]
            if len(non_nan_ids) > 0:
                obj_id = non_nan_ids[0].item()
                grapsed_item_path = scene_builder._all_item_prim_paths[f"env_{env_idx}"][obj_id]
                grasped_item_id = next(
                    (
                        item_info["item_id"]
                        for item_info in scene_builder._product_items[f"env_{env_idx}"]
                        if item_info["item_prim_path"] == grapsed_item_path
                    ),
                    None,
                )
                if grasped_item_id is not None and grasped_item_id in target_item_ids[env_idx]:
                    grasped[env_idx] = both_fingers_force_ok[0]
            
                if check_if_pulled and grasped_item_id is not None and grasped[env_idx]:
                    grasped_object = scene[grasped_item_id]
                    grasped_object_pos = grasped_object.data.root_pos_w[0]
                    shelf = scene_builder.get_shelf_by_item_id(env_idx, grasped_item_id)
                    shelf_bbox = get_item_bbox(shelf.cfg.prim_path)
                    if point_inside_bbox(grasped_object_pos, shelf_bbox, env.device, outside_distance=outside_distance):
                        grasped[env_idx] = False

    return grasped
