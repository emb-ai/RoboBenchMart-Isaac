
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from dsynth_isaac.tasks.utils import object_grasped
from dsynth_isaac.planning.utils import get_item_bbox

if TYPE_CHECKING:
    from dsynth_isaac.tasks.manager_based.fetching.fetching_env import FetchingBaseEnv

def target_object_grasped(
    env: FetchingBaseEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    diff_threshold: float = 0.06,
    force_threshold: float = 1.0,
) -> torch.Tensor: # finished here
    """
    Check if an object is grasped by the specified robot.
    Support both surface gripper and parallel gripper.
    If contact_grasp sensor is found, check if the contact force is greater than force_threshold.
    """

    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    scene = env.scene
    # object: RigidObject = env.scene[object_cfg.name]

    # object_pos = object.data.root_pos_w
    # closest_item_ids = []
    closest_items_pos = []
    grasped = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)
    target_item_ids = env.get_target_items_ids()
    for env_id in range(env.num_envs):
        if env.get_grasped_item(env_id) is not None:
            is_item_grasped = object_grasped(env,
                robot_cfg,
                ee_frame_cfg,
                SceneEntityCfg(env.get_grasped_item(env_id)),
                diff_threshold=diff_threshold,
                force_threshold=force_threshold,
            )
            if is_item_grasped:
                grasped[env_id] = is_item_grasped
                break
            else:
                env.clear_grasped_item(env_id)

        min_distance = float('inf')
        for target_item_id in target_item_ids[env_id]:
            scene_item = scene[target_item_id]
            item_pos = scene_item.data.root_pos_w[0]
            distance = torch.linalg.vector_norm(item_pos - ee_frame.data.target_pos_w[env_id, 0, :])
            if distance < min_distance:
                min_distance = distance
                closest_item_id = target_item_id
        # closest_item_ids.append(closest_item_id)
        is_item_grasped = object_grasped(env,
                robot_cfg,
                ee_frame_cfg,
                SceneEntityCfg(closest_item_id),
                diff_threshold=diff_threshold,
                force_threshold=force_threshold,
            )
        if is_item_grasped:
            env.set_grasped_item(env_id, closest_item_id)
            grasped[env_id] = is_item_grasped

    return grasped


def is_pulled(
    env: FetchingBaseEnv,
    robot_cfg: SceneEntityCfg,
    diff_threshold: float = 0.06,
) -> torch.Tensor: # finished here
    """
    Check if an object is grasped by the specified robot.
    Support both surface gripper and parallel gripper.
    If contact_grasp sensor is found, check if the contact force is greater than force_threshold.
    """

    robot: Articulation = env.scene[robot_cfg.name]
    scene = env.scene
    scene_builder = env.scene.scene_builder

    pulled = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)

    def bboxes_intersect(bbox_a: dict, bbox_b: dict) -> bool:
        """Return True when two axis-aligned bboxes overlap."""
        center_a = torch.as_tensor(bbox_a["center"], dtype=torch.float32, device=env.device)
        center_b = torch.as_tensor(bbox_b["center"], dtype=torch.float32, device=env.device)
        half_a = 0.5 * torch.as_tensor(bbox_a["extents"], dtype=torch.float32, device=env.device)
        half_b = 0.5 * torch.as_tensor(bbox_b["extents"], dtype=torch.float32, device=env.device)
        delta = torch.abs(center_a - center_b)
        return bool(torch.all(delta <= (half_a + half_b)))

    for env_id in range(env.num_envs):
        if env.get_grasped_item(env_id) is not None:
            item_id = env.get_grasped_item(env_id)
            item_bbox = get_item_bbox(scene[item_id].prim_path)
            shelf_name = scene_builder.get_shelf_by_item_id(env_id, item_id)
            shelf = scene[shelf_name]
            shelf_bbox = get_item_bbox(shelf.prim_path)
            if bboxes_intersect(item_bbox, shelf_bbox):
                pulled[env_id] = True

    return pulled
