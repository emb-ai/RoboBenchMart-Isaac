import numpy as np
import torch

from pxr import UsdGeom, Usd
import omni.usd
import isaaclab.utils.math as math_utils

def np_normalize_vector(v):
    v = np.array(v)
    return v / (np.linalg.norm(v) + 1e-3)

def rodrigues_rotation(v, axis, theta):
    axis = axis / np.linalg.norm(axis)  # Ensure unit vector
    
    # Rodrigues' formula for vector rotation
    v_rot = (v * np.cos(theta) + 
             np.cross(axis, v) * np.sin(theta) + 
             axis * np.dot(axis, v) * (1 - np.cos(theta)))
    return v_rot

def compute_box_grasp_thin_side_info(
    obb_info,
    ee_direction,
    target_closing=None,
    depth=0.0,
    ortho=True,
    n_grasps = 1,
    approaching_angles_range = [-np.pi/4, np.pi/4],
):
    """Compute grasp info given an oriented bounding box.
    The grasp info includes axes to define grasp frame, namely approaching, closing, orthogonal directions and center.

    Args:
        obb: oriented bounding box to grasp
        approaching: direction to approach the object
        target_closing: target closing direction, used to select one of multiple solutions
        depth: displacement from hand to tcp along the approaching vector. Usually finger length.
        ortho: whether to orthogonalize closing  w.r.t. approaching.
    """
    # NOTE(jigu): DO NOT USE `x.extents`, which is inconsistent with `x.primitive.transform`!
    extents = np.array(obb_info["extents"])
    T = np.array(obb_info['transform'])
    center = np.array(obb_info['center'])

    for i in range(3):
        if np.abs(T[:3, i] @ [0, 0, 1]) > 1e-1:
            height_extent_ids = i
            break

    inds = np.argsort(extents)
    inds = inds[inds != height_extent_ids]
    short_base_side_ind = inds[0]
    long_base_side_ind = inds[1]

    # height = extents[2]

    approaching_angles = [0.]
    if n_grasps > 1:
        approaching_angles.extend(np.linspace(approaching_angles_range[0], approaching_angles_range[1], n_grasps - 1))
    
    approaching_vectors = []
    default_approaching = np.array(T[:3, long_base_side_ind])
    default_approaching = np_normalize_vector(default_approaching)

    if ee_direction.cpu().numpy() @ default_approaching < 0:
        default_approaching = -default_approaching

    normal_direction = np.cross(default_approaching, [0, 0, 1])

    for approaching_angle in approaching_angles:
        approaching = rodrigues_rotation(default_approaching, normal_direction, approaching_angle)
        approaching_vectors.append(approaching)

    closing = np.array(T[:3, short_base_side_ind])
    if target_closing is not None and target_closing @ closing < 0:
        closing = -closing

    half_size = extents[long_base_side_ind] / 2

    grasps = []
    for approaching, approaching_angle in zip(approaching_vectors, approaching_angles):
        half_size_rot = half_size / np.cos(approaching_angle)   
        origin = center + approaching * (-half_size_rot + min(depth, half_size_rot))

        if ortho:
            cur_closing = closing - (approaching @ closing) * approaching
            cur_closing = np_normalize_vector(cur_closing)

        grasp_info = dict(
            approaching=approaching, closing=cur_closing, center=origin, extents=extents
        )
        grasps.append(grasp_info)

    if len(grasps) == 1:
        return grasps[0]
    return grasps

def build_grasp_pose(approaching, closing, center):
    """Build a grasp pose (panda_hand_tcp)."""
    assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
    assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
    assert np.abs(approaching @ closing) <= 1e-3
    ortho = np.cross(closing, approaching)
    T = np.eye(4)
    T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
    T[:3, 3] = center
    return T

def np_make_pose(pos, quat):
    return math_utils.make_pose(torch.tensor(pos), math_utils.matrix_from_quat(torch.tensor(quat))).cpu().numpy()

def get_item_bbox(prim_path: str):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Invalid prim path: {prim_path}")

    xformable = UsdGeom.Xformable(prim)
    T_world = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())  # Gf.Matrix4d
    prim_world_pos = T_world.ExtractTranslation()
    prim_world_quat = T_world.ExtractRotationQuat()
    prim_world_quat = [prim_world_quat.GetReal(), *prim_world_quat.GetImaginary()]
    T = np_make_pose(prim_world_pos, prim_world_quat)

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    # Object-space bbox (before world/parent transforms)
    world_bbox = bbox_cache.ComputeWorldBound(prim)
    aabb = world_bbox.ComputeAlignedBox()
    bbox_min = np.array(aabb.GetMin())      # (x_min, y_min, z_min)
    bbox_max = np.array(aabb.GetMax())      
    bbox_center = 0.5 * (bbox_min + bbox_max)
    T[:3, 3] = bbox_center
    
    local_bbox = bbox_cache.ComputeLocalBound(prim)
    local_aabb = local_bbox.ComputeAlignedBox()
    bbox_min_local = np.array(local_aabb.GetMin())
    bbox_max_local = np.array(local_aabb.GetMax())
    bbox_extents_local = bbox_max_local - bbox_min_local
    
    target_bbox = {
        "center": bbox_center,
        "transform": T,
        "extents": bbox_extents_local
    }
    return target_bbox
