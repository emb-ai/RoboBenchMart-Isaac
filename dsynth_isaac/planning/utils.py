import numpy as np

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
