from __future__ import annotations

import math
import torch
import re
from typing import TYPE_CHECKING

import carb
from pxr import UsdLux, Gf
from isaaclab.sim.utils import get_prim_at_path
from isaacsim.core.utils.extensions import enable_extension
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaacsim.core.utils.extensions import enable_extension
import isaaclab.sim as sim_utils
from isaaclab.utils.version import compare_versions
from isaaclab.sim.utils.stage import get_current_stage
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation

from dsynth_isaac.scene.scene_manager import SceneManager

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def set_default_joint_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    default_pose: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # Set the default pose for robots in all envs
    asset = env.scene[asset_cfg.name]
    asset.data.default_joint_pos = torch.tensor(default_pose, device=env.device).repeat(env.num_envs, 1)

def reset_joints(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]

    # Add gaussian noise to joint states
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    # joint_pos += math_utils.sample_gaussian(mean, std, joint_pos.shape, joint_pos.device)

    # Clamp joint pos to limits
    # joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    # joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    # # Don't noise the gripper poses
    # joint_pos[:, -2:] = asset.data.default_joint_pos[env_ids, -2:]

    # Set into the physics simulation
    asset.set_joint_position_target(joint_pos, env_ids=env_ids)
    asset.set_joint_velocity_target(joint_vel, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def set_default_base_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    p = [0, 0, 0],
    q = [1, 0, 0, 0]
):
    # Set the default pose for robots in all envs
    asset = env.scene[asset_cfg.name]
    
    env_ids_list = list(range(env.num_envs)) if env_ids is None else env_ids.tolist()
    
    manager: SceneManager = env.scene.cfg.dsynth_scene.spawn.manager
    base_p, base_q = manager.get_robot_default_poses()
    
    new_poses = []
    p_offset = torch.tensor(p, device=env.device, dtype=torch.float32).unsqueeze(0)
    q_offset = torch.tensor(q, device=env.device, dtype=torch.float32).unsqueeze(0)
    for env_id in env_ids_list:
        pos, rot = math_utils.combine_frame_transforms(
                env.scene.env_origins[None, env_id] + base_p[None, env_id],
                base_q[None, env_id],
                p_offset,
                q_offset
            )
        new_poses.append(torch.cat([pos, rot], dim=-1))
    new_poses = torch.cat(new_poses, dim=-2)
    velocities = asset.data.default_root_state[env_ids_list, 7:13]

    asset.write_root_pose_to_sim(
        new_poses,
        env_ids=torch.tensor(env_ids_list, device=env.device)
    )
    asset.write_root_velocity_to_sim(
                    velocities,
                    torch.tensor(env_ids_list, device=env.device)
    )

    # asset.data.default_root_state[env_ids_list, :3] = new_poses[:, :3]
    # asset.data.default_root_state[env_ids_list, 3:7] = new_poses[:, 3:7]    

class randomize_visual_texture_material(ManagerTermBase):
    """Randomize the visual texture of bodies on an asset using Replicator API.

    This function randomizes the visual texture of the bodies of the asset using the Replicator API.
    The function samples random textures from the given texture paths and applies them to the bodies
    of the asset. The textures are projected onto the bodies and rotated by the given angles.

    .. note::
        The function assumes that the asset follows the prim naming convention as:
        "{asset_prim_path}/{body_name}/visuals" where the body name is the name of the body to
        which the texture is applied. This is the default prim ordering when importing assets
        from the asset converters in Isaac Lab.

    .. note::
        When randomizing the texture of individual assets, please make sure to set
        :attr:`isaaclab.scene.InteractiveSceneCfg.replicate_physics` to False. This ensures that physics
        parser will parse the individual asset properties separately.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.
        """
        super().__init__(cfg, env)

        # check to make sure replicate_physics is set to False, else raise error
        # note: We add an explicit check here since texture randomization can happen outside of 'prestartup' mode
        #   and the event manager doesn't check in that case.
        if env.cfg.scene.replicate_physics:
            raise RuntimeError(
                "Unable to randomize visual texture material with scene replication enabled."
                " For stable USD-level randomization, please disable scene replication"
                " by setting 'replicate_physics' to False in 'InteractiveSceneCfg'."
            )

        # enable replicator extension if not already enabled
        enable_extension("omni.replicator.core")

        # we import the module here since we may not always need the replicator
        import omni.replicator.core as rep

        # read parameters from the configuration
        asset_main_prim_paths = cfg.params.get("prim_paths")
        if isinstance(asset_main_prim_paths, str):
            asset_main_prim_paths = [asset_main_prim_paths]
        
        material_prims = []
        for asset_main_prim_path in asset_main_prim_paths:

            body_names_regex = ".*"

            # create the affected prim path
            # Check if the pattern with '/visuals' yields results when matching `body_names_regex`.
            # If not, fall back to a broader pattern without '/visuals'.
            pattern_with_visuals = f"{asset_main_prim_path}/{body_names_regex}/visuals"
            # Use sim_utils to check if any prims currently match this pattern
            matching_prims = sim_utils.find_matching_prim_paths(pattern_with_visuals)
            if matching_prims:
                # If matches are found, use the pattern with /visuals
                prim_path = pattern_with_visuals
            else:
                # If no matches found, fall back to the broader pattern without /visuals
                # This pattern (e.g., /World/envs/env_.*/Table/.*) should match visual prims
                # whether they end in /visuals or have other structures.
                prim_path = f"{asset_main_prim_path}/.*"
                carb.log_info(
                    f"Pattern '{pattern_with_visuals}' found no prims. Falling back to '{prim_path}' for texture"
                    " randomization."
                )

            # extract the replicator version
            version = re.match(r"^(\d+\.\d+\.\d+)", rep.__file__.split("/")[-5][21:]).group(1)

            # use different path for different version of replicator
            if compare_versions(version, "1.12.4") < 0:
                raise RuntimeError("Replicator version >1.12.4 is required for texture randomization.")
            else:
                # acquire stage
                stage = get_current_stage()
                prims_group = rep.functional.get.prims(path_pattern=prim_path, stage=stage)

                num_prims = len(prims_group)
                # rng that randomizes the texture and rotation
                self.texture_rng = rep.rng.ReplicatorRNG()

                # Create the material first and bind it to the prims
                for i, prim in enumerate(prims_group):
                    # Disable instancble
                    if prim.IsInstanceable():
                        prim.SetInstanceable(False)

                # TODO: Should we specify the value when creating the material?
                cur_material_prims = rep.functional.create_batch.material(
                    mdl="OmniPBR.mdl", bind_prims=prims_group, count=num_prims, project_uvw=True
                )
                if len(material_prims) > 0:
                    assert len(material_prims[-1]) == len(cur_material_prims)
                material_prims.append(cur_material_prims)
        self.material_prims = material_prims

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        event_name: str,
        prim_paths: str,
        texture_paths: list[str],
        texture_rotation: tuple[float, float] = (0.0, 0.0),
    ):
        # note: This triggers the nodes for all the environments.
        #   We need to investigate how to make it happen only for a subset based on env_ids.
        # we import the module here since we may not always need the replicator
        import omni.replicator.core as rep

        # extract the replicator version
        version = re.match(r"^(\d+\.\d+\.\d+)", rep.__file__.split("/")[-5][21:]).group(1)

        # use different path for different version of replicator
        if compare_versions(version, "1.12.4") < 0:
            raise RuntimeError("Replicator version >1.12.4 is required for texture randomization.")
        else:
            # read parameters from the configuration
            texture_paths = texture_paths if texture_paths else self._cfg.params.get("texture_paths")
            texture_rotation = (
                texture_rotation if texture_rotation else self._cfg.params.get("texture_rotation", (0.0, 0.0))
            )

            # convert from radians to degrees
            texture_rotation = tuple(math.degrees(angle) for angle in texture_rotation)

            num_prims = len(self.material_prims[0])
            random_textures = self.texture_rng.generator.choice(texture_paths, size=num_prims)
            random_rotations = self.texture_rng.generator.uniform(
                texture_rotation[0], texture_rotation[1], size=num_prims
            )

            for material_prim in self.material_prims:
                # modify the material properties
                rep.functional.modify.attribute(material_prim, "diffuse_texture", random_textures)
                rep.functional.modify.attribute(material_prim, "texture_rotate", random_rotations)
                rep.functional.modify.attribute(material_prim, "world_or_object", True)

def reset_items_uniform(
    env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        pose_range: dict[str, tuple[float, float]],
        velocity_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg = None,
    ):
    manager: SceneManager = env.scene.cfg.dsynth_scene.spawn.manager
    active_shelvings_xforms_poses = manager.get_active_shelvings_xforms_poses()
    items = manager.get_item_entities()

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    pos_ranges = torch.tensor(range_list, device=env.device)
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    vel_ranges = torch.tensor(range_list, device=env.device)

    for env_id in env_ids.tolist():
        for active_shelf in active_shelvings_xforms_poses[f'env_{env_id}'].keys():
            shelf_p, shelf_q = active_shelvings_xforms_poses[f'env_{env_id}'][active_shelf]
            for item in items[f'env_{env_id}'][active_shelf].values():
                item_state = item.data.default_root_state.clone()

                rand_samples = math_utils.sample_uniform(pos_ranges[:, 0], pos_ranges[:, 1], (1, 6), device=item.device)
                orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
                
                # poses
                position_wrt_world, or_wrt_world = math_utils.combine_frame_transforms(
                    env.scene.env_origins[None, env_id] + shelf_p, 
                    shelf_q,
                    item_state[:, 0:3] + rand_samples[:, 0:3],
                    math_utils.quat_mul(item_state[:, 3:7], orientations_delta)
                )

                # velocities
                rand_samples = math_utils.sample_uniform(vel_ranges[:, 0], vel_ranges[:, 1], (1, 6), device=item.device)

                velocities = item_state[:, 7:13] + rand_samples

                 # set into the physics simulation
                item.write_root_pose_to_sim(
                    torch.cat([position_wrt_world, or_wrt_world], dim=-1),
                    env_ids=torch.tensor([0], device=item.device))
                item.write_root_velocity_to_sim(
                    velocities,
                    env_ids=torch.tensor([0], device=item.device))

def randomize_lights(env, env_ids, intensity_range=(10000.0, 30000.0),
                     color_range=((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))):
    """Randomize per‑env lights for Isaac Lab 2.3.0 / Isaac Sim 5.1.

    Args:
        env: DirectRLEnv instance (self in your task).
        env_ids: 1D tensor of env indices to randomize.
        intensity_range: (min, max) intensity.
        color_range: ((r_min, g_min, b_min), (r_max, g_max, b_max)).
    """
    device = env.device
    env_ids = env_ids.to(device)
    num = env_ids.shape[0]

    # Sample intensities and colors in torch
    min_I, max_I = intensity_range
    intensities = torch.rand(num, device=device) * (max_I - min_I) + min_I

    min_c = torch.tensor(color_range[0], device=device)
    max_c = torch.tensor(color_range[1], device=device)
    colors = torch.rand(num, 3, device=device) * (max_c - min_c) + min_c
    
    # Apply to each env light
    for i, env_id in enumerate(env_ids.tolist()):
        prim_path = f"/World/envs/env_{env_id}/dsynth_scene/light_.*"
        prim_paths = sim_utils.find_matching_prim_paths(prim_path)
        for path in prim_paths:
            stage = sim_utils.get_current_stage()
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                continue

            light_api = UsdLux.LightAPI(prim)
            if not light_api:
                carb.log_warn(f"Prim is not a UsdLux light, skipping randomization: {path}")
                continue

            light_api.CreateIntensityAttr().Set(float(intensities[i].item()))
            col = colors[i]
            light_api.CreateColorAttr().Set(
                Gf.Vec3f(float(col[0].item()), float(col[1].item()), float(col[2].item()))
            )
