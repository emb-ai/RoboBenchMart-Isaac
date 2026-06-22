import torch
from isaaclab.scene import InteractiveScene
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    DeformableObject,
    DeformableObjectCfg,
    RigidObject,
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
    SurfaceGripper,
    SurfaceGripperCfg,
)
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, SensorBase, SensorBaseCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim.utils.stage import get_current_stage, get_current_stage_id
from isaaclab.sim.views import XformPrimView
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils.version import get_isaac_sim_version
from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg
import isaaclab.sim as sim_utils

from .darkstore_scene_cfg import DarkstoreInteractiveSceneCfg
from dsynth_isaac.scene.scene_builder import SceneBuilder

class DarkstoreInteractiveScene(InteractiveScene):
    def __init__(self, cfg: DarkstoreInteractiveSceneCfg):
        super().__init__(cfg)
        robot_fingers_contact_prim_path = cfg.robot_fingers_contact_prim_path
        if robot_fingers_contact_prim_path is not None:
            robot_fingers_contact_prim_path = robot_fingers_contact_prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
        self.scene_builder = SceneBuilder(
            config_dir_path=cfg.scene_config_dir_path,
            max_scenes=cfg.max_scenes,
            seed=cfg.seed,
            robot_fingers_contact_prim_path=robot_fingers_contact_prim_path
        )
        self.scene_builder.build(f'{self.env_regex_ns}/dsynth_scene')
        self._rigid_objects.update(self.scene_builder.get_rigid_bodies())
        self._sensors.update(self.scene_builder._contact_grasps)

        objects_per_env = {i: [] for i in range(self.cfg.num_envs)}
        for obj_name, obj in self._rigid_objects.items():
            if hasattr(obj, 'env_ids'):
                for env_id in obj.env_ids:
                    objects_per_env[env_id].append(obj_name)
            else:
                for env_id in range(self.cfg.num_envs):
                    objects_per_env[env_id].append(obj_name)
        self._objects_per_env = objects_per_env

    @property
    def rigid_objects_env(self, env_id: int):
        """A dictionary of rigid objects in the scene."""
        if env_id not in self._objects_per_env:
            raise ValueError(f"Environment {env_id} not found in the scene.")
        return {obj_name: self._rigid_objects[obj_name] for obj_name in self._objects_per_env[env_id]}

    def get_state(self, is_relative: bool = False) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
        state = super().get_state(is_relative)
        state["rigid_object"] = dict()
        for asset_name, rigid_object in self._rigid_objects.items():
            asset_state = dict()
            asset_state["root_pose"] = rigid_object.data.root_pose_w.clone()
            if is_relative:
                asset_state["root_pose"][:, :3] -= self.env_origins
            asset_state["root_velocity"] = rigid_object.data.root_vel_w.clone()
            if hasattr(rigid_object, 'env_ids'):
                asset_state["env_ids"] = rigid_object.env_ids
            else:
                asset_state["env_ids"] = [i for i in range(self.cfg.num_envs)]
            state["rigid_object"][asset_name] = asset_state
        return state

    def _add_entities_from_cfg(self):
        """Add scene entities from the config."""
        # store paths that are in global collision filter
        self._global_prim_paths = list()
        # parse the entire scene config and resolve regex
        for asset_name, asset_cfg in self.cfg.__dict__.items():
            # skip keywords
            # note: easier than writing a list of keywords: [num_envs, env_spacing, lazy_sensor_update]
            if asset_name in DarkstoreInteractiveSceneCfg.__dataclass_fields__ or asset_cfg is None:
                continue
            # resolve regex
            if hasattr(asset_cfg, "prim_path"):
                asset_cfg.prim_path = asset_cfg.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
            # create asset
            if isinstance(asset_cfg, TerrainImporterCfg):
                # terrains are special entities since they define environment origins
                asset_cfg.num_envs = self.cfg.num_envs
                asset_cfg.env_spacing = self.cfg.env_spacing
                self._terrain = asset_cfg.class_type(asset_cfg)
            elif isinstance(asset_cfg, ArticulationCfg):
                self._articulations[asset_name] = asset_cfg.class_type(asset_cfg)
            elif isinstance(asset_cfg, DeformableObjectCfg):
                self._deformable_objects[asset_name] = asset_cfg.class_type(asset_cfg)
            elif isinstance(asset_cfg, RigidObjectCfg):
                self._rigid_objects[asset_name] = asset_cfg.class_type(asset_cfg)
            elif isinstance(asset_cfg, RigidObjectCollectionCfg):
                for rigid_object_cfg in asset_cfg.rigid_objects.values():
                    rigid_object_cfg.prim_path = rigid_object_cfg.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
                self._rigid_object_collections[asset_name] = asset_cfg.class_type(asset_cfg)
                for rigid_object_cfg in asset_cfg.rigid_objects.values():
                    if hasattr(rigid_object_cfg, "collision_group") and rigid_object_cfg.collision_group == -1:
                        asset_paths = sim_utils.find_matching_prim_paths(rigid_object_cfg.prim_path)
                        self._global_prim_paths += asset_paths
            elif isinstance(asset_cfg, SurfaceGripperCfg):
                # add surface grippers to scene
                self._surface_grippers[asset_name] = asset_cfg.class_type(asset_cfg)
            elif isinstance(asset_cfg, SensorBaseCfg):
                # Update target frame path(s)' regex name space for FrameTransformer
                if isinstance(asset_cfg, FrameTransformerCfg):
                    updated_target_frames = []
                    for target_frame in asset_cfg.target_frames:
                        target_frame.prim_path = target_frame.prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
                        updated_target_frames.append(target_frame)
                    asset_cfg.target_frames = updated_target_frames
                elif isinstance(asset_cfg, ContactSensorCfg):
                    updated_filter_prim_paths_expr = []
                    for filter_prim_path in asset_cfg.filter_prim_paths_expr:
                        updated_filter_prim_paths_expr.append(filter_prim_path.format(ENV_REGEX_NS=self.env_regex_ns))
                    asset_cfg.filter_prim_paths_expr = updated_filter_prim_paths_expr
                elif isinstance(asset_cfg, VisuoTactileSensorCfg):
                    if hasattr(asset_cfg, "camera_cfg") and asset_cfg.camera_cfg is not None:
                        asset_cfg.camera_cfg.prim_path = asset_cfg.camera_cfg.prim_path.format(
                            ENV_REGEX_NS=self.env_regex_ns
                        )
                    if (
                        hasattr(asset_cfg, "contact_object_prim_path_expr")
                        and asset_cfg.contact_object_prim_path_expr is not None
                    ):
                        asset_cfg.contact_object_prim_path_expr = asset_cfg.contact_object_prim_path_expr.format(
                            ENV_REGEX_NS=self.env_regex_ns
                        )

                self._sensors[asset_name] = asset_cfg.class_type(asset_cfg)
            elif isinstance(asset_cfg, AssetBaseCfg):
                # manually spawn asset
                if asset_cfg.spawn is not None:
                    asset_cfg.spawn.func(
                        asset_cfg.prim_path,
                        asset_cfg.spawn,
                        translation=asset_cfg.init_state.pos,
                        orientation=asset_cfg.init_state.rot,
                    )
                # store xform prim view corresponding to this asset
                # all prims in the scene are Xform prims (i.e. have a transform component)
                self._extras[asset_name] = XformPrimView(asset_cfg.prim_path, device=self.device, stage=self.stage)
            else:
                raise ValueError(f"Unknown asset config type for {asset_name}: {asset_cfg}")
            # store global collision paths
            if hasattr(asset_cfg, "collision_group") and asset_cfg.collision_group == -1:
                asset_paths = sim_utils.find_matching_prim_paths(asset_cfg.prim_path)
                self._global_prim_paths += asset_paths
