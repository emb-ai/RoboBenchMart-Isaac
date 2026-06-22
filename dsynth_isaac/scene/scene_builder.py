from typing import Callable, Optional, Dict, Any
from dataclasses import MISSING, dataclass, field
from pathlib import Path
import json
import numpy as np
from transforms3d.euler import euler2quat, quat2mat
import hydra
import torch

from isaacsim.core.prims import XFormPrim
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners import materials
from isaacsim.core.prims import XFormPrim
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.sensors import ContactSensor, ContactSensorCfg

from dsynth_isaac.core.assets.asset import load_assets_lib
from dsynth_isaac.core.scene_gen.utils import flatten_dict
from dsynth_isaac.scene.utils import _get_absolute_matrix, _get_pq
from dsynth_isaac.core.assets.asset import Asset
from dsynth_isaac.scene.env_subset_rigid_object_cfg import EnvSubsetRigidObjectCfg
from dsynth_isaac.scene.env_subset_contact_sensor_cfg import EnvSubsetContactSensorCfg
from dsynth_isaac.scene.env_subset_contact_sensor import EnvSubsetContactSensor
PI = np.pi # for brevity
@dataclass
class SceneBuilder: # TODO: redo builder
    config_dir_path: str = MISSING
    max_scenes: Optional[int] = None
    seed: int = 42

    robot_fingers_contact_prim_path: str = None

    HEIGHT = 3.0
    THICKNESS = 0.2

    _entities: Dict[str, Any] = field(default_factory=dict)
    _product_items: Dict[int, Any] = field(default_factory=dict)
    _contact_grasps: Dict[int, Any] = field(default_factory=dict)
    _assets_lib: Dict[str, Any] = field(default_factory=dict)
    _num_envs: int = 0

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def build(self, dsynth_prim_path):
        self._entities['prim_path'] = dsynth_prim_path
        self._contact_grasps = {}
        dsynth_parent_path = "/".join(dsynth_prim_path.split('/')[:-1])
        dsynth_target_path = dsynth_prim_path.split('/')[-1]

        scene_config_paths = sorted([p.name for p in Path(self.config_dir_path).glob('*layout_data*.json')])
        num_generated_scenes = len(scene_config_paths)
        if num_generated_scenes == 0:
            raise ValueError(f"No layout data files found in {self.config_dir_path}")

        scene_config_paths = self._rng.permutation(scene_config_paths)
        
        with hydra.initialize_config_dir(config_dir=str(Path(self.config_dir_path).absolute()), version_base=None):
            hydra_cfg = hydra.compose(config_name='input_config')
        self._assets_lib = flatten_dict(load_assets_lib(hydra_cfg.assets), sep='.')
        
        num_max_scenes = min(num_generated_scenes,  self.max_scenes) if  self.max_scenes is not None else num_generated_scenes
        prim_paths = sim_utils.find_matching_prim_paths(dsynth_parent_path)
        self._num_envs = len(prim_paths)

        for i, prim_path in enumerate(prim_paths):
            self._entities[f'env_{i}'] = self._build_env(
                layout_json_path=scene_config_paths[i % num_max_scenes],
                prim_path=f"{prim_path}/{dsynth_target_path}",
                env_id=i
            )

        self._all_item_prim_paths = {}
        for env_id, env_items in self._product_items.items():
            env_idx = int(env_id.split('_')[-1])
            self._all_item_prim_paths[env_id] = []
            for item in env_items:
                self._all_item_prim_paths[env_id].append(item["item_prim_path"])

            if len(self._all_item_prim_paths[env_id]) > 0 and self.robot_fingers_contact_prim_path is not None:
                contact_prim_path = self.robot_fingers_contact_prim_path.replace('env_.*', env_id)
                self._contact_grasps[f"contact_grasp_[{env_id}]"] = EnvSubsetContactSensor(
                    EnvSubsetContactSensorCfg(
                        prim_path=contact_prim_path,
                        update_period=0,
                        history_length=1,
                        track_contact_points=True, # track?
                        debug_vis=True,
                        filter_prim_paths_expr=self._all_item_prim_paths[env_id],
                        env_ids=[env_idx],
                    )
                )
        
        
    def _build_env(self, layout_json_path: str, prim_path: str, env_id: int):
        sim_utils.create_prim(prim_path, "Xform")

        env_entities = {'prim_path': prim_path}
        with open(Path(self.config_dir_path) / layout_json_path, "r") as f:
            scene_data = json.load(f)
        env_entities['room'] = self._build_room(scene_data, prim_path, env_id)

        scene_data = self.choose_and_replace_active_shelving(scene_data)
        env_entities['inactive_shelvings'] = self._build_inactive_shelvings(scene_data, prim_path, env_id)
        env_entities['active_shelving'] = self._build_active_shelving(scene_data, prim_path, env_id)
        # env_entities['lights'] = self._build_lighting(scene_data, prim_path)
        lamps_coords = self._get_lamps_coords(scene_data['size_x'], scene_data['size_y'])
        self._build_lighting(scene_data, prim_path, lamps_coords)
        self._build_lamps(scene_data, prim_path, lamps_coords, env_id)

        env_entities['door'] = self._build_door(scene_data, prim_path, env_id)

        # stage = get_current_stage()
        # prim = stage.GetPrimAtPath(prim_path)
        # result = sim_utils.standardize_xform_ops(prim)
        # if result:
        #     print("Transform stack standardized successfully")
        # else:
        #     print("Transform stack standardization failed")

        return env_entities

    def _build_door(self, scene_data, prim_path, env_id: int):
        return self._build_rigid_asset(
            f"{prim_path}/door",
            'fixtures.door',
            p=[scene_data['size_x'] - self.THICKNESS, scene_data['size_y'] - self._assets_lib['fixtures.door'].extents[1], 0],
            q=[1, 0, 0, 0],
            kinematic_enabled=True,
            env_id=env_id
        )

    def _build_lamps(self, scene_data, prim_path, lamps_coords, env_id: int):
        for i, (x, y) in enumerate(lamps_coords):
            self._build_rigid_asset(
                f"{prim_path}/lamp_{i}",
                'fixtures.lamp',
                p=[x, y, self.HEIGHT],
                q=[1, 0, 0, 0],
                kinematic_enabled=True,
                env_id=env_id
            )

    def _build_lighting(self, scene_data, prim_path, lamps_coords, intensity=30000.0):
        lamp_height = self._assets_lib['fixtures.lamp'].extents[2]
        height = self.HEIGHT
        height -= lamp_height + 0.1
        # rot_qat = euler2quat(0, -PI / 2, 0)
        rot_qat = [1, 0, 0, 0]
        for i, (x, y) in enumerate(lamps_coords):
            light = AssetBaseCfg(
                # prim_path=f"{prim_path}/light", 
                spawn=sim_utils.DiskLightCfg(intensity=intensity)
            )
            light.spawn.func(
                            f"{prim_path}/light_{i}",
                            light.spawn,
                            translation=[x, y, height],
                            orientation=rot_qat,
                        )

    def _get_lamps_coords(self, x_size, y_size, num_lamps_x=4, num_lamps_y=4, dist_from_wall=0.0):
        lamps_coords = []
        step_x = (x_size - 2*dist_from_wall)/(num_lamps_x+1)
        step_y = (y_size - 2*dist_from_wall)/(num_lamps_y+1)
        for x in range(1, num_lamps_x+1):
            for y in range(1, num_lamps_y+1):
                lamps_coords.append((dist_from_wall+step_x*x, dist_from_wall+step_y*y))
        return lamps_coords

    def choose_and_replace_active_shelving(self, scene_data: dict):
        if len(scene_data['layout_data']["active_shelvings"]) > 0:
            # pick random suitable inactive shelf and replace it to active
            active_shelving = scene_data['layout_data']["active_shelvings"][0]
            inactive_shelvings = scene_data['layout_data']['inactive_shelvings']
            to_be_replaced_idxs = []
            for i, fixture in enumerate(inactive_shelvings):
                if fixture['asset_name'] == active_shelving['asset_name']:
                    to_be_replaced_idxs.append(i)
            if len(to_be_replaced_idxs) < 1:
                raise ValueError("No suitable shelvings!")
            to_be_replaced_shelf_idx = self._rng.choice(to_be_replaced_idxs)
            active_shelf = inactive_shelvings.pop(to_be_replaced_shelf_idx)
            # print(">>> to_be_replaced_shelf_idx", to_be_replaced_shelf_idx)
            
            active_shelving['x'] = active_shelf['x']
            active_shelving['y'] = active_shelf['y']
            active_shelving['orientation'] = active_shelf['orientation']

            scene_data['layout_data']["inactive_shelvings"] = inactive_shelvings
            scene_data['layout_data']["active_shelvings"][0] = active_shelving
            return scene_data
        raise ValueError("No active shelvings!")

    def _build_inactive_shelvings(self, scene_data: dict, prim_path: str, env_id: int):
        inactive_fixtures_categories = [
            "inactive_shelvings",
            "inactive_wall_shelvings",
            "scene_fixtures"
        ]
        inactive_shelvings = {}
        fake_shelfs_mapping = scene_data['fake_arrangements_mapping']
        for fixture_category in inactive_fixtures_categories:
            for i, inactive_fixture in enumerate(scene_data['layout_data'][fixture_category]):
                shelf_asset_name = inactive_fixture['asset_name']
                fixture_name = inactive_fixture['name']
                if shelf_asset_name in fake_shelfs_mapping:
                    fake_shelves_id = fake_shelfs_mapping[shelf_asset_name]
                    shelf_asset_name = self._rng.choice(fake_shelves_id)

                item_name = f'inactive_{fixture_name}_{i}_{shelf_asset_name}'
                item_name = item_name.replace(":", "_").replace(".", "_")
                p = np.array([inactive_fixture['x'], inactive_fixture['y'], 0.])
                angle = 0.
                if inactive_fixture['orientation'] == 'vertical':
                    angle = PI / 2.
                q = euler2quat(0, 0, angle)

                inactive_shelvings[item_name] = self._build_rigid_asset(
                    f"{prim_path}/{item_name}",
                    shelf_asset_name,
                    p=p, q=q,
                    kinematic_enabled=True,
                    env_id=env_id
                )

        return inactive_shelvings

    def _build_rigid_asset(self, prim_path, asset_name,
        p=[0., 0., 0.],
        q=[1., 0., 0., 0.],
        kinematic_enabled=False,
        env_id=0,
        activate_contact_sensors=False,
    ):
        asset_cfg = EnvSubsetRigidObjectCfg(
            prim_path=prim_path,
            env_ids=[env_id],
            spawn=sim_utils.UsdFileCfg(
                usd_path=self._assets_lib[asset_name].asset_file_path,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=kinematic_enabled),
                activate_contact_sensors=activate_contact_sensors,
                # collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=EnvSubsetRigidObjectCfg.InitialStateCfg(pos=p, rot=q),
        )
        return asset_cfg.class_type(asset_cfg)

    def _build_active_shelving(self, scene_data: dict, prim_path: str, env_id: int):
        active_fixtures_categories = [
            "active_shelvings",
            "active_wall_shelvings"
            ]

        active_shelvings = {
            'xforms': {}, 
            'shelvings': {}, 
            'items': {},
            'start_shelf_name': None,
            'robot_default_pose': None}


        for fixture_category in active_fixtures_categories:
            for i, active_fixture in enumerate(scene_data['layout_data'][fixture_category]):
                shelf_asset_name = active_fixture['asset_name']
                fixture_name = active_fixture['name']

                p = np.array([active_fixture['x'], active_fixture['y'], 0.])
                angle = 0.
                if active_fixture['orientation'] == 'vertical':
                    angle = PI / 2.
                q = euler2quat(0, 0, angle)

                # rotate shelf to the center of the scene
                shelf_direction = quat2mat(q)[:3, 1]
                direction_to_scene_center = np.array([scene_data['size_x']/ 2, scene_data['size_y'] / 2, 0.]) - p
                direction_to_scene_center /= (np.linalg.norm(direction_to_scene_center) + 1e-3)
                if np.dot(direction_to_scene_center, shelf_direction) > 0:
                    angle += PI
                    q = euler2quat(0, 0, angle)

                # make Xform near each active shelf to place the shelf items
                shelf_xform_prim_path = f"{prim_path}/xform_{fixture_name}"
                sim_utils.create_prim(shelf_xform_prim_path, "Xform", translation=p, orientation=q)
                active_shelvings['xforms'][fixture_name] = XFormPrim(shelf_xform_prim_path)

                shelf_prim_path = f"{shelf_xform_prim_path}/shelving"
                active_shelvings['shelvings'][fixture_name] = self._build_rigid_asset(
                    shelf_prim_path,
                    shelf_asset_name,
                    kinematic_enabled=True,
                    env_id=env_id
                )

                active_shelvings['items'][fixture_name] = self._build_arrangement(
                    shelf_xform_prim_path,
                    fixture_name,
                    env_id=env_id
                )


        # choose random active shelving and set default robot base pose near the shelf
        active_shelvings['start_shelf_name'] = self._rng.choice(sorted(list(active_shelvings['xforms'].keys())))
        p, q = active_shelvings['xforms'][active_shelvings['start_shelf_name']].get_local_poses()
                
        direction_to_shelf = quat2mat(q[0].cpu().numpy().tolist())[:3, 1]
        robot_default_pose_path = f"{prim_path}/robot_default_pose"
        robot_default_p = p[0].cpu().numpy() - 1.4 * direction_to_shelf
        sim_utils.create_prim(robot_default_pose_path, "Xform", translation=robot_default_p, orientation=euler2quat(0, 0, angle + PI / 2.))
        active_shelvings['robot_default_pose'] = XFormPrim(robot_default_pose_path)

        return active_shelvings

    def _build_arrangement(self, prim_path, fixture_name, env_id):
        with open(Path(self.config_dir_path) / f'{fixture_name}.json') as f:
            shelf_arrangement = json.load(f)

        if env_id not in self._product_items:
            self._product_items[f'env_{env_id}'] = []

        nodes_dict = {}
        for node in shelf_arrangement["graph"]:
            nodes_dict[node[1]] = node

        items_prim_path = f"{prim_path}/items"
        sim_utils.create_prim(items_prim_path, "Xform")

        arrangement = {}
        item_prim_paths = []
        for node in shelf_arrangement["graph"]:
            parent_name, obj_name, props = node
            if '/' not in obj_name and 'SHELF' not in obj_name:
                abs_matrix = _get_absolute_matrix(node, nodes_dict)
                p, q = _get_pq(abs_matrix, [0., 0., 0.])
                # q = [0.525322,  0, 0.8509035, 0]
                # p += np.array([0.0, 0.0, 3.0])
                asset_name = f'products_hierarchy.{obj_name.split(":")[0]}'
                asset: Asset = self._assets_lib[asset_name]

                # usd api does not like colons and dots in the names
                obj_name = obj_name.replace(':', '_').replace('.', '_')
                item_prim_path = f"{items_prim_path}/{obj_name.replace(':', '_').replace('.', '_')}"

                arrangement[obj_name] = self._build_rigid_asset(
                    item_prim_path,
                    asset_name,
                    p=p, q=q,
                    env_id=env_id,
                    activate_contact_sensors=True,
                )
                self._product_items[f'env_{env_id}'].append({
                    'item_id': f'[env_{env_id}]_[{fixture_name}]_[{obj_name}]',
                    'asset_id': asset_name,
                    'item_name_string': asset.asset_name,
                    'item_prim_path': item_prim_path,
                })
                item_prim_paths.append(item_prim_path)

        return arrangement

    def _build_room(self, scene_data: dict, prim_path: str, env_id: int):
        x_size = scene_data['size_x']
        y_size = scene_data['size_y']

        sizes = [
            (x_size, self.THICKNESS, self.HEIGHT),
            (self.THICKNESS, y_size, self.HEIGHT),
            (x_size, self.THICKNESS, self.HEIGHT),
            (self.THICKNESS, y_size, self.HEIGHT)
        ]

        translations = [
            (x_size / 2, - self.THICKNESS / 2, self.HEIGHT / 2),
            (x_size + self.THICKNESS / 2 , y_size / 2, self.HEIGHT / 2),
            (x_size / 2, y_size + self.THICKNESS / 2, self.HEIGHT / 2),
            (-self.THICKNESS / 2, y_size / 2, self.HEIGHT / 2)
        ]

        room = {}
        for i, (size, translation) in enumerate(zip(sizes, translations)):
            wall_cfg = EnvSubsetRigidObjectCfg(
                prim_path=f"{prim_path}/wall_{i}",
                env_ids=[env_id],
                spawn=sim_utils.CuboidCfg(size=size,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.MdlFileCfg(
                        mdl_path="assets_usd/mdl/bricks3/bricks3.mdl",
                        project_uvw=True,
                        texture_scale=(1., 1.)
                    ),
                ),
                init_state=EnvSubsetRigidObjectCfg.InitialStateCfg(pos=translation),
            )
            room[f'wall_{i}'] = wall_cfg.class_type(wall_cfg)

        ceiling_cfg = EnvSubsetRigidObjectCfg(
            prim_path=f"{prim_path}/ceiling",
            env_ids=[env_id],
            spawn=sim_utils.CuboidCfg(size=[x_size, y_size, self.THICKNESS],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.MdlFileCfg(
                    mdl_path="assets_usd/mdl/bricks3/bricks3.mdl",
                    project_uvw=True,
                    texture_scale=(1., 1.)
                ),
            ),
            init_state=EnvSubsetRigidObjectCfg.InitialStateCfg(pos=[x_size / 2, y_size / 2, self.HEIGHT + self.THICKNESS / 2]),
        )
        room['ceiling'] = ceiling_cfg.class_type(ceiling_cfg)

        floor_cfg = EnvSubsetRigidObjectCfg(
            prim_path=f"{prim_path}/floor",
            env_ids=[env_id],
            spawn=sim_utils.CuboidCfg(size=[x_size, y_size, self.THICKNESS],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.MdlFileCfg(
                    mdl_path="assets_usd/mdl/bricks3/bricks3.mdl",
                    project_uvw=True,
                    texture_scale=(1., 1.)
                ),
                physics_material=materials.RigidBodyMaterialCfg()
            ),
            init_state=EnvSubsetRigidObjectCfg.InitialStateCfg(pos=[x_size / 2, y_size / 2, - self.THICKNESS / 2]),
        )
        room['floor'] = floor_cfg.class_type(floor_cfg)

        return room

    def get_robot_default_poses(self):
        base_pose_xprim = XFormPrim(f'{self._entities["prim_path"]}/robot_default_pose')
        return base_pose_xprim.get_local_poses()

    def get_active_shelvings_xforms_poses(self):
        active_shelvings_xforms_poses = {}
        for env_id in self._entities.keys():
            if not 'env_' in env_id:
                continue
            active_shelvings_xforms_poses[env_id] = {}
            for active_shelf in self._entities[env_id]['active_shelving']['xforms'].keys():
                active_shelvings_xforms_poses[env_id][active_shelf] = self._entities[env_id]['active_shelving']['xforms'][active_shelf].get_local_poses()
        return active_shelvings_xforms_poses

    def get_item_entities(self):
        item_entities = {}
        for env_id in self._entities.keys():
            if not 'env_' in env_id:
                continue
            item_entities[env_id] = {}
            for active_shelf in self._entities[env_id]['active_shelving']['items'].keys():
                item_entities[env_id][active_shelf] = self._entities[env_id]['active_shelving']['items'][active_shelf]
        return item_entities

    def get_rigid_bodies(self):
        rigid_bodies = {}
        for env_id in self._entities.keys():
            if not 'env_' in env_id:
                continue
            for active_shelf in self._entities[env_id]['active_shelving']['items'].keys():
                rigid_body_key = f'[{env_id}]_[{active_shelf}]'
                for item_name in self._entities[env_id]['active_shelving']['items'][active_shelf].keys():
                    item_rigid_body_key = f'{rigid_body_key}_[{item_name}]'
                    rigid_bodies[item_rigid_body_key] = self._entities[env_id]['active_shelving']['items'][active_shelf][item_name]
                rigid_bodies[rigid_body_key] = self._entities[env_id]['active_shelving']['shelvings'][active_shelf]
            for inactive_shelf in self._entities[env_id]['inactive_shelvings'].keys():
                rigid_body_key = f'[{env_id}]_[{inactive_shelf}]'
                rigid_bodies[rigid_body_key] = self._entities[env_id]['inactive_shelvings'][inactive_shelf]
            
            for scene_fixture in self._entities[env_id]['room'].keys():
                rigid_body_key = f'[{env_id}]_[{scene_fixture}]'
                rigid_bodies[rigid_body_key] = self._entities[env_id]['room'][scene_fixture]
        return rigid_bodies


    def get_item_names(self):
        item_names = {}
        for env_id in self._entities.keys():
            if not 'env_' in env_id:
                continue
            item_names[env_id] = {}
            for active_shelf in self._entities[env_id]['active_shelving']['items'].keys():
                item_names[env_id][active_shelf] = list(self._entities[env_id]['active_shelving']['items'][active_shelf].keys())
        return item_names
    
    # def get_item_default_local_pose(self, env_id: int, item_name: str):
    #     return self._entities[env_id]['active_shelving']['items'][item_name].get_local_poses()

    def reset_items(self, env_ids: torch.Tensor, env_origins: torch.Tensor):
        pass

    def get_shelf_by_item_id(self, env_id: int, item_id: str):
        try:
            active_shelf = None
            shelf_name = item_id.split(']_[')[1]
            if shelf_name in self._entities[f'env_{env_id}']['active_shelving']['shelvings'].keys():
                active_shelf = self._entities[f'env_{env_id}']['active_shelving']['shelvings'][shelf_name]
        except:
            active_shelf = None

        if active_shelf is None:
            for active_shelf in self._entities[f'env_{env_id}']['active_shelving']['items'].keys():
                if item_id in self._entities[f'env_{env_id}']['active_shelving']['items'][active_shelf].keys():
                    return active_shelf
        return active_shelf