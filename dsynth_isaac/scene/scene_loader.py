from typing import Callable, Optional, Dict, Any
from dataclasses import MISSING
from pathlib import Path
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
import json
import numpy as np
from transforms3d.euler import euler2quat, quat2mat
from pxr import UsdGeom
import hydra

# from isaacsim.core.prims import XFormPrimView
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.prims import XFormPrim
from isaaclab.utils import configclass
from isaaclab.sim.spawners.spawner_cfg import DeformableObjectSpawnerCfg, RigidObjectSpawnerCfg, SpawnerCfg
import isaaclab.sim.utils.prims as prim_utils
from isaaclab.sim.utils import bind_physics_material, bind_visual_material, clone
import isaaclab.sim as sim_utils
import isaaclab.sim.schemas as sim_schemas
from isaaclab.assets import ArticulationCfg, AssetBase, AssetBaseCfg, RigidObjectCollectionCfg, RigidObjectCfg, RigidObjectCollection
# from isaaclab.sim.views import XformPrimView
from dsynth_isaac.core.assets.asset import load_assets_lib
from dsynth_isaac.core.scene_gen.utils import flatten_dict
from dsynth_isaac.scene.utils import _get_absolute_matrix, _get_pq

PI = np.pi # for brevity

def create_prim(prim_path: str, cfg: Any, translation: np.ndarray, orientation: np.ndarray, **kwargs) -> Usd.Prim:
    sim_utils.create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation)
    return XFormPrim(prim_path, reset_xform_properties=False)

@configclass
class XFormSpawnerCfg(SpawnerCfg):
    func: Callable = create_prim
    visible: bool = False
    
def load_scene(dsynth_cfg, config_dir_path, num_envs):
    scene_config_paths = sorted([p.name for p in Path(config_dir_path).glob('*layout_data*.json')])
    num_generated_scenes = len(scene_config_paths)
    if num_generated_scenes == 0:
        raise ValueError(f"No layout data files found in {config_dir_path}")

    with hydra.initialize_config_dir(config_dir=str(Path(config_dir_path).absolute()), version_base=None):
        hydra_cfg = hydra.compose(config_name='input_config')
    assets_lib = flatten_dict(load_assets_lib(hydra_cfg.assets), sep='.')
    
    for env_id in range(num_envs):
        scene_config_path = scene_config_paths[env_id % num_generated_scenes]
        with open(Path(config_dir_path) / scene_config_path, "r") as f:
            scene_data = json.load(f)
        
        env_prim_path = f"/World/envs/env_{env_id}"
        
        active_fixtures_categories = [
            "active_shelvings",
            "active_wall_shelvings"
            ]
        inactive_fixtures_categories = [
                "inactive_shelvings",
        ]

        if len(scene_data['layout_data']["active_shelvings"]) > 0:
            # pick random suitable inactive shelf and replace it to active
            active_shelving = scene_data['layout_data']["active_shelvings"][0]

            inactive_shelvings = scene_data['layout_data']['inactive_shelvings']
            to_be_replaced_idxs = []
            for i, fixture in enumerate(inactive_shelvings):
                if fixture['asset_name'] == active_shelving['asset_name']:
                    to_be_replaced_idxs.append(i)
            if len(to_be_replaced_idxs) < 1:
                return RuntimeError("No suitable shelvings!")
            to_be_replaced_shelf_idx = np.random.choice(to_be_replaced_idxs)
            active_shelf = inactive_shelvings.pop(to_be_replaced_shelf_idx)
            print(">>> to_be_replaced_shelf_idx", to_be_replaced_shelf_idx)
            
            active_shelving['x'] = active_shelf['x']
            active_shelving['y'] = active_shelf['y']
            active_shelving['orientation'] = active_shelf['orientation']

            scene_data['layout_data']["inactive_shelvings"] = inactive_shelvings
            scene_data['layout_data']["active_shelvings"][0] = active_shelving

        fake_shelfs_mapping = scene_data['fake_arrangements_mapping']
        # for fixture_category in inactive_fixtures_categories:
        #     for i, inactive_fixture in enumerate(scene_data['layout_data'][fixture_category]):
        #         shelf_asset_name = inactive_fixture['asset_name']
        #         fixture_name = inactive_fixture['name']
        #         if shelf_asset_name in fake_shelfs_mapping:
        #             fake_shelves_id = fake_shelfs_mapping[shelf_asset_name]
        #             shelf_asset_name = np.random.choice(fake_shelves_id)


        #         item_name = f'inactive_{fixture_name}_{i}_{shelf_asset_name}'
        #         item_name = item_name.replace(":", "_").replace(".", "_")
        #         p = np.array([inactive_fixture['x'], inactive_fixture['y'], 0.])
        #         angle = 0.
        #         if inactive_fixture['orientation'] == 'vertical':
        #             angle = PI / 2.
        #         q = euler2quat(0, 0, angle)

        #         dsynth_cfg.active_shelf = RigidObjectCfg(
        #             prim_path=f"{env_prim_path}/{item_name}",
        #             spawn=sim_utils.UsdFileCfg(
        #                 usd_path=assets_lib[shelf_asset_name].asset_file_path,
        #                 rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        #                 collision_props=sim_utils.CollisionPropertiesCfg(),
        #             ),
        #             init_state=RigidObjectCfg.InitialStateCfg(pos=p),
        #         )

        #         # shelf_cfg = sim_utils.UsdFileCfg(
        #         #     usd_path=cfg.assets_lib[shelf_asset_name].asset_file_path,
        #         #     rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        #         #     collision_props=sim_utils.CollisionPropertiesCfg(),
        #         # )
        #         # shelf_cfg.func(f"{prim_path}/{item_name}", shelf_cfg, translation=p, orientation=q)

        for fixture_category in active_fixtures_categories:
            for i, active_fixture in enumerate(scene_data['layout_data'][fixture_category]):
                shelf_asset_name = active_fixture['asset_name']
                fixture_name = active_fixture['name']
                shelf_prim_name = f'active_shelf_{fixture_name}_{i}'
                p = np.array([active_fixture['x'], active_fixture['y'], 0.])
                angle = 0.
                if active_fixture['orientation'] == 'vertical':
                    angle = PI / 2.
                q = euler2quat(0, 0, angle)

                shelf_direction = quat2mat(q)[:3, 1]
                direction_to_scene_center = np.array([scene_data['size_x']/ 2, scene_data['size_y'] / 2, 0.]) - p
                direction_to_scene_center /= (np.linalg.norm(direction_to_scene_center) + 1e-3)
                if np.dot(direction_to_scene_center, shelf_direction) > 0:
                    angle += PI
                    q = euler2quat(0, 0, angle)

                direction_to_shelf = quat2mat(q)[:3, 1]
                robot_default_pose_path = f"{env_prim_path}/robot_default_pose"
                robot_default_p = p - 1.4 * direction_to_shelf
                # sim_utils.create_prim(robot_default_pose_path, "Xform", translation=robot_default_p, orientation=euler2quat(0, 0, angle + PI / 2.))
                dsynth_cfg.__setattr__(f"robot_default_pose_{env_id}_{i}", AssetBaseCfg(
                    prim_path=robot_default_pose_path,
                    spawn=XFormSpawnerCfg(),
                    init_state=AssetBaseCfg.InitialStateCfg(pos=robot_default_p, rot=euler2quat(0, 0, angle + PI / 2.)),
                ))
                
                # shelf_prim_path = f"{prim_path}/{shelf_prim_name}"
                # sim_utils.create_prim(shelf_prim_path, "Xform", translation=p, orientation=q)

                dsynth_cfg.__setattr__(f"active_shelving_{env_id}_{i}", RigidObjectCfg(
                    prim_path=f"{env_prim_path}/active_shelving",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=assets_lib[shelf_asset_name].asset_file_path,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=p, rot=q),
                ))

                # shelf_cfg = sim_utils.UsdFileCfg(
                #     usd_path=cfg.assets_lib[shelf_asset_name].asset_file_path,
                #     rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                #     collision_props=sim_utils.CollisionPropertiesCfg(),
                # )
                # shelf_cfg.func(f"{shelf_prim_path}/shelving", shelf_cfg, translation=(0.0, 0.0, 0.0))
                
                

                # with open(Path(config_dir_path) / f'{fixture_name}.json') as f:
                #     shelf_arrangement = json.load(f)

                # nodes_dict = {}
                # for node in shelf_arrangement["graph"]:
                #     nodes_dict[node[1]] = node

                # for node in shelf_arrangement["graph"]:
                #     parent_name, obj_name, props = node
                #     if '/' not in obj_name and 'SHELF' not in obj_name:
                #         abs_matrix = _get_absolute_matrix(node, nodes_dict)
                #         p, q = _get_pq(abs_matrix, [0., 0., 0.])

                #         asset_name = f'products_hierarchy.{obj_name.split(":")[0]}'
                #         asset = assets_lib[asset_name]

                #         asset_cfg = sim_utils.UsdFileCfg(
                #             usd_path=asset.asset_file_path,
                #             rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                #             collision_props=sim_utils.CollisionPropertiesCfg(),
                #             mass_props=sim_utils.MassPropertiesCfg()
                #         )
                #         asset_cfg.func(f"{shelf_prim_path}/{obj_name.replace(':', '_').replace('.', '_')}", asset_cfg, translation=p, orientation=q)

