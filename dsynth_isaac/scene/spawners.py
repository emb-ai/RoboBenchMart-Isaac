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
 
def load_dsynth_multiscene(prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    print("WTF>>>", cfg.config_dir_path)
    scene_config_paths = sorted([p.name for p in Path(cfg.config_dir_path).glob('*layout_data*.json')])
    num_generated_scenes = len(scene_config_paths)
    if num_generated_scenes == 0:
        raise ValueError(f"No layout data files found in {cfg.config_dir_path}")

    with hydra.initialize_config_dir(config_dir=str(Path(cfg.config_dir_path).absolute()), version_base=None):
        hydra_cfg = hydra.compose(config_name='input_config')
    assets_lib = flatten_dict(load_assets_lib(hydra_cfg.assets), sep='.')


    num_max_scenes = min(num_generated_scenes, cfg.max_scenes) if cfg.max_scenes is not None else num_generated_scenes
    
    dsynth_prim_path = "/World/envs/env_.*/dsynth_scene"
    prim_paths = sim_utils.find_matching_prim_paths("/World/envs/env_.*")
    cfg.data = {}
    for i, prim_path in enumerate(prim_paths):
        dsynth_scene = DsynthSceneCfg(
            config_dir_path=cfg.config_dir_path,
            layout_json_path=scene_config_paths[i % num_max_scenes],
            assets_lib=assets_lib
        )
        dsynth_scene.func(f"{prim_path}/dsynth_scene", dsynth_scene, translation=(0.0, 0.0, 0.0))

        cfg.data[prim_path] = dsynth_scene



    # dsynth_scene = AssetBaseCfg(
    #     prim_path=dsynth_prim_path,
    #     spawn=sim_utils.MultiAssetSpawnerCfg(
    #         assets_cfg=[
    #             DsynthSceneCfg(
    #                 config_dir_path=cfg.config_dir_path,
    #                 layout_json_path=scene_config_paths[i],
    #                 assets_lib=assets_lib
    #             ) for i in range(num_max_scenes)
    #         ],
    #         random_choice=False,
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    # )
    # dsynth_scene.spawn.func(dsynth_prim_path, dsynth_scene.spawn, translation=(0.0, 0.0, 0.0))

# @clone
def load_dsynth_scene(prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:

    with open(Path(cfg.config_dir_path) / cfg.layout_json_path, "r") as f:
        scene_data = json.load(f)

    active_fixtures_categories = [
        "active_shelvings",
        "active_wall_shelvings"
        ]
    inactive_fixtures_categories = [
            "inactive_shelvings",
    ]
    x_size = scene_data['size_x']
    y_size = scene_data['size_y']
    HEIGHT = 3.0
    thk = 0.2


    wall_cfg = sim_utils.CuboidCfg(size=(x_size, thk, HEIGHT),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.MdlFileCfg(
                mdl_path="/home/kvsoshin/Work/RoboBenchMart-Isaac-Private/assets_usd/mdl/bricks3/bricks3.mdl",
                project_uvw=True,
                # scale=(1., 1.)
            )
    )
    wall_cfg.func(f"{prim_path}/wall_1", wall_cfg, translation=[x_size / 2, thk / 2, HEIGHT / 2])

    wall_cfg = sim_utils.CuboidCfg(size=(thk, y_size, HEIGHT),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.MdlFileCfg(
                mdl_path="/home/kvsoshin/Work/RoboBenchMart-Isaac-Private/assets_usd/mdl/bricks3/bricks3.mdl",
                project_uvw=True,
                # scale=(1., 1.)
            )
    )
    wall_cfg.func(f"{prim_path}/wall_2", wall_cfg, translation=[x_size, y_size / 2, HEIGHT / 2])

    wall_cfg = sim_utils.CuboidCfg(size=(x_size, thk, HEIGHT),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.MdlFileCfg(
                mdl_path="/home/kvsoshin/Work/RoboBenchMart-Isaac-Private/assets_usd/mdl/bricks3/bricks3.mdl",
                project_uvw=True,
                # scale=(1., 1.)
            )
    )
    wall_cfg.func(f"{prim_path}/wall_3", wall_cfg, translation=[x_size / 2, y_size, HEIGHT / 2])

    wall_cfg = sim_utils.CuboidCfg(size=(thk, y_size, HEIGHT),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.MdlFileCfg(
                mdl_path="/home/kvsoshin/Work/RoboBenchMart-Isaac-Private/assets_usd/mdl/bricks3/bricks3.mdl",
                project_uvw=True,
                # scale=(1., 1.)
            )
    )
    wall_cfg.func(f"{prim_path}/wall_4", wall_cfg, translation=[thk / 2, y_size / 2, HEIGHT / 2])

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
    for fixture_category in inactive_fixtures_categories:
        for i, inactive_fixture in enumerate(scene_data['layout_data'][fixture_category]):
            shelf_asset_name = inactive_fixture['asset_name']
            fixture_name = inactive_fixture['name']
            if shelf_asset_name in fake_shelfs_mapping:
                fake_shelves_id = fake_shelfs_mapping[shelf_asset_name]
                shelf_asset_name = np.random.choice(fake_shelves_id)


            item_name = f'inactive_{fixture_name}_{i}_{shelf_asset_name}'
            item_name = item_name.replace(":", "_").replace(".", "_")
            p = np.array([inactive_fixture['x'], inactive_fixture['y'], 0.])
            angle = 0.
            if inactive_fixture['orientation'] == 'vertical':
                angle = PI / 2.
            q = euler2quat(0, 0, angle)

            shelf_cfg = sim_utils.UsdFileCfg(
                usd_path=cfg.assets_lib[shelf_asset_name].asset_file_path,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            )
            shelf_cfg.func(f"{prim_path}/{item_name}", shelf_cfg, translation=p, orientation=q)

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
            robot_default_pose_path = f"{prim_path}/robot_default_pose"
            robot_default_p = p - 1.4 * direction_to_shelf
            sim_utils.create_prim(robot_default_pose_path, "Xform", translation=robot_default_p, orientation=euler2quat(0, 0, angle + PI / 2.))
            
            
            shelf_prim_path = f"{prim_path}/{shelf_prim_name}"
            sim_utils.create_prim(shelf_prim_path, "Xform", translation=p, orientation=q)

            shelf_cfg = sim_utils.UsdFileCfg(
                usd_path=cfg.assets_lib[shelf_asset_name].asset_file_path,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            )
            shelf_cfg.func(f"{shelf_prim_path}/shelving", shelf_cfg, translation=(0.0, 0.0, 0.0))
            
            # xform_prim = XFormPrim(shelf_prim_path)
            # t, o = xform_prim.get_local_pose()
            # xform_prim.set_local_poses(translations=[p.tolist()], orientations=[q.tolist()])
            # t, o = xform_prim.get_local_pose()
            
            

            with open(Path(cfg.config_dir_path) / f'{fixture_name}.json') as f:
                shelf_arrangement = json.load(f)

            nodes_dict = {}
            for node in shelf_arrangement["graph"]:
                nodes_dict[node[1]] = node

            for node in shelf_arrangement["graph"]:
                parent_name, obj_name, props = node
                if '/' not in obj_name and 'SHELF' not in obj_name:
                    abs_matrix = _get_absolute_matrix(node, nodes_dict)
                    p, q = _get_pq(abs_matrix, [0., 0., 0.])
                    p += np.array([0.0, 0.0, 3.0])
                    asset_name = f'products_hierarchy.{obj_name.split(":")[0]}'
                    asset = cfg.assets_lib[asset_name]

                    # asset_cfg = sim_utils.UsdFileCfg(
                    #     usd_path=asset.asset_file_path,
                    #     rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    #     collision_props=sim_utils.CollisionPropertiesCfg(),
                    #     mass_props=sim_utils.MassPropertiesCfg()
                    # )
                    # asset_cfg.func(f"{shelf_prim_path}/{obj_name.replace(':', '_').replace('.', '_')}", asset_cfg, translation=p, orientation=q)

                    asset_cfg = RigidObjectCfg(
                        prim_path=f"{shelf_prim_path}/{obj_name.replace(':', '_').replace('.', '_')}",
                        spawn=sim_utils.UsdFileCfg(usd_path=asset.asset_file_path,
                            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                            collision_props=sim_utils.CollisionPropertiesCfg(),
                        ),
                        init_state=RigidObjectCfg.InitialStateCfg(pos=p, rot=q),
                    )
                    cfg.obj = asset_cfg.class_type(asset_cfg)


    # Spawn the cube first
    # cube_cfg = sim_utils.CuboidCfg(size=(1.1, 1.1, 1.1),
    #                 rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    #             collision_props=sim_utils.CollisionPropertiesCfg(),
    #             visual_material=sim_utils.MdlFileCfg(
    #                 # mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Aluminum_Anodized.mdl",
    #                 # mdl_path="/home/kvsoshin/Downloads/Brick_Pavers.mdl",
    #                 mdl_path="/home/kvsoshin/Work/RoboBenchMart-Isaac-Private/assets_usd/mdl/my_bricks.mdl",
    #                 # mdl_path='assets/textures/walls/white_bricks_2.png',
    #                 project_uvw=True,
    #             ),
    #         )
    # cube_cfg.func(f"{prim_path}/my_cube/geometry", cube_cfg, translation=(0.0, 0.0, 3.0))
    # Then bind a material
    # material_cfg = sim_utils.MdlFileCfg(mdl_path='assets/textures/walls/white_bricks_2.png')
    # material_cfg = sim_utils.MdlFileCfg(mdl_path='{NVIDIA_NUCLEUS_DIR}/Materials/Base/Metals/Aluminum_Anodized.mdl')
    # bind_visual_material(f"{prim_path}/my_cube", material_cfg)

    # material_cfg = sim_utils.MdlFileCfg(mdl_path="home/kvsoshin/Downloads/Brick_Pavers.mdl")
    # material_cfg.func(f"{prim_path}/my_cube/material", material_cfg)
    # bind_visual_material(f"{prim_path}/my_cube/geometry", f"{prim_path}/my_cube/material")


    prim = prim_utils.get_prim_at_path(prim_path)
    return prim


def load_scene(cfg, config_dir_path, num_envs):
    scene_config_paths = sorted([p.name for p in Path(cfg.config_dir_path).glob('*layout_data*.json')])
    num_generated_scenes = len(scene_config_paths)
    if num_generated_scenes == 0:
        raise ValueError(f"No layout data files found in {cfg.config_dir_path}")

    

@configclass
class DsynthSceneCfg(SpawnerCfg):
    config_dir_path: str = MISSING
    layout_json_path: str = MISSING
    assets_lib: Dict[str, Any] = MISSING
    func: Callable = load_dsynth_scene
    
    obj: Any = None

@configclass
class DsynthMultiSceneCfg(SpawnerCfg):
    config_dir_path: str = MISSING
    max_scenes: Optional[int] = None

    func: Callable = load_dsynth_multiscene
    data: Any = None

