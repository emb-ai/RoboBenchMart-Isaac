import os
from dataclasses import dataclass, field
from pathlib import Path
import logging
from typing import Dict, Any, IO, BinaryIO, Union, Literal, Optional, List, Sequence
from omegaconf import DictConfig, OmegaConf
from transforms3d import quaternions
import torch
import numpy as np
import trimesh

import scene_synthesizer as synth
from scene_synthesizer import utils

logger = logging.getLogger(__name__)

ASSET_TYPE_MAPPING = {
    "MeshAsset": synth.assets.MeshAsset,
    "USDAsset": synth.assets.USDAsset,
    "URDFAsset": synth.assets.URDFAsset,
    "Asset": synth.Asset
}

@dataclass
class Asset:
    asset_file_path: Union[str, os.PathLike, BinaryIO, IO[bytes]]
    ss_asset_type: Any = 'USDAsset'
    ss_params: Dict = field(default_factory = lambda: {})
    ms_is_static: bool = False # not used in Isaac Sim setup
    ms_is_nonconvex_collision: bool = False # not used in Isaac Sim setup

    disable_caching: bool = False
    _ss_asset: Any = None
    _ss_asset_convex: Any = None
    _trimesh_scene: Any = None
    _ms_scale: Any = None # not used in Isaac Sim setup
    _ms_origin: Any = None # not used in Isaac Sim setup

    asset_name: str = 'asset'

    def __post_init__(self):
        if not Path(self.asset_file_path).exists():
            logger.warning(f'Asset path {str(self.asset_file_path)} does not exist!')
        if Path(self.asset_file_path).suffix != '.usd':
            raise ValueError(f"Asset file path {str(self.asset_file_path)} must have a .usd extension!")

        if self.ss_asset_type != 'USDAsset':
            raise ValueError(f"Asset type {self.ss_asset_type} is not supported in Isaac Sim setup! Only USDAsset is supported!")

    @property
    def ss_asset(self):
        if self._ss_asset is not None:
            return self._ss_asset
        
        contructor = ASSET_TYPE_MAPPING.get(self.ss_asset_type, None)
        if contructor is None:
            raise ValueError(f"Wrong asset type: {self.ss_asset_type}, possible values: {list(ASSET_TYPE_MAPPING.keys())}")
        
        ss_asset = contructor(self.asset_file_path, **self.ss_params)
        if not self.disable_caching:
            self._ss_asset = ss_asset

        return ss_asset

    @property
    def ss_asset_convex(self):
        if self._ss_asset_convex is not None:
            return self._ss_asset_convex
        
        ss_asset_convex = self.ss_asset
        ss_asset_convex._model = trimesh.Scene(ss_asset_convex._model.convex_hull)

        if not self.disable_caching:
            self._ss_asset_convex = ss_asset_convex

        return ss_asset_convex
        
    
    @property
    def trimesh_scene(self):
        if self._trimesh_scene is not None:
            return self._trimesh_scene
        
        trimesh_scene = self.ss_asset.as_trimesh_scene()
        if not self.disable_caching:
            self._trimesh_scene = trimesh_scene

        return trimesh_scene
    
    @property
    def extents(self):
        return self.trimesh_scene.extents
    
    def scale_and_transform(self):
        use_collision_geometry = self.ss_params.get('use_collision_geometry', True)
        trimesh_scene = self.ss_asset._as_trimesh_scene(
            namespace="", use_collision_geometry=use_collision_geometry
        )
        trimesh_scene = utils.normalize_and_bake_scale(trimesh_scene)
        scale = self.ss_asset._get_scale(raw_extents=trimesh_scene.extents)
        
        scaled_scene = utils.scaled_trimesh_scene(trimesh_scene, scale=scale)
        center_mass = utils.center_mass(trimesh_scene=scaled_scene, node_names=scaled_scene.graph.nodes_geometry)
        origin = self.ss_asset._get_origin_transform(
            bounds=scaled_scene.bounds,
            center_mass= center_mass,
            centroid=scaled_scene.centroid,
        )
        return scale, origin
    
    @property
    def ms_scale(self):
        if self._ms_scale is not None:
            return self._ms_scale
        
        scale, origin = self.scale_and_transform()
        if not self.disable_caching:
            self._ms_scale, self._ms_origin = scale, origin

        return scale
    
    @property
    def ms_origin(self):
        if self._ms_origin is not None:
            return self._ms_origin

        scale, origin = self.scale_and_transform()
        if not self.disable_caching:
            self._ms_scale, self._ms_origin = scale, origin

        return origin
    
    def ms_build_actor(
        self,
        obj_name,
        scene,
        pose = None,
        T = None,
        scene_idxs = None,
        force_static = False
    ):
        raise NotImplementedError("This method is not used in Isaac Sim setup. Please refer to the original implementation https://github.com/emb-ai/RoboBenchMart")

    def load_actor_as_urdf(self, obj_name, scene, pose, scale, scene_idxs):
        raise NotImplementedError("This method is not used in Isaac Sim setup. Please refer to the original implementation https://github.com/emb-ai/RoboBenchMart")

def load_assets_lib(products_hierarchy_dict: DictConfig, disable_caching=False):
    assets_dict = {}
    products_dict = OmegaConf.to_container(products_hierarchy_dict, resolve = True)
    if 'asset_file_path' in products_dict.keys():
        return Asset(**products_dict, disable_caching = disable_caching)
    for key, val in products_dict.items():
        if not isinstance(val, Dict):
            assets_dict[key] = val
        else:
            assets_dict[key] = load_assets_lib(products_hierarchy_dict[key], disable_caching=disable_caching)
    return assets_dict


if __name__ == '__main__':
    a = {
        'asset_file_path': 'sasha_assets/milkHandle.glb',
        'ss_params': {
            'height': 0.25,
            'up': [0, 1, 0],
            'front': [0, 0, -1] ,
            'origin': ["left", "bottom", "com"]
            # 'up': [0, 1, 0], 
            # 'front': [0, 0, -1], 
            # 'origin': ["left", "bottom", "com"]
        }
    }
    a = Asset(**a)
    scale = a.scale

    print(a)