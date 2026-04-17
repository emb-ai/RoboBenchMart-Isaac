from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Union
from enum import Enum

import numpy as np


# class RandomAssetSampler:
#     def __init__(self, assets_dict: dict, assets_ids: list = [], regexp: str = None):
#         self.assets_dict = assets_dict
#         self.assets_ids = assets_ids
#         self.regexp = regexp

#     def __call__(self):
#         return True


class QuantitySpecifier:
    def __init__(self, int_quantity: int = 0, less_than: int = None, more_than: int = None):
        pass


@dataclass
class ShelvingConfig:
    shelving_id: str
    shelving_name: str


@dataclass
class SectionCfg:
    # scene id
    section_id: str
    section_name: str
    num_inactive_shelvings: Union[int, QuantitySpecifier] = 0

@dataclass
class SceneCfg:
    # scene id
    scene_id: str

    # Prop fixtures (inactive)
    prop_inner_fixtures_list: Optional[List[str]] = []
    num_prop_inner_fixtures: Union[int, QuantitySpecifier] = 0
    prop_wall_fixture_list: Optional[List[str]] = []
    num_wall_inner_fixtures: Union[int, QuantitySpecifier] = 0


    # list of sections
    section_list: Optional[List[SectionCfg]] = []


empty_scene_cfg_with_props = SceneCfg('empty_scene_props', prop_inner_fixtures_list=[PropInner() * 10
 ])




Node(Section, 'RANDOM[2:3]')
s1 = Shelving('...', products, **kwargs)
Section.add_node()