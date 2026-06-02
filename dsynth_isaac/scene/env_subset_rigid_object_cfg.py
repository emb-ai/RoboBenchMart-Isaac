from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass
from dataclasses import MISSING

from .env_subset_rigid_object import EnvSubsetRigidObject

@configclass
class EnvSubsetRigidObjectCfg(RigidObjectCfg):
    env_ids: list[int] = MISSING

    class_type: type = EnvSubsetRigidObject