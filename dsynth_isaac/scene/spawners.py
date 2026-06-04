from typing import Callable, Optional, Any
from dataclasses import MISSING
from pxr import Usd
# from isaacsim.core.prims import XFormPrimView
from isaaclab.utils import configclass
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg

from dsynth_isaac.scene.scene_builder import SceneBuilder

 
def load_dsynth_scene(prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    cfg.manager = SceneManager(
        config_dir_path=cfg.config_dir_path,
        max_scenes=cfg.max_scenes,
        seed=cfg.seed
    )
    cfg.manager.build(prim_path)
    
@configclass
class DsynthSceneCfg(SpawnerCfg):
    config_dir_path: str = MISSING
    max_scenes: Optional[int] = None
    seed: int = 42

    func: Callable = load_dsynth_scene
    manager: Any = None
