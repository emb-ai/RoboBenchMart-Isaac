
from typing import Any, Sequence
import re
import torch

from dsynth_isaac.tasks.manager_based.darkstore_manager_based_rl_env import DarkstoreManagerBasedRLEnv
from dsynth_isaac.tasks.manager_based.fetching.fetching_env_cfg import FetchAnyItemEnvCfg

import logging
logger = logging.getLogger(__name__)

class FetchingBaseEnv(DarkstoreManagerBasedRLEnv):
    def setup_target_items(self, env_ids):
        pass

    def get_target_items_ids(self, env_ids):
        pass

    def reset(
        self, seed: int | None = None, env_ids: Sequence[int] | None = None, options: dict[str, Any] | None = None
    ):
        ret = super().reset(seed, env_ids, options)
        self.setup_target_items(env_ids)
        return ret

    def get_language_instruction(self):
        pass

class FetchItemEnv(FetchingBaseEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached_target_items = dict()

    def setup_target_items(self, env_ids):
        env_cfg: FetchAnyItemEnvCfg = self.cfg
        target_item_name_regex = env_cfg.target_item_name_regex

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        
        product_items = self.scene.scene_builder._product_items # TODO: redo builder

        def _filter_fn(item_info):
            return re.match(target_item_name_regex, item_info['item_name_string'])
        
        for env_id in env_ids.cpu().detach().tolist():
            if env_id not in self._cached_target_items:
                self._cached_target_items[env_id] = []
            
            target_items_infos = list(filter(_filter_fn, product_items[f'env_{env_id}']))
            if len(target_items_infos) == 0:
                raise RuntimeError(f"No target items found for env {env_id} with name regex {target_item_name_regex}")
            self._cached_target_items[env_id] = [target_item_info['item_id'] for target_item_info in target_items_infos]
            logger.info(f"Found {len(target_items_infos)} target items for env {env_id} with name regex {target_item_name_regex}")

    def get_language_instruction(self):
        return self.cfg.language_instruction