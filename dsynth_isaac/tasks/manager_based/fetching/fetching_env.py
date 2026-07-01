
from typing import Any, Sequence
import re
import torch

from dsynth_isaac.tasks.manager_based.darkstore_manager_based_rl_env import DarkstoreManagerBasedRLEnv
from dsynth_isaac.tasks.manager_based.fetching.fetching_env_cfg import FetchAnyItemEnvCfg

import logging
logger = logging.getLogger(__name__)

class FetchingBaseEnv(DarkstoreManagerBasedRLEnv):
    """Base environment for darkstore fetching tasks with target-item selection."""

    def setup_target_items(self, env_ids: Sequence[int] | torch.Tensor | None):
        """Populate target items cache for selected environment IDs."""
        raise NotImplementedError

    def get_target_items_ids(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        """Return target item IDs grouped by environment ID."""
        raise NotImplementedError

    def reset(
        self, seed: int | None = None, env_ids: Sequence[int] | None = None, options: dict[str, Any] | None = None
    ):
        """Reset envs and refresh per-env target-item cache for the same env IDs."""
        ret = super().reset(seed, env_ids, options)
        self.setup_target_items(env_ids)
        return ret

    def get_language_instruction(self):
        raise NotImplementedError

class FetchItemEnv(FetchingBaseEnv):
    """Fetch task where targets are selected by product-name regex per environment."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached_target_items = dict()

    def setup_target_items(self, env_ids):
        env_cfg: FetchAnyItemEnvCfg = self.cfg
        target_item_name_regex = env_cfg.target_item_name_regex

        if env_ids is None:
            env_ids_tensor = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        elif isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids.to(device=self.device, dtype=torch.int64)
        else:
            env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.int64, device=self.device)

        def _filter_fn(item_info):
            return re.search(target_item_name_regex, item_info['item_name_string']) is not None
        
        for env_id in env_ids_tensor.cpu().detach().tolist():
            if env_id not in self._cached_target_items:
                self._cached_target_items[env_id] = []

            product_items = self.scene.scene_builder.get_product_items(env_id)
            target_items_infos = list(filter(_filter_fn, product_items))
            if len(target_items_infos) == 0:
                raise RuntimeError(f"No target items found for env {env_id} with name regex {target_item_name_regex}")
            self._cached_target_items[env_id] = [target_item_info['item_id'] for target_item_info in target_items_infos]
            logger.info(f"Found {len(target_items_infos)} target items for env {env_id} with name regex {target_item_name_regex}")

    def get_target_items_ids(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        if len(self._cached_target_items) == 0:
            raise RuntimeError("Target items not setup. Call setup_target_items first.")
        if env_ids is None:
            return self._cached_target_items
        if isinstance(env_ids, torch.Tensor):
            env_ids_list = env_ids.detach().cpu().tolist()
        else:
            env_ids_list = list(env_ids)
        return {env_id: self._cached_target_items.get(env_id, []) for env_id in env_ids_list}

    def get_language_instruction(self):
        return self.cfg.language_instruction
