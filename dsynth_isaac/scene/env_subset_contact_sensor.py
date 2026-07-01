from __future__ import annotations

import re
import torch
import numpy as np
from typing import Sequence
from isaaclab.sensors import ContactSensor
import isaaclab.sim as sim_utils

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .env_subset_contact_sensor_cfg import EnvSubsetContactSensorCfg


class EnvSubsetContactSensor(ContactSensor):
    """
    Contact sensor wrapper for assets present in only selected environments.

    The sensor update loop passes local buffer indices, while scene resets pass
    global environment IDs. This class only remaps in reset-like paths.
    """

    cfg: EnvSubsetContactSensorCfg

    def __init__(self, cfg: EnvSubsetContactSensorCfg):
        super().__init__(cfg)
        self.env_ids = cfg.env_ids

        if isinstance(self.env_ids, torch.Tensor):
            self.env_ids = self.env_ids.detach().cpu()
        self.env_ids = np.array(self.env_ids, dtype=np.int32)
        
        assert self.env_ids.ndim == 1
        
        self.env_ids = np.sort(self.env_ids)
        if np.unique(self.env_ids).shape[0] != len(self.env_ids):
            raise RuntimeError(f"Env ids {self.env_ids} are not unique.")
        
        self.env_ids = self.env_ids.tolist()

        matching_prims = sim_utils.find_matching_prims(self.cfg.prim_path)
        if len(matching_prims) == 0:
            raise RuntimeError(
                f"No matching prims found for contact sensor path: {self.cfg.prim_path}"
            )
        env_ids_from_prims = []
        for matching_prim in matching_prims:
            prim_path = matching_prim.GetPath().pathString
            env_match = re.search(r"/env_(\d+)(?:/|$)", prim_path)
            if env_match is None:
                raise RuntimeError(f"Could not resolve env id from prim path: {prim_path}")
            env_ids_from_prims.append(int(env_match.group(1)))
        if set(env_ids_from_prims) != set(self.env_ids):
            raise RuntimeError(f"Env ids from prims {sorted(set(env_ids_from_prims))} do not match {self.env_ids}.")

    def _env_ids_to_indices(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids_to_resolve = self.env_ids
        elif isinstance(env_ids, torch.Tensor):
            env_ids_to_resolve = env_ids.detach().cpu().tolist()
        else:
            env_ids_to_resolve = list(env_ids)

        indices = []
        env_ids_mask = []
        for env_id in env_ids_to_resolve:
            if env_id not in self.env_ids:
                env_ids_mask.append(False)
                continue
            indices.append(self.env_ids.index(env_id))
            env_ids_mask.append(True)

        if len(indices) == 0:
            return None, env_ids_mask
        indices = np.array(indices, dtype=np.int32)
        if isinstance(env_ids, torch.Tensor):
            indices = torch.from_numpy(indices).to(dtype=torch.long, device=env_ids.device)
        return indices, env_ids_mask

    def _initialize_impl(self):
        super()._initialize_impl()
        if self._num_envs != len(self.env_ids):
            raise RuntimeError(
                f"Number of resolved sensor envs ({self._num_envs}) does not match configured env_ids "
                f"({len(self.env_ids)}). prim_path={self.cfg.prim_path}"
            )

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        # `SensorBase` passes local sensor indices here, not global env IDs.
        super()._update_buffers_impl(env_ids)

    def reset(self, env_ids: Sequence[int] | None = None):
        idxs, _ = self._env_ids_to_indices(env_ids)
        if idxs is not None:
            super().reset(idxs)
