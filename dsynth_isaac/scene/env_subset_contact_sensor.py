from __future__ import annotations

import torch
import numpy as np
from typing import Sequence
from isaaclab.sensors import ContactSensor, SensorBase
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .env_subset_contact_sensor_cfg import EnvSubsetContactSensorCfg


class EnvSubsetContactSensor(ContactSensor):
    """
    A sensor that exists only in a subset of environments.
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

    def _env_ids_to_indices(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids_to_resolve = self.env_ids
        elif isinstance(env_ids, torch.Tensor):
            env_ids_to_resolve = env_ids.detach().cpu().tolist()

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
            indices = torch.from_numpy(indices).to(dtype=torch.int32, device=env_ids.device)
        return indices, env_ids_mask

    def _initialize_impl(self):
        super()._initialize_impl()
        assert self._num_envs == len(self.env_ids)
        # super()._initialize_impl()

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        idxs, _ = self._env_ids_to_indices(env_ids)
        if not idxs is None:
            super()._update_buffers_impl(idxs)

    def reset(self, env_ids: Sequence[int] | None = None):
        idxs, _ = self._env_ids_to_indices(env_ids)
        if not idxs is None:
            super().reset(idxs)