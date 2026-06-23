from __future__ import annotations

import re
import torch
import numpy as np
from typing import Sequence
from isaaclab.assets import RigidObject
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .env_subset_rigid_object_cfg import EnvSubsetRigidObjectCfg

class EnvSubsetRigidObject(RigidObject):
    """
    Rigid object wrapper for assets instantiated in only selected environments.

    Isaac Lab write/reset APIs receive global environment IDs, while the wrapped
    `RigidObject` stores data in local instance order. This class maps global
    env IDs to local instance indices before delegating to parent methods.
    """

    cfg: EnvSubsetRigidObjectCfg

    def __init__(self, cfg: EnvSubsetRigidObjectCfg):
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
        if len(matching_prims) != len(self.env_ids):
            raise RuntimeError(f"Number of matching prims {len(matching_prims)} does not match number of env ids {len(self.env_ids)}.")

        self._env_ids_to_prims = {}

        for matching_prim in matching_prims:
            prim_path = matching_prim.GetPath().pathString
            env_match = re.search(r"/env_(\d+)(?:/|$)", prim_path)
            if env_match is None:
                raise RuntimeError(f"Could not resolve env id from prim path: {prim_path}")
            env_id = int(env_match.group(1))
            if env_id not in self.env_ids:
                raise RuntimeError(f"Env id {env_id} not in {self.env_ids}.")
            self._env_ids_to_prims[env_id] = matching_prim
        
        if  set(self._env_ids_to_prims.keys()) != set(self.env_ids):
            raise RuntimeError(f"Env ids {set(self._env_ids_to_prims.keys())} do not match {set(self.env_ids)}.")

    @property
    def num_instances(self) -> int:
        num_instances = super().num_instances
        if num_instances != len(self.env_ids):
            raise RuntimeError(f"Number of instances {num_instances} does not match number of env ids {len(self.env_ids)}.")
        return num_instances

    def set_visibility(self, visible: bool, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = self.env_ids
        elif isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.detach().cpu().tolist()
        
        for env_id in env_ids:
            self._env_ids_to_prims[env_id].SetVisibility(visible)

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

    def reset(self, env_ids: Sequence[int] | None = None):
        # resolve all indices
        indices, _ = self._env_ids_to_indices(env_ids)
        if indices is None:
            return
        # reset external wrench
        self._instantaneous_wrench_composer.reset(indices)
        self._permanent_wrench_composer.reset(indices)

    def write_root_link_pose_to_sim(self, root_pose: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root link pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).

        Args:
            root_pose: Root link poses in simulation frame. Shape is (len(env_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
        """
        indices, env_ids_mask = self._env_ids_to_indices(env_ids)
        if indices is None:
            return
        super().write_root_link_pose_to_sim(root_pose[env_ids_mask], indices)
        
    def write_root_com_pose_to_sim(self, root_pose: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root center of mass pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).
        The orientation is the orientation of the principle axes of inertia.

        Args:
            root_pose: Root center of mass poses in simulation frame. Shape is (len(env_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # resolve all indices
        local_env_ids, env_ids_mask = self._env_ids_to_indices(env_ids)
        if local_env_ids is None:
            return
        
        # set into internal buffers
        self._data.root_com_pose_w[local_env_ids] = root_pose[env_ids_mask].clone()
        # update these buffers only if the user is using them. Otherwise this adds to overhead.
        if self._data._root_com_state_w.data is not None:
            self._data.root_com_state_w[local_env_ids, :7] = self._data.root_com_pose_w[local_env_ids]

        # get CoM pose in link frame
        com_pos_b = self.data.body_com_pos_b[local_env_ids, 0, :]
        com_quat_b = self.data.body_com_quat_b[local_env_ids, 0, :]
        # transform input CoM pose to link frame
        root_link_pos, root_link_quat = math_utils.combine_frame_transforms(
            root_pose[env_ids_mask, :3],
            root_pose[env_ids_mask, 3:7],
            math_utils.quat_apply(math_utils.quat_inv(com_quat_b), -com_pos_b),
            math_utils.quat_inv(com_quat_b),
        )
        root_link_pose = torch.cat((root_link_pos, root_link_quat), dim=-1)

        # write transformed pose in link frame to sim
        super().write_root_link_pose_to_sim(root_link_pose, env_ids=local_env_ids)

    def write_root_com_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's center of mass rather than the roots frame.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (len(env_ids), 6).
            env_ids: Environment indices. If None, then all indices are used.
        """
        indices, env_ids_mask = self._env_ids_to_indices(env_ids)
        if indices is None:
            return
        super().write_root_com_velocity_to_sim(root_velocity[env_ids_mask], indices)
    
    def write_root_link_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root link velocity over selected environment indices into the simulation.

        The velocity comprises linear velocity (x, y, z) and angular velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's frame rather than the roots center of mass.

        Args:
            root_velocity: Root frame velocities in simulation world frame. Shape is (len(env_ids), 6).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # resolve all indices
        local_env_ids, env_ids_mask = self._env_ids_to_indices(env_ids)
        if local_env_ids is None:
            return

        # set into internal buffers
        self._data.root_link_vel_w[local_env_ids] = root_velocity[env_ids_mask].clone()
        # update these buffers only if the user is using them. Otherwise this adds to overhead.
        if self._data._root_link_state_w.data is not None:
            self._data.root_link_state_w[local_env_ids, 7:] = self._data.root_link_vel_w[local_env_ids]

        # get CoM pose in link frame
        quat = self.data.root_link_quat_w[local_env_ids]
        com_pos_b = self.data.body_com_pos_b[local_env_ids, 0, :]
        # transform input velocity to center of mass frame
        root_com_velocity = root_velocity[env_ids_mask].clone()
        root_com_velocity[:, :3] += torch.linalg.cross(
            root_com_velocity[:, 3:], math_utils.quat_apply(quat, com_pos_b), dim=-1
        )

        # write transformed velocity in CoM frame to sim
        super().write_root_com_velocity_to_sim(root_com_velocity, env_ids=local_env_ids)

    def set_external_force_and_torque(
        self,
        forces: torch.Tensor,
        torques: torch.Tensor,
        positions: torch.Tensor | None = None,
        body_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | None = None,
        is_global: bool = False,
    ):
        indices, env_ids_mask = self._env_ids_to_indices(env_ids)
        if indices is None:
            return

        masked_positions = positions[env_ids_mask] if positions is not None else None
        super().set_external_force_and_torque(forces[env_ids_mask],
            torques[env_ids_mask],
            masked_positions,
            body_ids,
            indices,
            is_global
        )

    def _create_buffers(self):
        super()._create_buffers()
        self.env_ids_pt = torch.tensor(self.env_ids, dtype=torch.int32, device=self.device)