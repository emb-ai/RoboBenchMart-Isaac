import numpy as np
import torch
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from scipy.spatial.transform import Rotation

import carb
import omni

from isaaclab.devices.keyboard import Se3Keyboard, Se3KeyboardCfg
from isaaclab.devices import DeviceBase


class MobileSe3Keyboard(Se3Keyboard):        
    def advance(self) -> torch.Tensor:
        """Provides the result from keyboard event state.

        Returns:
            torch.Tensor: A 7-element tensor containing:
                - delta pose: First 6 elements as [x, y, z, rx, ry, rz] in meters and radians.
                - gripper command: Last element as a binary value (+1.0 for open, -1.0 for close).
        """
        # convert to rotation vector
        rot_vec = Rotation.from_euler("XYZ", self._delta_rot).as_rotvec()
        # return the command and gripper state
        command = np.concatenate([self._delta_pos, rot_vec])
        if self.gripper_term:
            gripper_value = -1.0 if self._close_gripper else 1.0
            command = np.append(command, gripper_value)
        command = np.concatenate((command, [0, 0, 0, 0 ,0]))
        return torch.tensor(command, dtype=torch.float32, device=self._sim_device)


@dataclass
class MobileSe3KeyboardCfg(Se3KeyboardCfg):
    class_type: type[DeviceBase] = MobileSe3Keyboard