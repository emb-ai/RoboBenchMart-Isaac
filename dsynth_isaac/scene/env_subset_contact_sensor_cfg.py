from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from dataclasses import MISSING

from .env_subset_contact_sensor import EnvSubsetContactSensor


@configclass
class EnvSubsetContactSensorCfg(ContactSensorCfg):
    """ContactSensor config for sensors instantiated on a subset of envs."""

    env_ids: list[int] = MISSING

    class_type: type = EnvSubsetContactSensor
