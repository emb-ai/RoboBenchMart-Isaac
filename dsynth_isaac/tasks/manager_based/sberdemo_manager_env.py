import builtins
import logging
import warnings

import torch

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs.ui import ViewportCameraController
from isaaclab.managers import EventManager
from isaaclab.sim import SimulationContext
from isaaclab.sim.utils.stage import attach_stage_to_usd_context, use_stage
from isaaclab.utils.timer import Timer

from dsynth_isaac.scene.sberdemo_scene import SberDemoInteractiveScene, fix_store_usd_physics_for_isaac_lab

logger = logging.getLogger(__name__)


class SberDemoManagerBasedRLEnv(ManagerBasedRLEnv):
    """Manager-based RL environment wired to ``SberDemoInteractiveScene``."""

    _store_physics_fixed: bool = False

    def _ensure_store_physics_fixed(self) -> None:
        if self._store_physics_fixed:
            return
        fix_store_usd_physics_for_isaac_lab(
            self.scene,
            physics_prim_path=self.cfg.sim.physics_prim_path,
        )
        self._store_physics_fixed = True

    def reset(self, *args, **kwargs):
        if not self._store_physics_fixed:
            self._ensure_store_physics_fixed()
        return super().reset(*args, **kwargs)

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device, dtype=torch.long)

        cfg.validate()
        self.cfg = cfg
        self._is_closed = False

        if self.cfg.seed is not None:
            self.cfg.seed = self.seed(self.cfg.seed)
        else:
            logger.warning("Seed not set for the environment. The environment creation may not be deterministic.")

        if SimulationContext.instance() is None:
            self.sim: SimulationContext = SimulationContext(self.cfg.sim)
        else:
            if not builtins.ISAAC_LAUNCHED_FROM_TERMINAL:
                raise RuntimeError("Simulation context already exists. Cannot create a new one.")
            self.sim: SimulationContext = SimulationContext.instance()

        if "cuda" in self.device:
            torch.cuda.set_device(self.device)

        print("[INFO]: SberDemo environment:")
        print(f"\tEnvironment device    : {self.device}")
        print(f"\tEnvironment seed      : {self.cfg.seed}")
        print(f"\tPhysics step-size     : {self.physics_dt}")
        print(f"\tRendering step-size   : {self.physics_dt * self.cfg.sim.render_interval}")
        print(f"\tEnvironment step-size : {self.step_dt}")

        if self.cfg.sim.render_interval < self.cfg.decimation:
            msg = (
                f"The render interval ({self.cfg.sim.render_interval}) is smaller than the decimation "
                f"({self.cfg.decimation}). Multiple render calls will happen for each environment step. "
                "If this is not intended, set the render interval to be equal to the decimation."
            )
            logger.warning(msg)

        self._sim_step_counter = 0
        self.extras = {}

        with Timer("[INFO]: Time taken for scene creation", "scene_creation"):
            with use_stage(self.sim.get_initial_stage()):
                self.scene = SberDemoInteractiveScene(self.cfg.scene)
                attach_stage_to_usd_context()
        print("[INFO]: Scene manager: ", self.scene)

        if self.sim.render_mode >= self.sim.RenderMode.PARTIAL_RENDERING:
            self.viewport_camera_controller = ViewportCameraController(self, self.cfg.viewer)
        else:
            self.viewport_camera_controller = None

        self.event_manager = EventManager(self.cfg.events, self)
        if self.cfg.events is not None and "prestartup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="prestartup")

        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
            print("[INFO]: Starting the simulation. This may take a few seconds. Please wait...")
            with Timer("[INFO]: Time taken for simulation start", "simulation_start"):
                with use_stage(self.sim.get_initial_stage()):
                    self.sim.reset()
                self._ensure_store_physics_fixed()
                self.scene.update(dt=self.physics_dt)
            self.load_managers()

        if self.sim.has_gui() and self.cfg.ui_window_class_type is not None:
            self.setup_manager_visualizers()
            self._window = self.cfg.ui_window_class_type(self, window_name="IsaacLab")
        else:
            self._window = None

        self.obs_buf = {}

        if self.cfg.export_io_descriptors:
            self.export_IO_descriptors()

        if self.cfg.rerender_on_reset:
            msg = (
                "\033[93m\033[1m[DEPRECATION WARNING] ManagerBasedEnvCfg.rerender_on_reset is deprecated. Use"
                " ManagerBasedEnvCfg.num_rerenders_on_reset instead.\033[0m"
            )
            warnings.warn(msg, FutureWarning, stacklevel=2)
            if self.cfg.num_rerenders_on_reset == 0:
                self.cfg.num_rerenders_on_reset = 1

        self.render_mode = render_mode
        self.metadata["render_fps"] = 1 / self.step_dt
        print("[INFO]: Completed setting up the SberDemo environment...")
