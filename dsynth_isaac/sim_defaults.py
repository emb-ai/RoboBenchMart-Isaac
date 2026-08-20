"""CLI simulation defaults shared by the Sberchel entry-point scripts.

Imported before ``AppLauncher`` starts the app, so this module must stay free of
``isaaclab`` / ``isaacsim`` imports.
"""

from __future__ import annotations

# green_challenge evaluates with ``sim.device = "cpu"``
# (``envs/env_registration.make_env``), so demos are collected there too: CPU and
# GPU PhysX are different implementations and the device is part of the plant.
#
# It is also simply better here. Measured on the demo task, single env, identical
# otherwise -- standing 15 s at zero velocity command and walking 16 s:
#
#   device    stand path   base pitch   walk distance   sim speed
#   cuda:0    0.056 m      -4.60 deg    5.59 m          17.5 steps/s
#   cpu       0.056 m      -4.70 deg    7.53 m          39.9 steps/s
#
# With ``num_envs=1`` the GPU pipeline's per-step synchronisation costs more than
# it saves, and the rest of the loop (zest policy inference, Pink IK) is on CPU
# anyway. Rendering is unaffected -- it stays on the RTX device, and all four
# robot cameras were verified to produce images with physics on CPU.
SBERCHEL_DEFAULT_DEVICE = "cpu"


def apply_sberchel_cli_defaults(parser) -> None:
    """Point ``--device`` at CPU by default. Call AFTER ``add_app_launcher_args``.

    ``parse_env_cfg`` assigns ``cfg.sim.device`` unconditionally, after the env
    cfg's ``__post_init__`` has run, so the env cfg cannot set this -- the CLI
    default is the only place that sticks. Pass ``--device cuda:0`` to override.
    """
    parser.set_defaults(device=SBERCHEL_DEFAULT_DEVICE)
