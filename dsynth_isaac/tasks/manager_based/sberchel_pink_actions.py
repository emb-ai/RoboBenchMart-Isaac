"""Pink IK action term for the closed-kinematics Sberchel wrist.

On green_challenge's ``robot_v3_0a`` MJCF the wrist pitch is a four-bar linkage:
``*_wrist_crank_joint`` is the motor dof and ``*_wrist_pitch_joint`` is its
passive output, connected through ``*_wrist_rod_link`` by an equality
constraint. Only the pitch joint lies on the kinematic chain to the hand, so
Pink must *solve* for it; only the crank joint has a drive, so the solution
must be *written* there.

The linkage is a parallelogram -- measured off the MJCF, crank arm 0.0210134 vs
output arm 0.0210002, coupler 0.0712507 vs ground link 0.0712507, with the
crank arm parallel to the output arm at the reference pose -- so the output
rotates 1:1 with the input, same sign, and the solved pitch angle can be
written to the crank unchanged. Both joints also carry the same +-0.785 rad
range, so no reachability is lost.

Even if that ratio were off, the loop would still converge: Pink is a
differential IK that re-reads the *measured* pitch every step and integrates
until the hand-frame error vanishes, so a transmission error changes the
required crank angle, not the hand pose it settles at.
"""

from __future__ import annotations

from isaaclab.envs.mdp.actions import pink_task_space_actions
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass


class SberchelPinkIKAction(pink_task_space_actions.PinkInverseKinematicsAction):
    """Pink IK that writes each solved joint target to a different drive dof.

    ``cfg.write_joint_remap`` maps a *solved* joint name to the *driven* joint
    name that stands behind it. Everything the solver sees is untouched:
    ``_isaaclab_controlled_joint_ids`` and ``cfg.controller.joint_names`` keep
    the solved joints, which ``PinkIKController._validate_consistency``
    cross-checks against each other. Only the write targets move, so
    ``apply_actions``, the gravity-compensation path and any replay that reuses
    ``_controlled_joint_ids`` all follow automatically.
    """

    cfg: SberchelPinkInverseKinematicsActionCfg

    def _initialize_joint_info(self) -> None:
        super()._initialize_joint_info()
        self._check_hand_joint_order()

        remap = dict(self.cfg.write_joint_remap or {})
        if not remap:
            return

        all_joint_names = list(self._asset.data.joint_names)
        write_ids = list(self._controlled_joint_ids)
        write_names = list(self._controlled_joint_names)
        remapped: list[str] = []
        for slot, name in enumerate(write_names):
            target = remap.get(name)
            if target is None:
                continue
            if target not in all_joint_names:
                raise ValueError(
                    f"[SberchelPinkIKAction] write_joint_remap sends {name!r} to {target!r}, "
                    f"which is not a joint of asset {self.cfg.asset_name!r}. "
                    "Is the robot still spawned from the open-kinematics MJCF?"
                )
            write_ids[slot] = all_joint_names.index(target)
            write_names[slot] = target
            remapped.append(f"{name}->{target}")

        missing = sorted(set(remap) - set(self._controlled_joint_names))
        if missing:
            raise ValueError(
                f"[SberchelPinkIKAction] write_joint_remap keys {missing} are not Pink-controlled "
                "joints; nothing would be remapped for them."
            )

        self._controlled_joint_ids = write_ids
        self._controlled_joint_names = write_names
        print(f">>> Pink IK write remap: {', '.join(remapped)}", flush=True)

    def _check_hand_joint_order(self) -> None:
        """Fail loudly if ``cfg.hand_joint_names`` is not in articulation order.

        ``Articulation.find_joints`` returns ids in articulation dof order, not
        in the order the cfg lists them, but the retargeter emits its 12 finger
        values in *cfg* order. If the two disagree the mismatch is silent --
        both vectors are length 12 -- and thumb commands land on the pinky.
        Swapping the robot model reorders dofs, so this is checked at startup.
        """
        expected = list(self.cfg.hand_joint_names)
        actual = list(self._hand_joint_names)
        if expected == actual:
            return
        raise ValueError(
            "[SberchelPinkIKAction] hand_joint_names is not in articulation dof order, so "
            "retargeted finger values would be applied to the wrong joints.\n"
            f"  cfg order   : {expected}\n"
            f"  articulation: {actual}\n"
            "Update SBERCHEL_HAND_JOINT_NAMES in sberchel_upperbody_xr_env_cfg.py to the "
            "articulation order above."
        )


@configclass
class SberchelPinkInverseKinematicsActionCfg(PinkInverseKinematicsActionCfg):
    """Pink IK action cfg with a solved-joint -> driven-joint write remap."""

    class_type: type[ActionTerm] = SberchelPinkIKAction

    write_joint_remap: dict[str, str] = {}
    """Maps a Pink-controlled joint to the dof its target is written to instead.

    Used for the wrist four-bar: Pink solves ``*_wrist_pitch_joint`` (on the
    chain to the hand) and the target is applied to ``*_wrist_crank_joint``
    (the dof with a drive). Empty means no remap, i.e. stock behaviour.
    """
