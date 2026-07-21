"""Build dex-retargeting hand URDFs from the locally cached GR1T2 robot URDF."""

from __future__ import annotations

import copy
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

_RIGHT_PALM_LINK = "GR1T2_fourier_hand_6dof_right_hand_pitch_link"
_LEFT_PALM_LINK = "GR1T2_fourier_hand_6dof_left_hand_pitch_link"
_RIGHT_WRIST_LINK = "r_hand_base_link"
_LEFT_WRIST_LINK = "l_hand_base_link"

# Matches the wrist frame used by Isaac Lab's GR1T2 dex-retargeting assets.
_RIGHT_FIXED_ORIGIN = {"xyz": "0.0 0.0 0.0", "rpy": "3.14159 0.0 1.57079"}
_LEFT_FIXED_ORIGIN = {"xyz": "0.0 0.0 0.0", "rpy": "0.0 0.0 1.57079"}


def _link_tag_name(link_elem: ET.Element) -> str:
    return link_elem.attrib["name"]


def _joint_tag_name(joint_elem: ET.Element) -> str:
    return joint_elem.attrib["name"]


def _get_child_link(joint_elem: ET.Element) -> str:
    return joint_elem.find("child").attrib["link"]


def _get_parent_link(joint_elem: ET.Element) -> str:
    return joint_elem.find("parent").attrib["link"]


def _set_parent_link(joint_elem: ET.Element, parent_link: str) -> None:
    joint_elem.find("parent").attrib["link"] = parent_link


def _kinematic_link(link_elem: ET.Element) -> ET.Element:
    """Keep only inertial data; dex-retargeting does not need meshes."""
    link = ET.Element("link", name=_link_tag_name(link_elem))
    inertial = link_elem.find("inertial")
    if inertial is not None:
        link.append(copy.deepcopy(inertial))
    return link


def _fixed_joint(parent: str, child: str, origin: dict[str, str]) -> ET.Element:
    joint = ET.Element("joint", name="FixedJoint", type="fixed")
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=child)
    ET.SubElement(joint, "origin", **origin)
    ET.SubElement(joint, "axis", xyz="1.0 0.0 0.0")
    return joint


def _collect_hand_elements(
    robot_root: ET.Element,
    joint_prefix: str,
    palm_link: str,
    wrist_link: str,
) -> tuple[list[ET.Element], list[ET.Element]]:
    links_by_name = {_link_tag_name(link): link for link in robot_root.findall("link")}
    hand_links: dict[str, ET.Element] = {}
    hand_joints: list[ET.Element] = []

    for joint in robot_root.findall("joint"):
        joint_name = _joint_tag_name(joint)
        if not joint_name.startswith(joint_prefix):
            continue

        parent = _get_parent_link(joint)
        child = _get_child_link(joint)
        if parent == palm_link:
            parent = wrist_link

        for link_name in (parent, child):
            if link_name in (wrist_link, "base_link"):
                continue
            if link_name in links_by_name:
                hand_links[link_name] = links_by_name[link_name]

        joint_copy = copy.deepcopy(joint)
        if _get_parent_link(joint_copy) == palm_link:
            _set_parent_link(joint_copy, wrist_link)
        hand_joints.append(joint_copy)

    return list(hand_links.values()), hand_joints


def _build_hand_urdf(
    robot_root: ET.Element,
    joint_prefix: str,
    palm_link: str,
    wrist_link: str,
    fixed_origin: dict[str, str],
) -> ET.ElementTree:
    hand_links, hand_joints = _collect_hand_elements(robot_root, joint_prefix, palm_link, wrist_link)

    robot = ET.Element("robot", name=f"GR1T2_{joint_prefix.strip('_').lower()}_hand")
    ET.SubElement(robot, "link", name="base_link")
    ET.SubElement(robot, "link", name=wrist_link)
    robot.append(_fixed_joint("base_link", wrist_link, fixed_origin))

    for link_elem in hand_links:
        robot.append(_kinematic_link(link_elem))
    for joint_elem in hand_joints:
        robot.append(joint_elem)

    return ET.ElementTree(robot)


def _write_urdf(tree: ET.ElementTree, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="unicode", xml_declaration=True)
    return str(output_path)


def prepare_gr1t2_dex_hand_urdfs(full_robot_urdf_path: str | Path, cache_dir: str | Path) -> tuple[str, str]:
    """Create left/right hand URDFs for dex-retargeting from the cached full robot URDF."""
    full_robot_urdf_path = Path(full_robot_urdf_path)
    cache_dir = Path(cache_dir)
    dex_dir = cache_dir / "dex_retargeting"
    right_path = dex_dir / "GR1_T2_right_hand.urdf"
    left_path = dex_dir / "GR1_T2_left_hand.urdf"

    raw_urdf = full_robot_urdf_path.parent / full_robot_urdf_path.name.replace("_pink.urdf", ".urdf")
    if not raw_urdf.exists():
        raw_urdf = full_robot_urdf_path

    if right_path.exists() and left_path.exists() and right_path.stat().st_mtime >= raw_urdf.stat().st_mtime:
        return str(left_path), str(right_path)

    robot_root = ET.parse(raw_urdf).getroot()
    right_tree = _build_hand_urdf(robot_root, "R_", _RIGHT_PALM_LINK, _RIGHT_WRIST_LINK, _RIGHT_FIXED_ORIGIN)
    left_tree = _build_hand_urdf(robot_root, "L_", _LEFT_PALM_LINK, _LEFT_WRIST_LINK, _LEFT_FIXED_ORIGIN)
    return _write_urdf(left_tree, left_path), _write_urdf(right_tree, right_path)


def prepare_gr1t2_dex_retargeting_configs(cache_dir: str | Path) -> tuple[str, str]:
    """Copy dex-retargeting YAML configs into the cache and return their paths."""
    import isaaclab.devices.openxr.retargeters.humanoid.fourier.gr1_t2_dex_retargeting_utils as dex_utils

    cache_dir = Path(cache_dir)
    src_dir = Path(dex_utils.__file__).resolve().parent / "data/configs/dex-retargeting"
    dst_dir = cache_dir / "dex_retargeting"
    dst_dir.mkdir(parents=True, exist_ok=True)

    left_cfg = dst_dir / "fourier_hand_left_dexpilot.yml"
    right_cfg = dst_dir / "fourier_hand_right_dexpilot.yml"
    shutil.copy2(src_dir / "fourier_hand_left_dexpilot.yml", left_cfg)
    shutil.copy2(src_dir / "fourier_hand_right_dexpilot.yml", right_cfg)
    return str(left_cfg), str(right_cfg)
