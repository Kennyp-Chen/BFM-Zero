from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HT_URDF_REPO = REPO_ROOT.parent / "ht_urdf"
if HT_URDF_REPO.exists() and str(HT_URDF_REPO) not in sys.path:
    sys.path.insert(0, str(HT_URDF_REPO))

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as Rotation

from humanoidverse.utils.asset_paths import resolve_asset_path


DEFAULT_INPUT_DIR = Path("data_process/dataset/g1_lafan_dataset")
DEFAULT_OUTPUT_DIR = Path("humanoidverse/data")
DEFAULT_ROBOT_XML = Path("humanoidverse/data/robots/g1/g1_29dof.xml")
# PIPLUS_LSE_INPUT_DIR = Path("data_process/dataset/pi_LSE_lafan_dataset_20260629")
PIPLUS_LSE_INPUT_DIR = Path("data_process/dataset/pi_LSE_lafan_dataset_20260706_walk50")
PIPLUS_LSE_ROBOT_XML = "package://ht_urdf/PiPlus_S_12L8A0G2H1W_LSE_260611/xml/PiPlus_S_12L8A0G2H1W_LSE_260611.xml"
PIPLUS_H0W_INPUT_DIR = Path("data_process/dataset/pi_0W_lafan_dataset_20260714")
PIPLUS_H0W_ROBOT_XML = "package://ht_urdf/PiPlus_S_12L8A0G2H0W/xml/PiPlus_S_12L8A0G2H0W.xml"
PIPLUS_H0W_ROBOTS = {"piplus_h0w", "PiPlus_S_12L8A0G2H0W"}
PIPLUS_ROBOTS = {"piplus_lse"} | PIPLUS_H0W_ROBOTS
H1_260402_INPUT_DIR = Path("data_process/dataset/Hi_P_12L10A0G2H1W_260402_lafan_260714")
H1_260402_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "Hi_P_12L10A0G2H1W_260402_lafan"
H1_260402_ROBOT_URDF = (
    "package://ht_urdf/Hi_P_12L10A0G2H1W_260402/urdf/"
    "Hi_P_12L10A0G2H1W_Simplify_260402.urdf"
)
H1_260402_ROBOTS = {"h1_260402", "Hi_P_12L10A0G2H1W_260402"}
WXYZ_QUAT_ROBOTS = PIPLUS_ROBOTS | H1_260402_ROBOTS


def install_numpy_pickle_compat() -> None:
    try:
        importlib.import_module("numpy._core.multiarray")
        return
    except ModuleNotFoundError:
        pass

    import numpy.core as np_core

    sys.modules.setdefault("numpy._core", np_core)
    for module_name in ("multiarray", "_multiarray_umath", "numeric", "fromnumeric"):
        try:
            module = importlib.import_module(f"numpy.core.{module_name}")
        except ModuleNotFoundError:
            continue
        sys.modules.setdefault(f"numpy._core.{module_name}", module)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert GMR-retargeted LaFan pkl files into the HumanoidVerse "
            "lafan_29dof.pkl and lafan_29dof_10s-clipped.pkl formats."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_ROBOT_XML)
    parser.add_argument("--name", default="gmr_lafan")
    parser.add_argument(
        "--robot",
        choices=(
            "g1",
            "piplus_lse",
            "piplus_h0w",
            "PiPlus_S_12L8A0G2H0W",
            "h1_260402",
            "Hi_P_12L10A0G2H1W_260402",
        ),
        default="g1",
        help="Apply matching dataset and robot model defaults for g1, PiPlus, or H1 robots.",
    )
    parser.add_argument(
        "--quat-order",
        choices=("xyzw", "wxyz"),
        default=None,
        help="Quaternion component order in the source files. Defaults to wxyz for PiPlus robots, xyzw otherwise.",
    )
    parser.add_argument("--clip-seconds", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.robot == "piplus_lse":
        if args.input_dir == DEFAULT_INPUT_DIR:
            args.input_dir = PIPLUS_LSE_INPUT_DIR
        if args.robot_xml == DEFAULT_ROBOT_XML:
            args.robot_xml = resolve_asset_path("", PIPLUS_LSE_ROBOT_XML)
        if args.name == "gmr_lafan":
            args.name = "piplus_lse_lafan"
    elif args.robot in PIPLUS_H0W_ROBOTS:
        if args.input_dir == DEFAULT_INPUT_DIR:
            args.input_dir = PIPLUS_H0W_INPUT_DIR
        if args.robot_xml == DEFAULT_ROBOT_XML:
            args.robot_xml = resolve_asset_path("", PIPLUS_H0W_ROBOT_XML)
        if args.name == "gmr_lafan":
            args.name = "piplus_h0w_lafan"
    elif args.robot in H1_260402_ROBOTS:
        if args.input_dir == DEFAULT_INPUT_DIR:
            args.input_dir = H1_260402_INPUT_DIR
        if args.output_dir == DEFAULT_OUTPUT_DIR:
            args.output_dir = H1_260402_OUTPUT_DIR
        if args.robot_xml == DEFAULT_ROBOT_XML:
            args.robot_xml = H1_260402_ROBOT_URDF
        if args.name == "gmr_lafan":
            args.name = "h1_lafan"
    args.robot_xml = resolve_robot_model_path(args.robot_xml)
    if args.quat_order is None:
        args.quat_order = "wxyz" if args.robot in WXYZ_QUAT_ROBOTS else "xyzw"
    return args


def normalize_package_path(path: Path | str) -> str:
    path_text = str(path)
    if path_text.startswith("package:/") and not path_text.startswith("package://"):
        return path_text.replace("package:/", "package://", 1)
    return path_text


def resolve_robot_model_path(path: Path | str) -> Path:
    resolved = resolve_asset_path("", normalize_package_path(path))
    if not resolved.is_dir():
        return resolved

    robot_name = resolved.name
    for candidate in (
        resolved / "xml" / f"{robot_name}.xml",
        resolved / "urdf" / f"{robot_name}.urdf",
    ):
        if candidate.exists():
            return candidate

    candidates = sorted((resolved / "xml").glob("*.xml")) + sorted((resolved / "urdf").glob("*.urdf"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"{resolved} does not contain xml/*.xml or urdf/*.urdf robot model files.")


def load_mjcf_dof_metadata(root: ET.Element, robot_xml: Path) -> tuple[list[str], np.ndarray, list[str], dict[str, str]]:
    motor_joints = [
        motor.attrib.get("joint", motor.attrib.get("name"))
        for actuator in root.iter("actuator")
        for motor in actuator
    ]
    motor_joints = [name for name in motor_joints if name is not None]

    joint_axes = {}
    joint_order = []
    for joint in root.iter("joint"):
        if joint.attrib.get("type") == "free":
            continue
        axis = joint.attrib.get("axis")
        if axis is None:
            continue
        name = joint.attrib.get("name")
        joint_axes[name] = [float(value) for value in axis.split()]
        joint_order.append(name)

    motor_joints = [name for name in motor_joints if name in joint_axes]
    joint_names = motor_joints or joint_order
    axes = []
    for name in joint_names:
        if name not in joint_axes:
            raise ValueError(f"Actuated joint {name} has no one-DOF axis in {robot_xml}.")
        axes.append(joint_axes[name])

    if not axes:
        raise ValueError(f"No one-DOF actuated joints found in {robot_xml}.")

    body_names = []
    body_to_joint = {}
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{robot_xml}: missing MJCF worldbody.")

    def add_body(body: ET.Element) -> None:
        body_name = body.attrib.get("name")
        if body_name is None:
            return
        body_names.append(body_name)
        for joint in body.findall("joint"):
            joint_name = joint.attrib.get("name")
            if joint_name in joint_names:
                body_to_joint[body_name] = joint_name
        for child in body.findall("body"):
            add_body(child)

    for body in worldbody.findall("body"):
        add_body(body)

    return joint_names, np.asarray(axes, dtype=np.float32), body_names, body_to_joint


def load_urdf_dof_metadata(root: ET.Element, robot_urdf: Path) -> tuple[list[str], np.ndarray, list[str], dict[str, str]]:
    body_names = [link.attrib["name"] for link in root.findall("link") if "name" in link.attrib]
    if not body_names:
        raise ValueError(f"No links found in {robot_urdf}.")

    joint_names = []
    axes = []
    body_to_joint = {}
    for joint in root.findall("joint"):
        joint_type = joint.attrib.get("type")
        if joint_type in {"fixed", "floating", "planar"}:
            continue

        name = joint.attrib.get("name")
        if name is None:
            continue

        axis = joint.find("axis")
        axis_xyz = "1 0 0" if axis is None else axis.attrib.get("xyz", "1 0 0")
        child = joint.find("child")
        child_link = None if child is None else child.attrib.get("link")

        joint_names.append(name)
        axes.append([float(value) for value in axis_xyz.split()])
        if child_link is not None:
            body_to_joint[child_link] = name

    if not axes:
        raise ValueError(f"No movable one-DOF joints found in {robot_urdf}.")

    return joint_names, np.asarray(axes, dtype=np.float32), body_names, body_to_joint


def load_dof_metadata(robot_model: Path) -> tuple[list[str], np.ndarray, list[str], dict[str, str]]:
    root = ET.parse(robot_model).getroot()
    if root.tag == "mujoco":
        return load_mjcf_dof_metadata(root, robot_model)
    if root.tag == "robot":
        return load_urdf_dof_metadata(root, robot_model)
    raise ValueError(f"{robot_model}: unsupported robot model root tag {root.tag!r}.")


def make_body_aligned_pose_aa(
    root_axis_angle: np.ndarray,
    dof: np.ndarray,
    joint_names: list[str],
    dof_axes: np.ndarray,
    body_names: list[str],
    body_to_joint: dict[str, str],
) -> np.ndarray:
    joint_pose = {
        joint_name: dof_axes[joint_id][None, :] * dof[:, joint_id, None]
        for joint_id, joint_name in enumerate(joint_names)
    }
    pose_aa = np.zeros((dof.shape[0], len(body_names), 3), dtype=np.float32)
    pose_aa[:, 0, :] = root_axis_angle
    for body_id, body_name in enumerate(body_names[1:], start=1):
        joint_name = body_to_joint.get(body_name)
        if joint_name is not None:
            pose_aa[:, body_id, :] = joint_pose[joint_name]
    return pose_aa


def normalize_quat_xyzw(quat: np.ndarray, quat_order: str) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    if quat_order == "wxyz":
        quat = quat[..., [1, 2, 3, 0]]
    elif quat_order != "xyzw":
        raise ValueError(f"Unsupported quaternion order: {quat_order}.")
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norm <= 0.0):
        raise ValueError("root_rot contains zero-length quaternion(s).")
    return quat / norm


def get_raw_field(raw: dict, *names: str, source: Path):
    for name in names:
        if name in raw:
            return raw[name]
    raise KeyError(f"{source} is missing required field. Tried aliases: {names}")


def convert_motion(
    raw: dict,
    joint_names: list[str],
    dof_axes: np.ndarray,
    body_names: list[str],
    body_to_joint: dict[str, str],
    source: Path,
    quat_order: str,
    robot: str,
) -> dict:
    required = (
        ("fps", "framerate"),
        ("root_pos", "base_pos_w"),
        ("root_rot", "base_quat_w"),
        ("dof_pos", "joint_pos"),
    )
    missing = [aliases for aliases in required if not any(alias in raw for alias in aliases)]
    if missing:
        raise KeyError(f"{source} is missing required field alias group(s): {missing}")

    root_pos = np.asarray(get_raw_field(raw, "root_pos", "base_pos_w", source=source), dtype=np.float32)
    root_rot = normalize_quat_xyzw(get_raw_field(raw, "root_rot", "base_quat_w", source=source), quat_order)
    dof = np.asarray(get_raw_field(raw, "dof_pos", "joint_pos", source=source), dtype=np.float32)
    fps = int(get_raw_field(raw, "fps", "framerate", source=source))
    if robot in PIPLUS_ROBOTS and source.stem.lower().startswith("walk"):
        fps = 50

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"{source}: root_pos must have shape (T, 3), got {root_pos.shape}.")
    if root_rot.ndim != 2 or root_rot.shape[1] != 4:
        raise ValueError(f"{source}: root_rot must have shape (T, 4), got {root_rot.shape}.")
    if dof.ndim != 2 or dof.shape[1] != len(joint_names):
        raise ValueError(f"{source}: dof_pos must have shape (T, {len(joint_names)}), got {dof.shape}.")
    if "joint_names" in raw and list(raw["joint_names"]) != joint_names:
        raise ValueError(
            f"{source}: joint_names do not match robot XML motor order.\n"
            f"raw: {raw['joint_names']}\nxml: {joint_names}"
        )
    if not (root_pos.shape[0] == root_rot.shape[0] == dof.shape[0]):
        raise ValueError(
            f"{source}: root_pos, root_rot, and dof_pos frame counts do not match: "
            f"{root_pos.shape[0]}, {root_rot.shape[0]}, {dof.shape[0]}."
        )
    if fps <= 0:
        raise ValueError(f"{source}: fps must be positive, got {fps}.")

    root_axis_angle = Rotation.from_quat(root_rot).as_rotvec().astype(np.float32)
    pose_aa = make_body_aligned_pose_aa(root_axis_angle, dof, joint_names, dof_axes, body_names, body_to_joint)

    return {
        "root_trans_offset": root_pos,
        "pose_aa": pose_aa,
        "dof": dof,
        "root_rot": root_rot,
        "smpl_joints": np.zeros((root_pos.shape[0], 24, 3), dtype=np.float32),
        "fps": fps,
        "joint_names": joint_names,
    }


def load_dataset(
    input_dir: Path,
    joint_names: list[str],
    dof_axes: np.ndarray,
    body_names: list[str],
    body_to_joint: dict[str, str],
    quat_order: str,
    robot: str,
) -> dict[str, dict]:
    files = sorted(input_dir.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No .pkl files found in {input_dir}.")

    dataset = {}
    install_numpy_pickle_compat()
    for path in files:
        dataset[path.stem] = convert_motion(
            joblib.load(path),
            joint_names,
            dof_axes,
            body_names,
            body_to_joint,
            path,
            quat_order,
            robot,
        )
    return dataset


def make_clips(dataset: dict[str, dict], clip_seconds: float) -> dict[str, dict]:
    if clip_seconds <= 0.0:
        raise ValueError(f"clip-seconds must be positive, got {clip_seconds}.")

    clips = {}
    for motion_name, motion in dataset.items():
        fps = int(motion["fps"])
        clip_frames = int(round(clip_seconds * fps))
        if clip_frames <= 0:
            raise ValueError(f"{motion_name}: clip length rounded to {clip_frames} frames.")

        num_clips = motion["dof"].shape[0] // clip_frames
        for clip_id in range(num_clips):
            start = clip_id * clip_frames
            end = start + clip_frames
            clip = {
                key: value[start:end].copy() if isinstance(value, np.ndarray) else value
                for key, value in motion.items()
            }
            clip["motion_name"] = motion_name
            clips[f"{motion_name}_clip{clip_id}"] = clip
    return clips


def dump_dataset(data: dict[str, dict], path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(data, path)


def main() -> None:
    args = parse_args()
    output_full = args.output_dir / f"{args.name}.pkl"
    output_clips = args.output_dir / f"{args.name}_10s-clipped.pkl"

    joint_names, dof_axes, body_names, body_to_joint = load_dof_metadata(args.robot_xml)
    dataset = load_dataset(args.input_dir, joint_names, dof_axes, body_names, body_to_joint, args.quat_order, args.robot)
    clips = make_clips(dataset, args.clip_seconds)

    dump_dataset(dataset, output_full, args.overwrite)
    dump_dataset(clips, output_clips, args.overwrite)

    total_frames = sum(motion["dof"].shape[0] for motion in dataset.values())
    clip_frames = sum(motion["dof"].shape[0] for motion in clips.values())
    print(f"Saved {len(dataset)} motions / {total_frames} frames to {output_full}")
    print(f"Saved {len(clips)} clips / {clip_frames} frames to {output_clips}")


if __name__ == "__main__":
    main()
