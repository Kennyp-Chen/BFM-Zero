from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import joblib
import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HT_URDF_REPO = REPO_ROOT.parent / "ht_urdf"
if HT_URDF_REPO.exists() and str(HT_URDF_REPO) not in sys.path:
    sys.path.insert(0, str(HT_URDF_REPO))

from data_process.convert_gmr_lafan import make_body_aligned_pose_aa
from humanoidverse.utils.asset_paths import resolve_asset_path


PIPLUS_LSE_DATASET_DIR = REPO_ROOT / "data_process/dataset/pi_LSE_lafan_dataset_20260623"
PIPLUS_LSE_REFERENCE = PIPLUS_LSE_DATASET_DIR / "PiPlus_S_12L8A0G2H1W_LSE_260424_run2_subject4_20260617.pkl"
PIPLUS_LSE_OUTPUT = PIPLUS_LSE_DATASET_DIR / "piplus_lse_joint_sweep_gmr.pkl"
PIPLUS_LSE_STAND_OUTPUT = PIPLUS_LSE_DATASET_DIR / "piplus_lse_stand_gmr.pkl"
PIPLUS_LSE_MOTION_LIB_OUTPUT = REPO_ROOT / "humanoidverse/data/pi_LSE_joint_sweep/piplus_lse_joint_sweep.pkl"
PIPLUS_LSE_MOTION_LIB_STAND_OUTPUT = REPO_ROOT / "humanoidverse/data/pi_LSE_joint_sweep/piplus_lse_stand.pkl"
PIPLUS_LSE_ROBOT_XML = (
    "package://ht_urdf/PiPlus_S_12L8A0G2H1W_LSE_260611/"
    "xml/PiPlus_S_12L8A0G2H1W_LSE_260611.xml"
)

PIPLUS_H0W_DATASET_DIR = REPO_ROOT / "data_process/dataset/piplus_lafan_260629"
PIPLUS_H0W_REFERENCE = PIPLUS_H0W_DATASET_DIR / "PiPlus_S_12L8A0G2H0W_run2_subject4_soma_20260629.pkl"
PIPLUS_H0W_OUTPUT = PIPLUS_H0W_DATASET_DIR / "piplus_h0w_joint_sweep_gmr.pkl"
PIPLUS_H0W_STAND_OUTPUT = PIPLUS_H0W_DATASET_DIR / "piplus_h0w_stand_gmr.pkl"
PIPLUS_H0W_MOTION_LIB_OUTPUT = REPO_ROOT / "humanoidverse/data/PiPlus_S_12L8A0G2H0W_generated/piplus_h0w_joint_sweep.pkl"
PIPLUS_H0W_MOTION_LIB_STAND_OUTPUT = REPO_ROOT / "humanoidverse/data/PiPlus_S_12L8A0G2H0W_generated/piplus_h0w_stand.pkl"
PIPLUS_H0W_ROBOT_XML = "package://ht_urdf/PiPlus_S_12L8A0G2H0W/xml/PiPlus_S_12L8A0G2H0W.xml"
PIPLUS_H0W_ROBOTS = {"piplus_h0w", "PiPlus_S_12L8A0G2H0W"}

DEFAULT_DATASET_DIR = PIPLUS_LSE_DATASET_DIR
DEFAULT_REFERENCE = PIPLUS_LSE_REFERENCE
DEFAULT_OUTPUT = PIPLUS_LSE_OUTPUT
DEFAULT_STAND_OUTPUT = PIPLUS_LSE_STAND_OUTPUT
DEFAULT_ROBOT_XML = PIPLUS_LSE_ROBOT_XML

ROOT_ROT_LYING_XYZW = Rotation.from_euler("y", -90.0, degrees=True).as_quat().astype(np.float32)

JOINT_SWEEP_ORDER = [
    "head_pitch_joint",
    "head_yaw_joint",
    "l_shoulder_pitch_joint",
    "l_shoulder_roll_joint",
    "l_upper_arm_joint",
    "l_elbow_joint",
    "r_shoulder_pitch_joint",
    "r_shoulder_roll_joint",
    "r_upper_arm_joint",
    "r_elbow_joint",
    "l_hip_pitch_joint",
    "l_hip_roll_joint",
    "l_thigh_joint",
    "l_calf_joint",
    "l_ankle_pitch_joint",
    "l_ankle_roll_joint",
    "r_hip_pitch_joint",
    "r_hip_roll_joint",
    "r_thigh_joint",
    "r_calf_joint",
    "r_ankle_pitch_joint",
    "r_ankle_roll_joint",
    "waist_yaw_joint",
]

JOINT_SIGNS = {
    # With the root lying on its back, local +X points upward. Pitch joints use
    # signs that move the limb toward local +X; roll signs move left/right limbs
    # away from the body midline.
    "head_pitch_joint": 1.0,
    "head_yaw_joint": 1.0,
    "l_shoulder_pitch_joint": -1.0,
    "l_shoulder_roll_joint": 1.0,
    "l_upper_arm_joint": 1.0,
    "l_elbow_joint": -1.0,
    "r_shoulder_pitch_joint": -1.0,
    "r_shoulder_roll_joint": -1.0,
    "r_upper_arm_joint": -1.0,
    "r_elbow_joint": -1.0,
    "l_hip_pitch_joint": -1.0,
    "l_hip_roll_joint": 1.0,
    "l_thigh_joint": 1.0,
    "l_calf_joint": -1.0,
    "l_ankle_pitch_joint": 1.0,
    "l_ankle_roll_joint": 1.0,
    "r_hip_pitch_joint": -1.0,
    "r_hip_roll_joint": -1.0,
    "r_thigh_joint": -1.0,
    "r_calf_joint": -1.0,
    "r_ankle_pitch_joint": 1.0,
    "r_ankle_roll_joint": -1.0,
    "waist_yaw_joint": 1.0,
}

STAND_JOINT_SIGN_OVERRIDES = {
    "l_calf_joint": 1.0,
    "r_calf_joint": 1.0,
}

OPTIONAL_SWEEP_JOINTS = {"waist_yaw_joint"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a PiPlus GMR-style pkl with one joint sweeping at a time, "
            "matching the raw PiPlus retargeted pkl schema."
        )
    )
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--robot-xml", default=DEFAULT_ROBOT_XML)
    parser.add_argument(
        "--output-format",
        choices=("gmr", "motion-lib"),
        default="gmr",
        help="Use gmr for raw base_pos_w/joint_pos schema, or motion-lib for tracking_inference.",
    )
    parser.add_argument(
        "--motion-name",
        default=None,
        help="Motion key inside motion-lib output. Defaults to the output file stem.",
    )
    parser.add_argument(
        "--robot",
        choices=(
            "piplus_lse",
            "PiPlus_S_12L8A0G2H1W_LSE",
            "piplus_h0w",
            "PiPlus_S_12L8A0G2H0W",
        ),
        default="piplus_lse",
        help="Apply matching default reference, robot XML, and output paths.",
    )
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--amplitude", type=float, default=0.8)
    parser.add_argument("--frames-per-joint", type=int, default=60)
    parser.add_argument("--rest-frames", type=int, default=10)
    parser.add_argument("--root-height", type=float, default=0.18, help="Root height for the lying joint-sweep motion.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--limit-margin",
        type=float,
        default=1e-3,
        help="Safety margin in radians when clipping joint targets to MJCF limits.",
    )
    parser.add_argument(
        "--stand",
        action="store_true",
        help="Generate the one-joint-at-a-time sweep using the MJCF default standing free-body pose.",
    )
    parser.add_argument(
        "--clip-to-limits",
        action="store_true",
        default=True,
        help="Clip the requested joint amplitude to MJCF joint ranges. Enabled by default.",
    )
    parser.add_argument(
        "--no-clip-to-limits",
        dest="clip_to_limits",
        action="store_false",
        help="Disable clipping joint targets to MJCF joint ranges.",
    )
    args = parser.parse_args()
    if args.robot in PIPLUS_H0W_ROBOTS:
        if args.reference == DEFAULT_REFERENCE:
            args.reference = PIPLUS_H0W_REFERENCE
        if args.robot_xml == DEFAULT_ROBOT_XML:
            args.robot_xml = PIPLUS_H0W_ROBOT_XML
        if args.output is None:
            if args.output_format == "motion-lib":
                args.output = PIPLUS_H0W_MOTION_LIB_STAND_OUTPUT if args.stand else PIPLUS_H0W_MOTION_LIB_OUTPUT
            else:
                args.output = PIPLUS_H0W_STAND_OUTPUT if args.stand else PIPLUS_H0W_OUTPUT
    elif args.output is None:
        if args.output_format == "motion-lib":
            args.output = PIPLUS_LSE_MOTION_LIB_STAND_OUTPUT if args.stand else PIPLUS_LSE_MOTION_LIB_OUTPUT
        else:
            args.output = DEFAULT_STAND_OUTPUT if args.stand else DEFAULT_OUTPUT
    return args


def resolve_robot_xml(path: str | Path) -> Path:
    return resolve_asset_path("", str(path))


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


def load_reference_metadata(path: Path) -> tuple[list[str], int]:
    install_numpy_pickle_compat()
    data = joblib.load(path)
    if "joint_names" in data:
        return list(data["joint_names"]), int(data["framerate"])
    motion = next(iter(data.values()))
    return list(motion["joint_names"]), int(motion["fps"])


def load_mjcf_metadata(
    robot_xml: Path,
    joint_names: list[str],
) -> tuple[np.ndarray, list[str], dict[str, str], dict[str, tuple[float, float]], np.ndarray, np.ndarray]:
    root = ET.parse(robot_xml).getroot()
    joint_axes: dict[str, list[float]] = {}
    joint_ranges: dict[str, tuple[float, float]] = {}
    for joint in root.iter("joint"):
        name = joint.attrib.get("name")
        if name is None or joint.attrib.get("type") == "free":
            continue
        axis = joint.attrib.get("axis")
        if axis is not None:
            joint_axes[name] = [float(value) for value in axis.split()]
        joint_range = joint.attrib.get("range")
        if joint_range is not None:
            lo, hi = (float(value) for value in joint_range.split())
            joint_ranges[name] = (lo, hi)

    missing_axes = [name for name in joint_names if name not in joint_axes]
    if missing_axes:
        raise ValueError(f"{robot_xml} is missing axes for joints: {missing_axes}")

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{robot_xml}: missing MJCF worldbody.")

    body_names: list[str] = []
    body_to_joint: dict[str, str] = {}

    free_body = None
    for body in worldbody.iter("body"):
        if any(joint.attrib.get("type") == "free" for joint in body.findall("joint")):
            free_body = body
            break
    if free_body is None:
        raise ValueError(f"{robot_xml}: missing a body with a free joint.")

    free_body_pos = np.asarray(
        [float(value) for value in free_body.attrib.get("pos", "0 0 0").split()],
        dtype=np.float32,
    )
    if free_body_pos.shape != (3,):
        raise ValueError(f"{robot_xml}: free-body pos must have 3 values, got {free_body_pos}.")

    free_body_quat_wxyz = np.asarray(
        [float(value) for value in free_body.attrib.get("quat", "1 0 0 0").split()],
        dtype=np.float32,
    )
    if free_body_quat_wxyz.shape != (4,):
        raise ValueError(f"{robot_xml}: free-body quat must have 4 values, got {free_body_quat_wxyz}.")
    quat_norm = np.linalg.norm(free_body_quat_wxyz)
    if quat_norm <= 0.0:
        raise ValueError(f"{robot_xml}: free-body quat has zero norm.")
    free_body_quat_xyzw = (free_body_quat_wxyz / quat_norm)[[1, 2, 3, 0]].astype(np.float32)

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

    dof_axes = np.asarray([joint_axes[name] for name in joint_names], dtype=np.float32)
    return dof_axes, body_names, body_to_joint, joint_ranges, free_body_pos, free_body_quat_xyzw


def smooth_profile(frames: int, rest_frames: int) -> np.ndarray:
    if frames < 4:
        raise ValueError(f"frames-per-joint must be at least 4, got {frames}.")
    if rest_frames < 0:
        raise ValueError(f"rest-frames must be non-negative, got {rest_frames}.")

    active = np.sin(np.linspace(0.0, np.pi, frames, dtype=np.float32))
    active = active / np.max(active)
    if rest_frames == 0:
        return active
    rest = np.zeros(rest_frames, dtype=np.float32)
    return np.concatenate([rest, active, rest])


def make_dof(
    joint_names: list[str],
    joint_ranges: dict[str, tuple[float, float]],
    amplitude: float,
    frames_per_joint: int,
    rest_frames: int,
    clip_to_limits: bool,
    limit_margin: float,
    stand: bool,
) -> tuple[np.ndarray, list[tuple[str, float]]]:
    if limit_margin < 0.0:
        raise ValueError(f"limit-margin must be non-negative, got {limit_margin}.")

    joint_to_index = {name: idx for idx, name in enumerate(joint_names)}
    missing_required = [
        name
        for name in JOINT_SWEEP_ORDER
        if name not in joint_to_index and name not in OPTIONAL_SWEEP_JOINTS
    ]
    if missing_required:
        raise ValueError(f"Joint sweep order contains unknown joints: {missing_required}")
    sweep_order = [name for name in JOINT_SWEEP_ORDER if name in joint_to_index]
    if not sweep_order:
        raise ValueError("Joint sweep order does not contain any joints from the reference metadata.")

    profile = smooth_profile(frames_per_joint, rest_frames)
    total_frames = len(profile) * len(sweep_order)
    dof = np.zeros((total_frames, len(joint_names)), dtype=np.float32)
    applied: list[tuple[str, float]] = []
    joint_signs = JOINT_SIGNS | STAND_JOINT_SIGN_OVERRIDES if stand else JOINT_SIGNS

    for sweep_id, joint_name in enumerate(sweep_order):
        target = float(amplitude) * joint_signs[joint_name]
        if clip_to_limits and joint_name in joint_ranges:
            lo, hi = joint_ranges[joint_name]
            safe_lo = lo + limit_margin
            safe_hi = hi - limit_margin
            if safe_lo > safe_hi:
                raise ValueError(
                    f"{joint_name} range [{lo}, {hi}] is narrower than "
                    f"2 * limit-margin ({2.0 * limit_margin})."
                )
            target = float(np.clip(target, safe_lo, safe_hi))
        start = sweep_id * len(profile)
        end = start + len(profile)
        dof[start:end, joint_to_index[joint_name]] = profile * target
        applied.append((joint_name, target))

    return dof, applied


def xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    return quat[..., [3, 0, 1, 2]]


def make_motion(args: argparse.Namespace) -> tuple[dict, list[tuple[str, float]]]:
    joint_names, ref_fps = load_reference_metadata(args.reference)
    fps = ref_fps if args.fps is None else args.fps
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}.")

    robot_xml = resolve_robot_xml(args.robot_xml)
    dof_axes, body_names, body_to_joint, joint_ranges, stand_root_pos, stand_root_quat_xyzw = load_mjcf_metadata(
        robot_xml,
        joint_names,
    )
    dof, applied = make_dof(
        joint_names,
        joint_ranges,
        args.amplitude,
        args.frames_per_joint,
        args.rest_frames,
        args.clip_to_limits,
        args.limit_margin,
        args.stand,
    )

    num_frames = dof.shape[0]
    if args.stand:
        root_trans_offset = np.repeat(stand_root_pos[None, :], num_frames, axis=0)
        root_rot = np.repeat(stand_root_quat_xyzw[None, :], num_frames, axis=0)
    else:
        root_trans_offset = np.zeros((num_frames, 3), dtype=np.float32)
        root_trans_offset[:, 2] = np.float32(args.root_height)
        root_rot = np.repeat(ROOT_ROT_LYING_XYZW[None, :], num_frames, axis=0)

    if args.output_format == "motion-lib":
        root_axis_angle = Rotation.from_quat(root_rot).as_rotvec().astype(np.float32)
        pose_aa = make_body_aligned_pose_aa(
            root_axis_angle,
            dof,
            joint_names,
            dof_axes,
            body_names,
            body_to_joint,
        )
        motion = {
            "root_trans_offset": root_trans_offset.astype(np.float32),
            "pose_aa": pose_aa,
            "dof": dof.astype(np.float32),
            "root_rot": root_rot.astype(np.float32),
            "smpl_joints": np.zeros((num_frames, 24, 3), dtype=np.float32),
            "fps": int(fps),
            "joint_names": joint_names,
        }
        return motion, applied

    motion = {
        "framerate": int(fps),
        "base_pos_w": root_trans_offset.astype(np.float64),
        "base_quat_w": xyzw_to_wxyz(root_rot).astype(np.float64),
        "joint_pos": dof.astype(np.float64),
        "joint_names": joint_names,
        "local_body_pos": None,
        "link_body_list": None,
    }
    return motion, applied


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} already exists. Pass --overwrite to replace it.")

    motion, applied = make_motion(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output_format == "motion-lib":
        motion_name = args.motion_name or args.output.stem
        joblib.dump({motion_name: motion}, args.output)
        print(f"Saved motion-lib motion {motion_name} to {args.output}")
        print(f"frames={motion['dof'].shape[0]}, fps={motion['fps']}, dof={motion['dof'].shape[1]}")
    else:
        joblib.dump(motion, args.output)
        print(f"Saved GMR-style motion to {args.output}")
        print(f"frames={motion['joint_pos'].shape[0]}, fps={motion['framerate']}, dof={motion['joint_pos'].shape[1]}")

    if args.stand:
        if args.output_format == "motion-lib":
            print(
                "standing root pose: "
                f"root={motion['root_trans_offset'][0].tolist()}, "
                f"root_rot_xyzw={motion['root_rot'][0].tolist()}"
            )
        else:
            print(
                "standing root pose: "
                f"root={motion['base_pos_w'][0].tolist()}, "
                f"base_quat_wxyz={motion['base_quat_w'][0].tolist()}"
            )
    print("joint targets:")
    for joint_name, target in applied:
        print(f"  {joint_name}: {target:+.3f}")


if __name__ == "__main__":
    main()
