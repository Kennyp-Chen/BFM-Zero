from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np

from humanoidverse.utils.asset_paths import resolve_asset_path


RobotName = Literal["g1", "piplus_lse", "piplus_h0w", "h1_260402", "Hi_P_12L10A0G2H1W_260402"]
HUMANOIDVERSE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RobotSpec:
    name: RobotName
    default_data_path: Path
    mujoco_xml_path: Path | str
    dof_size: int
    hydra_robot: str


ROBOT_SPECS: dict[RobotName, RobotSpec] = {
    "g1": RobotSpec(
        name="g1",
        default_data_path=HUMANOIDVERSE_DIR / "data" / "lafan_29dof.pkl",
        mujoco_xml_path=HUMANOIDVERSE_DIR / "data" / "robots" / "g1" / "scene_29dof_freebase_mujoco.xml",
        dof_size=29,
        hydra_robot="g1/g1_29dof",
    ),
    "piplus_lse": RobotSpec(
        name="piplus_lse",
        default_data_path=HUMANOIDVERSE_DIR / "data" / "piplus_lse_lafan.pkl",
        mujoco_xml_path="package://ht_urdf/PiPlus_S_12L8A0G2H1W_LSE_260611/xml/PiPlus_S_12L8A0G2H1W_LSE_260611.xml",
        dof_size=23,
        hydra_robot="piplus/PiPlus_S_12L8A0G2H1W_LSE",
    ),
    "piplus_h0w": RobotSpec(
        name="piplus_h0w",
        default_data_path=HUMANOIDVERSE_DIR / "data" / "piplus_h0w_lafan.pkl",
        mujoco_xml_path="package://ht_urdf/PiPlus_S_12L8A0G2H0W/xml/PiPlus_S_12L8A0G2H0W.xml",
        dof_size=22,
        hydra_robot="piplus/PiPlus_S_12L8A0G2H0W",
    ),
    "h1_260402": RobotSpec(
        name="h1_260402",
        default_data_path=HUMANOIDVERSE_DIR / "data" / "Hi_P_12L10A0G2H1W_260402_lafandataset_260714" / "h1_lafan.pkl",
        mujoco_xml_path="package://ht_urdf/Hi_P_12L10A0G2H1W_260402/xml/Hi_P_12L10A0G2H1W_Simplify_260402_with_armature.xml",
        dof_size=25,
        hydra_robot="Hi/Hi_P_12L10A0G2H1W_260402",
    ),
    "Hi_P_12L10A0G2H1W_260402": RobotSpec(
        name="h1_260402",
        default_data_path=HUMANOIDVERSE_DIR / "data" / "Hi_P_12L10A0G2H1W_260402_lafandataset_260714" / "h1_lafan.pkl",
        mujoco_xml_path="package://ht_urdf/Hi_P_12L10A0G2H1W_260402/xml/Hi_P_12L10A0G2H1W_Simplify_260402_with_armature.xml",
        dof_size=25,
        hydra_robot="Hi/Hi_P_12L10A0G2H1W_260402",
    ),
}


def get_robot_spec(robot: RobotName) -> RobotSpec:
    try:
        return ROBOT_SPECS[robot]
    except KeyError as exc:
        raise ValueError(f"Unsupported robot {robot!r}. Expected one of: {sorted(ROBOT_SPECS)}") from exc


def _load_motion(data_path: Path, motion: int | str):
    data = joblib.load(data_path)
    if not isinstance(data, dict):
        raise TypeError(f"Expected {data_path} to contain a dict, got {type(data).__name__}.")

    keys = list(data.keys())
    if isinstance(motion, int):
        if motion < 0 or motion >= len(keys):
            raise IndexError(f"Motion index {motion} is out of range. File has {len(keys)} motions.")
        key = keys[motion]
    else:
        key = motion
        if key not in data:
            preview = ", ".join(keys[:10])
            raise KeyError(f"Motion key {key!r} not found. First keys: {preview}")

    return key, data[key], len(keys)


def _motion_to_qpos(motion_data: dict, dof_size: int) -> np.ndarray:
    required = ("root_trans_offset", "root_rot", "dof")
    missing = [key for key in required if key not in motion_data]
    if missing:
        raise KeyError(f"Motion is missing required field(s): {missing}")

    root_pos = np.asarray(motion_data["root_trans_offset"], dtype=np.float64)
    root_quat_xyzw = np.asarray(motion_data["root_rot"], dtype=np.float64)
    dof = np.asarray(motion_data["dof"], dtype=np.float64)

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"root_trans_offset must have shape (T, 3), got {root_pos.shape}.")
    if root_quat_xyzw.ndim != 2 or root_quat_xyzw.shape[1] != 4:
        raise ValueError(f"root_rot must have shape (T, 4), got {root_quat_xyzw.shape}.")
    if dof.ndim != 2 or dof.shape[1] != dof_size:
        raise ValueError(f"dof must have shape (T, {dof_size}), got {dof.shape}.")
    if not (root_pos.shape[0] == root_quat_xyzw.shape[0] == dof.shape[0]):
        raise ValueError(
            "root_trans_offset, root_rot, and dof must have the same frame count: "
            f"{root_pos.shape[0]}, {root_quat_xyzw.shape[0]}, {dof.shape[0]}."
        )

    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]
    return np.concatenate([root_pos, root_quat_wxyz, dof], axis=-1)


def _set_qpos(model, data, qpos: np.ndarray) -> None:
    import mujoco

    if qpos.shape[-1] != model.nq:
        raise ValueError(f"Motion qpos has {qpos.shape[-1]} values, but MuJoCo model nq is {model.nq}.")
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _open_model(robot_spec: RobotSpec):
    import mujoco

    return mujoco.MjModel.from_xml_path(str(resolve_asset_path("", robot_spec.mujoco_xml_path)))


def _camera_name(model, requested: str | None) -> str | None:
    import mujoco

    if requested is None:
        requested = "track"
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, requested)
    return requested if camera_id >= 0 else None


def render_video(
    qpos: np.ndarray,
    robot_spec: RobotSpec,
    output: Path,
    fps: int,
    width: int,
    height: int,
    camera: str | None,
    stride: int,
) -> None:
    import mujoco
    import mediapy as media

    model = _open_model(robot_spec)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=width, height=height)
    camera = _camera_name(model, camera)

    frames = []
    sampled_qpos = qpos[::stride]
    for idx, q in enumerate(sampled_qpos):
        if idx == 0 or idx + 1 == len(sampled_qpos) or (idx + 1) % 100 == 0:
            print(f"Rendering frame {idx + 1}/{len(sampled_qpos)}")
        _set_qpos(model, data, q)
        if camera is None:
            renderer.update_scene(data)
        else:
            renderer.update_scene(data, camera=camera)
        frames.append(renderer.render())

    output.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(str(output), frames, fps=max(1, round(fps / stride)))
    renderer.close()


def play_viewer(qpos: np.ndarray, robot_spec: RobotSpec, fps: int, camera: str | None, stride: int) -> None:
    import mujoco
    import mujoco.viewer

    model = _open_model(robot_spec)
    data = mujoco.MjData(model)
    camera = _camera_name(model, camera)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        if camera is not None:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        frame_dt = stride / fps
        for q in qpos[::stride]:
            if not viewer.is_running():
                break
            start = time.time()
            _set_qpos(model, data, q)
            viewer.sync()
            sleep_time = frame_dt - (time.time() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)


def main(
    data_path: Path | None = None,
    robot: RobotName = "g1",
    motion: int | str = 0,
    output: Path | None = None,
    viewer: bool = True,
    start: int = 0,
    max_frames: int | None = 500,
    stride: int = 1,
    width: int = 640,
    height: int = 480,
    camera: str | None = "track",
) -> None:
    """Visualize a converted robot motion pkl with MuJoCo."""

    if not viewer:
        os.environ.setdefault("MUJOCO_GL", "egl")

    robot_spec = get_robot_spec(robot)
    if data_path is None:
        data_path = robot_spec.default_data_path
    data_path = data_path.resolve()
    key, motion_data, num_motions = _load_motion(data_path, motion)
    qpos = _motion_to_qpos(motion_data, robot_spec.dof_size)

    fps = int(motion_data.get("fps", 30))
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}.")
    if start < 0 or start >= len(qpos):
        raise ValueError(f"start must be in [0, {len(qpos) - 1}], got {start}.")
    end = len(qpos) if max_frames is None else min(len(qpos), start + max_frames)
    qpos = qpos[start:end]

    mujoco_xml_path = resolve_asset_path("", robot_spec.mujoco_xml_path)
    print(f"Loaded {data_path}")
    print(f"Robot: {robot_spec.name} / XML: {mujoco_xml_path}")
    print(f"Motion {motion!r}: {key} ({num_motions} motions in file)")
    print(f"Frames: {len(qpos)} / fps: {fps} / stride: {stride}")

    if viewer:
        play_viewer(qpos, robot_spec=robot_spec, fps=fps, camera=camera, stride=stride)
        return

    if output is None:
        stem = data_path.stem.replace("/", "_")
        safe_key = str(key).replace("/", "_")
        output = Path("outputs") / "motion_videos" / f"{robot_spec.name}_{stem}_{safe_key}.mp4"
    render_video(qpos, robot_spec=robot_spec, output=output, fps=fps, width=width, height=height, camera=camera, stride=stride)
    print(f"Saved {output}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
