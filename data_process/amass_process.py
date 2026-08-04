import numpy as np
import joblib
import glob
import os
from scipy.spatial.transform import Rotation as sRot

# Step 1: Load reference pkl to align types
ref_data = joblib.load("data/lafan_tail.pkl")
ref = ref_data["dance1_subject2"]

# Step 2: Find all npz files
npz_files = glob.glob("dataprocess/dancedb/*.npz")

all_data = {}
import ipdb; ipdb.set_trace()
for npz_path in npz_files:
    key = os.path.splitext(os.path.basename(npz_path))[0]
    data_npz = np.load(npz_path)
    data_dict = {k: data_npz[k] for k in data_npz.files}

    # 取出需要的四元数片段（仍是 wxyz）
    # import ipdb; ipdb.set_trace()


    # 转为 axis-angle
    root_rot = np.concatenate([
            data_dict["body_rotations"][:, 0, 1:],  # 取 x, y, z
            data_dict["body_rotations"][:, 0, :1],  # 取 w
        ], axis=-1).astype(ref["root_rot"].dtype)
    # import ipdb; ipdb.set_trace()
    root_rot_vec = sRot.from_quat(root_rot).as_rotvec()

    dof_axis = np.array([[0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]])

    dof = np.concatenate([
        data_dict["dof_positions"]
    ], axis=1).astype(ref["dof"].dtype)
    pose_aa = np.concatenate([root_rot_vec[:, None, :], dof_axis * dof[:, :, None]], axis=1)

    # 构造新 dict
    new_data_dict = {
        "root_trans_offset": data_dict["body_positions"][:, 0].astype(ref["root_trans_offset"].dtype),
        "pose_aa": pose_aa.astype(ref["pose_aa"].dtype),
        "fps": int(ref["fps"])
    }

    all_data[key] = new_data_dict

# Step 3: Save using joblib
save_path = "dataprocess/dancedb.pkl"
joblib.dump(all_data, save_path)
print(f"✅ Saved data to {save_path} using joblib, keys: {list(all_data.keys())}")