import yaml
import copy

INPUT_FILE = "g1_29dof_hard_waist.yaml"
OUTPUT_FILE = "g1_23dof_hard_waist.yaml"

# === 要删除的 joints ===
joints_to_remove = {
    'waist_roll_joint',
    'waist_pitch_joint',
    'left_wrist_pitch_joint',
    'left_wrist_yaw_joint',
    'right_wrist_pitch_joint',
    'right_wrist_yaw_joint'
}

# === 文件名替换规则 ===
REPLACE_RULES = [
    ("29dof", "23dof"),
    ("_29dof", "_23dof"),
]


def replace_string(obj):
    """递归替换所有字符串中的 29dof -> 23dof"""
    if isinstance(obj, dict):
        return {k: replace_string(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_string(v) for v in obj]
    elif isinstance(obj, str):
        new_s = obj
        for old, new in REPLACE_RULES:
            new_s = new_s.replace(old, new)
        return new_s
    else:
        return obj


def remove_by_index(lst, remove_indices):
    return [v for i, v in enumerate(lst) if i not in remove_indices]


def filter_name_list(lst):
    return [x for x in lst if x not in joints_to_remove]


def rebuild_symmetric_indices(sym_dict, remove_indices, total_len):
    old_to_new = {}
    new_idx = 0
    for old_idx in range(total_len):
        if old_idx not in remove_indices:
            old_to_new[old_idx] = new_idx
            new_idx += 1

    def remap(lst):
        return [old_to_new[i] for i in lst if i in old_to_new]

    new_sym = {}
    for k, v in sym_dict.items():
        new_list = remap(v)
        if len(new_list) > 0:
            new_sym[k] = new_list

    return new_sym


def main():
    with open(INPUT_FILE, 'r') as f:
        data = yaml.safe_load(f)

    robot = data["robot"]

    # === 1. 找删除 index ===
    dof_names = robot["dof_names"]
    remove_indices = {i for i, name in enumerate(dof_names) if name in joints_to_remove}

    print("Remove indices:", remove_indices)

    # === 2. name list 过滤 ===
    name_keys = [
        "dof_names",
        "upper_dof_names",
        "upper_left_arm_dof_names",
        "upper_right_arm_dof_names",
        "lower_dof_names",
        "waist_dof_names",
        "arm_dof_names",
        "left_arm_dof_names",
        "right_arm_dof_names"
    ]

    for key in name_keys:
        if key in robot:
            robot[key] = filter_name_list(robot[key])

    # === 3. default_joint_angles ===
    if "init_state" in robot and "default_joint_angles" in robot["init_state"]:
        angles = robot["init_state"]["default_joint_angles"]
        robot["init_state"]["default_joint_angles"] = {
            k: v for k, v in angles.items() if k not in joints_to_remove
        }

    # === 4. DOF 对齐 list ===
    list_keys = [
        "dof_pos_lower_limit_list",
        "dof_pos_upper_limit_list",
        "dof_vel_limit_list",
        "dof_effort_limit_list",
        "dof_armature_list",
        "dof_joint_friction_list"
    ]

    for key in list_keys:
        if key in robot:
            robot[key] = remove_by_index(robot[key], remove_indices)

    # === 5. symmetric 重建 ===
    if "symmetric_dofs_idx" in robot:
        robot["symmetric_dofs_idx"] = rebuild_symmetric_indices(
            robot["symmetric_dofs_idx"],
            remove_indices,
            len(dof_names)
        )

    # === 6. 更新维度 ===
    new_dof = len(robot["dof_names"])
    robot["dof_obs_size"] = new_dof
    robot["actions_dim"] = new_dof

    if "upper_dof_names" in robot:
        robot["upper_body_actions_dim"] = len(robot["upper_dof_names"])

    if "lower_dof_names" in robot:
        robot["lower_body_actions_dim"] = len(robot["lower_dof_names"])

    print(f"New DOF: {new_dof}")

    # === 7. 🔥 统一替换所有路径/模型名 ===
    data = replace_string(data)

    # === 8. 保存 ===
    with open(OUTPUT_FILE, 'w') as f:
        yaml.dump(data, f, sort_keys=False)

    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()