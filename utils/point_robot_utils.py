import json
import numpy as np
from pathlib import Path

def load_goals_from_task_info(json_path, save_path=None):
    """
    从 task_info.json 中读取 goal，并转换为 numpy array

    Parameters
    ----------
    json_path : str or Path
        task_info_xxx.json 的路径
    save_path : str or Path, optional
        若提供，则将 goals 保存为 .npy 文件

    Returns
    -------
    goals : np.ndarray
        shape = (num_tasks, goal_dim)
    """
    json_path = Path(json_path)

    with open(json_path, 'r') as f:
        task_info = json.load(f)

    # 按 task id 排序，确保顺序一致
    goals = []
    for task_key in sorted(task_info.keys(), key=lambda x: int(x.split()[-1])):
        goal = task_info[task_key]['goal']
        goals.append(goal)

    goals = np.array(goals, dtype=np.float32)

    if save_path is not None:
        save_path = Path(save_path)
        np.save(save_path, goals)

    return goals