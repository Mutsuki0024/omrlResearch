import argparse
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.envs.point_robot import PointEnv
from rlkit.sac import SAC


# ============================================================
# Load task info (your exact JSON format)
# ============================================================

def load_tasks_from_task_info(data_dir):
    """
    task_info.json format:
    {
        "task 0": {"goal": [x, y], "return_scale": [min, max]},
        ...
    }
    """
    path = os.path.join(data_dir, "task_info.json")
    with open(path, "r") as f:
        task_info = json.load(f)

    task_keys = sorted(
        task_info.keys(),
        key=lambda k: int(k.split()[-1])
    )

    goals = np.array(
        [task_info[k]["goal"] for k in task_keys],
        dtype=np.float32
    )

    return goals, task_info


# ============================================================
# Rollout (STOCHASTIC + visualization-only termination)
# ============================================================

def rollout(env, agent, max_steps, goal, stop_eps, init_state_noise):
    obs = env.reset()
    
    traj = [np.array(obs, dtype=np.float32)]

    for _ in range(max_steps):
        with torch.no_grad():
            # stochastic action (Gaussian sampling)
            action = agent.select_action(obs, evaluate=False)

        next_obs, _, done, _ = env.step(action)
        traj.append(np.array(next_obs, dtype=np.float32))
        obs = next_obs

        # visualization-only stop condition
        if np.linalg.norm(obs - goal) < stop_eps:
            break

        if done:
            break

    return np.array(traj)


# ============================================================
# Plotting
# ============================================================

def plot_trajectories(ax, trajectories, goal, task_id):
    for i, traj in enumerate(trajectories):
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            linewidth=1.5,
            alpha=0.8,
            label="trajectory" if i == 0 else None
        )

    ax.scatter(0.0, 0.0, s=80, marker="s", label="start")
    ax.scatter(goal[0], goal[1], s=140, marker="*", label="goal")

    ax.annotate(
        f"goal (task {task_id})",
        (goal[0], goal[1]),
        textcoords="offset points",
        xytext=(6, 6)
    )

    ax.set_title(f"Task {task_id} | stochastic SAC rollouts")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--task_id", type=int, required=True)

    parser.add_argument("--episodes", type=int, default=1,
                        help="number of trajectories to overlay")
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--stop_eps", type=float, default=0.02)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    
    parser.add_argument("--init_state_noise", type=float, default=0.1)

    # SAC hyperparameters (must match training)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)

    parser.add_argument("--save_png", type=str, default="")
    args = parser.parse_args()

    # NOTE:
    # We still fix the seed so randomness is *stochastic but reproducible*
    #np.random.seed(args.seed)
    #torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------
    # Load task info
    # --------------------------------------------------

    goals, _ = load_tasks_from_task_info(args.data_dir)
    assert 0 <= args.task_id < len(goals)

    goal = goals[args.task_id]

    # --------------------------------------------------
    # Environment
    # --------------------------------------------------

    env = PointEnv(
        max_episode_steps=args.max_steps,
        num_tasks=len(goals)
    )
    env.load_all_tasks(goals)
    env.reset_task(args.task_id)

    # --------------------------------------------------
    # Agent
    # --------------------------------------------------

    agent = SAC(
        env,
        args.hidden_dim,
        args.alpha,
        args.lr,
        args.gamma,
        args.tau,
        device=device
    )
    agent.load(args.ckpt)

    # --------------------------------------------------
    # Rollouts
    # --------------------------------------------------

    trajectories = []
    for _ in range(args.episodes):
        traj = rollout(
            env,
            agent,
            args.max_steps,
            goal,
            stop_eps=args.stop_eps,
            init_state_noise=args.init_state_noise
        )
        trajectories.append(traj)

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_trajectories(ax, trajectories, goal, args.task_id)

    plt.tight_layout()

    if args.save_png:
        plt.savefig(args.save_png, dpi=200)
        print(f"Saved figure to {args.save_png}")

    plt.show()


if __name__ == "__main__":
    main()
