import argparse
from collections import OrderedDict
import gymnasium
#import metaworld
import torch
import itertools
import json
import os
import pickle
import numpy as np
import shutil

from torch.utils.tensorboard import SummaryWriter

from rlkit.evaluation import evaluate_episode_return
from configs.env import args_point_robot

from rlkit.replay_memory import ReplayMemory, ReplayMemoryForML1
from rlkit.sac import SAC
from src.envs.point_robot import PointEnv

parser = argparse.ArgumentParser()
parser.add_argument('--env_type', default='point_robot')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--log_freq', type=int, default=100)
parser.add_argument('--save_freq', type=int, default=40)
parser.add_argument('--task_id_start', type=int, default=0)
parser.add_argument('--task_id_end', type=int, default=5)
parser.add_argument("--reset_dir",action="store_true",help="Reset dataset directories")
args, rest_args = parser.parse_known_args()
env_type = args.env_type
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
log_freq = args.log_freq
save_freq = args.save_freq
task_id_start = args.task_id_start
task_id_end = args.task_id_end
reset_dir = args.reset_dir

if env_type == 'point_robot':
    args = args_point_robot.get_args(rest_args)
else:
    raise NotImplementedError

ReplayBuffer = ReplayMemoryForML1 if env_type == 'reach' else ReplayMemory

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Environment
if env_type == 'point_robot':
    #tasks = np.load('./datasets/PointRobot-v0/task_goals.npy')
    env = PointEnv(max_episode_steps=args.max_episode_steps, num_tasks=args.num_tasks)
    #env.load_all_tasks(tasks)
else:
    raise NotImplementedError
#env.seed(args.seed)
if env_type not in ['walker', 'hopper']:
    env.action_space.seed(args.seed)

# directories preparing
def prepare_dir(path, reset=False):
    """
    If reset=True:
        - delete the directory if it exists, then recreate it
    If reset=False:
        - create the directory only if it does not exist
    """
    if reset and os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

save_data_path = f"datasets/{args.env_name}"
save_buffer_path = f"{save_data_path}/replay_buffer"
save_checkpoint_path = f"{save_data_path}/checkpoints"

prepare_dir(save_data_path, reset=reset_dir)
prepare_dir(save_buffer_path, reset=reset_dir)
prepare_dir(save_checkpoint_path, reset=reset_dir)

# save goal positions of all tasks (given task_goals)
# if env_type=='point_robot':
#     np.save(f'{save_data_path}/task_goals.npy', np.array(env.goals))
# else:
#     with open(f'{save_data_path}/task_goals.pkl', 'wb') as file:
#         pickle.dump(env.tasks, file)

return_scales = []  # record the min and max returns of the collected offline datasets
for task_id in range(task_id_start, task_id_end):
    # set the specific task id for data collection
    print(f'\n\n=============== Collect data for task {task_id} ===============')
    env.reset_task(task_id)

    # Agent
    agent = SAC(env, args.hidden_dim, args.alpha, args.lr, args.gamma, args.tau, device=device)

    # Tesnorboard
    results_dir = os.path.join(f'runs/{args.env_name}/data_collection/task_{task_id}')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    writer = SummaryWriter(results_dir)
    # checkpoints path for each task
    checkpoint_path_task = f"{save_checkpoint_path}/task_{task_id}"
    if not os.path.exists(checkpoint_path_task):
        os.mkdir(checkpoint_path_task)

    # Memory
    memory = ReplayBuffer(args.replay_size, args.seed)

    # Training Loop
    total_numsteps = 0

    # a list of episode trajectories, each trajectory is saved in the format of dictionary
    episode_returns = []
    for i_episode in itertools.count(1):
        epi_return = 0.0
        state = env.reset()

        for step in range(args.max_episode_steps):
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > args.batch_size:
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss = agent.update_parameters(memory, args.batch_size)

                if total_numsteps % log_freq == 0:
                    writer.add_scalar('loss/critic_1', critic_1_loss, total_numsteps)
                    writer.add_scalar('loss/critic_2', critic_2_loss, total_numsteps)
                    writer.add_scalar('loss/policy', policy_loss, total_numsteps)

            total_numsteps += 1
            next_state, reward, done, info = env.step(action) 
            epi_return += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = True if step == args.max_episode_steps-1 else (not done)

            # Append transition to memory
            if env_type == 'reach':
                memory.push(state, action, reward, next_state, done, mask, info['grasp_success'], info['success']) 
            else:
                memory.push(state, action, reward, next_state, done, mask) 

            state = np.copy(next_state)

            # save model
            if total_numsteps % save_freq == 0:
                agent.save(f"{checkpoint_path_task}/agent_{total_numsteps}.pt")
            if done:
                break

        episode_returns.append(epi_return)

        writer.add_scalar('episode return/train', epi_return, total_numsteps)
        print(f'\nTask {task_id}, Episode {i_episode}, total numsteps {total_numsteps}, episode steps {step+1}, return: {round(epi_return, 2)}')

        # evaluate the trained agent
        eval_epi_return = evaluate_episode_return(env, agent, num_eval_episodes=args.num_eval_episodes,max_len=args.max_episode_steps)
            
        writer.add_scalar('episode return/test', eval_epi_return, total_numsteps)  
        print(f'Evaluate on {args.num_eval_episodes} episodes, average return {round(eval_epi_return, 2)}')  

        if total_numsteps >= args.num_steps:
            break

    # save the offline dataset
    # memory.save_buffer(f'{save_buffer_path}/dataset_task_{task_id}.pkl')

    # record the return of the offline dataset
    print(f'Return scale of the offline dataset: [{np.min(episode_returns)},{np.max(episode_returns)}]')
    return_scales.append([np.min(episode_returns), np.max(episode_returns)])


# save the task information
# containing: 1) the goal, and 2) the min and max returns in the collected offline dataset
task_info = OrderedDict()
for task_id in range(task_id_start, task_id_end):
    single_task = OrderedDict()
    single_task['goal'] = env.goals[task_id].tolist()
    #single_task['goal'] = list(env.tasks[task_id].values())
    single_task['return_scale'] = return_scales[task_id]
    task_info[f'task {task_id}'] = single_task
with open(f'{save_data_path}/task_info.json', 'w') as f:
    f.write(json.dumps(task_info, indent=4))
    f.close()