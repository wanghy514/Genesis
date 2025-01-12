import argparse
import os
import pickle
import shutil


import torch

from go2_env import Go2Env, ControlType
from runner import Runner

import genesis as gs

USE_CUDA = torch.cuda.is_available()

## Reward config
reward_cfg= {
    'target_height': 0.3,
    'inv_dist_bound': 1.0,
    'reward_scales': {
        'lift_height': 1.0,
        'obj_inv_dist': 1.0,
        'finger_pressure': 0.0,
        'action_rate': -0.0001, 
        "similar_to_default": -0.1,
    }
}


## Environment config
env_cfg = {
    'control_type': ControlType.DISCRETE,
    'num_links': 11,
    'num_fingers': 2,
    'default_joint_angles': [-1.0432,  1.4372,  1.5254, -1.7213, -1.4453,  1.6352,  1.4565, 0.04, 0.04],
    'dof_names': [
        'joint1',
        'joint2',
        'joint3',
        'joint4',
        'joint5',
        'joint6',
        'joint7',
        'finger_joint1',
        'finger_joint2',
    ],
    'link_names': [
        'link0',
        'link1',
        'link2',
        'link3',
        'link4',
        'link5',
        'link6',
        'link7',
        'hand',
        'left_finger',
        'right_finger',
    ],
    'episode_length_s': 5.0, 
    #'resampling_time_s': 4.0, 
    'action_scale': 1.0, 
    'simulate_action_latency': True, 
    'clip_actions': 100.0
}

if env_cfg["control_type"] == ControlType.DISCRETE:
    env_cfg["num_actions"] = 15 # 6x2 move hand commands + 2 open/close gripper commands + 1 move hand velocity
elif env_cfg["control_type"] == ControlType.CARTESIAN:
    env_cfg["num_actions"] = 8 # 6 dof of hand + 2 dof for fingers
elif env_cfg["control_type"] == ControlType.JOINT:
    env_cfg["num_actions"] = 9

## Observation config
obs_cfg= {
    'reach_dir_and_dist': True,         # 6
    'finger_contact_force': True,      # 2x3
    'links_lin_vel': False,             # 11x3
    'links_ang_vel': False,             # 11x3
    'links_projected_gravity': True,   # 11x3
    'dof_pos': True,                    # 9
    'dof_vel': False,                   # 9
    'dof_force': True,                 # 9
    'actions': True,
    'obs_scales': {
        'lin_vel': 0.1,
        'ang_vel': 0.1, 
        'contact_force': 2.0,
        'dof_pos': 1.0, 
        'dof_vel': 0.05,
        'dof_force': 1.0,
    }
}

num_obs = 0
if obs_cfg["reach_dir_and_dist"]:
    num_obs += 6
if obs_cfg["finger_contact_force"]:
    num_obs += 6
if obs_cfg["links_lin_vel"]:
    num_obs += 33
if obs_cfg["links_ang_vel"]:
    num_obs += 33
if obs_cfg["links_projected_gravity"]:
    num_obs += 33
if obs_cfg["dof_pos"]:
    num_obs += 9
if obs_cfg["dof_vel"]:
    num_obs += 9
if obs_cfg["dof_force"]:
    num_obs += 9    
if obs_cfg["actions"]:
    num_obs += env_cfg["num_actions"]

obs_cfg["num_obs"] = num_obs


# Training config
train_cfg= {
    'algorithm': {
        'clip_param': 0.2, 
        'desired_kl': 0.01, 
        'entropy_coef': 0.01, 
        'gamma': 0.99, 
        'lam': 0.95, 
        'learning_rate': 0.001, 
        'max_grad_norm': 1.0, 
        'num_learning_epochs': 5, 
        'num_mini_batches': 4, 
        'schedule': 'adaptive', 
        'use_clipped_value_loss': True, 
        'value_loss_coef': 1.0
    }, 
    'init_member_classes': {}, 
    'policy': {
        'activation': 'elu', 
        'actor_hidden_dims': [512, 256, 128], 
        'critic_hidden_dims': [512, 256, 128], 
        'init_noise_std': 1.0
    }, 
    'runner': {
        'algorithm_class_name': 'PPO', 
        'checkpoint': -1, 
        'experiment_name': 'learn2grasp', 
        'load_run': -1, 
        'log_interval': 1, 
        'max_iterations': 100, 
        'num_steps_per_env': 24, 
        'policy_class_name': 'ResActorCritic', 
        'record_interval': -1, 
        'resume': False, 
        'resume_path': None, 
        'run_name': '', 
        'runner_class_name': 'runner_class_name', 
        'save_interval': 100
    }, 
    'runner_class_name': 'Runner', 
    'seed': 1
}


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = train_cfg
    train_cfg_dict["runner"]["experiment_name"] = exp_name
    train_cfg_dict["runner"]["max_iterations"] = max_iterations    
    return train_cfg_dict


def get_cfgs():    
    return env_cfg, obs_cfg, reward_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="learn2grasp")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=100)
    args = parser.parse_args()

    if USE_CUDA:
        gs.init(logging_level="warning")
    else:
        gs.init(backend=gs.cpu, logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)


    if USE_CUDA:
        device = "cuda:0"
    else:
        device = "cpu"

    env = Go2Env(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, device=device
    )    
    runner = Runner(env, train_cfg, log_dir, device=device)        

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training
python examples/locomotion/go2_train.py
"""
