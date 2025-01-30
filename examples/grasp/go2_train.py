import argparse
import os
import pickle
import shutil
import datetime

import torch

from go2_env import Go2Env, ControlType, NUM_ACTIONS_OF_CTRL_TYPE, get_num_obs
from teacher import teacher_policy

import genesis as gs
from runner import RunnerWithTB as Runner

USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
    # from google.colab import drive
    # drive.mount('/mnt/drive')
    google_drive_dir = "/mnt/drive/MyDrive/logs"

workdir = "logs"

def get_cfgs(args):

    ## Reward config
    reward_cfg= {
        'target_height': 0.05,
        'reward_scales': {
            'lift_height': 1.0,
            'obj_dist': 0.5,
            'finger_pressure': 0.0,
            'action_rate': -0.0001, 
            "similar_to_default": -0.1,
        }
    }

    ## Environment config
    env_cfg = {
        'control_type': ControlType(args.control_type),
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
        'action_scale': 1.0, 
        'simulate_action_latency': True, 
        'clip_actions': 100.0
    }
    env_cfg["num_actions"] = NUM_ACTIONS_OF_CTRL_TYPE[env_cfg["control_type"]]

    if env_cfg["control_type"] == ControlType.DISCRETE:
        env_cfg["num_actions"] = 15 # 6x2 move hand commands + 2 open/close gripper commands + 1 move hand velocity
    elif env_cfg["control_type"] == ControlType.CARTESIAN:
        env_cfg["num_actions"] = 8 # 6 dof of hand + 2 dof for fingers
    elif env_cfg["control_type"] in [ControlType.JOINT_POS, ControlType.JOINT_VEL, ControlType.JOINT_FORCE]:
        env_cfg["num_actions"] = 9

    ## Observation config
    obs_cfg= {
        'reach_dir_and_dist': True,         # 6
        'finger_contact_force': True,      # 2x3
        'links_lin_vel': False,             # 11x3
        'links_ang_vel': False,             # 11x3
        'links_projected_gravity': False,   # 11x3
        'dof_pos': True,                    # 9
        'dof_vel': True,                   # 9
        'dof_force': False,                 # 9
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
    obs_cfg["num_obs"] = get_num_obs(obs_cfg, env_cfg)

    train_cfg = {
        'algorithm': {            
            'gamma': 0.99,                         
            'num_learning_epochs': args.epochs, 
            'num_mini_batches': args.num_batches,
            'use_clipped_value_loss': True,             
        }, 
        'init_member_classes': {}, 
        'policy': {
            'activation': 'elu', 
            'actor_hidden_dims': [512, 256, 128],
            'critic_hidden_dims': [512, 256, 128],
            # 'actor_shortcut': args.actor_shortcut,
            # 'critic_shortcut': args.critic_shortcut,
        }, 
        'runner': {
            'algorithm_class_name': args.alg_name,
            'checkpoint': -1, 
            'experiment_name': args.exp_name, 
            'load_run': -1, 
            'log_interval': 1, 
            'max_iterations': args.max_iterations, 
            'num_steps_per_env': args.num_steps_per_env,
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

    if  train_cfg["runner"]["algorithm_class_name"] == "PPO":
        train_cfg["algorithm"].update({
            'clip_param': 0.2,            
            'desired_kl': 0.01, 
            'entropy_coef': 0.01, 
            'lam': 0.95,             
            'max_grad_norm': 1.0, 
            "learning_rate": 0.001,
            'schedule': 'adaptive',
            'value_loss_coef': 1.0, 
        })
        train_cfg["policy"].update({
            'init_noise_std': 1.0,
        })
    elif train_cfg["runner"]["algorithm_class_name"] == "DDPG":
        train_cfg["algorithm"].update({
            "actor_lr": 1e-4,
            "critic_lr": 1e-3,
        })
    
    return env_cfg, obs_cfg, reward_cfg, train_cfg


def parse_args():
    parser = argparse.ArgumentParser()

    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d_%H-%M-%S")
    default_exp_name = f"grasp_{now}"
    parser.add_argument("-e", "--exp_name", type=str, default=default_exp_name)
    parser.add_argument("-a", "--alg_name", type=str, default="PPO")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=100)
    parser.add_argument("--num_batches", type=int, default=16)
    parser.add_argument("--num_steps_per_env", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--control_type", type=int, default=2)
    parser.add_argument("--teacher", action='store_true')
    parser.add_argument("--verbose", default=False, action='store_true')
    # parser.add_argument("--actor_shortcut", default=False, action='store_true')
    # parser.add_argument("--critic_shortcut", default=False, action='store_true')
    args = parser.parse_args()
    return args

def main():
    
    args = parse_args() 
    if USE_CUDA:
        gs.init(logging_level="warning")
    else:
        gs.init(backend=gs.cpu, logging_level="warning")

    log_dir = os.path.join(workdir, f"{args.exp_name}")
    env_cfg, obs_cfg, reward_cfg, train_cfg = get_cfgs(args)     

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    if USE_CUDA:
        device = "cuda:0"
    else:
        device = "cpu"

    env = Go2Env(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, device=device, verbose=args.verbose,        
    )    

    if args.teacher:
        teacher = lambda obs : teacher_policy(
            env.franka, 
            env.end_effector, 
            env.default_dof_pos, 
            env.episode_length_buf,
            obs, 
            env_cfg, 
            obs_cfg,
            device,
        )
    else:
        teacher = None

    runner = Runner(env, train_cfg, log_dir, teacher=teacher, device=device)        

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

    if USE_CUDA:
        os.system(f"cp -r {log_dir} {os.path.join(google_drive_dir, args.exp_name)}")


if __name__ == "__main__":
    main()

"""
# training
python examples/locomotion/go2_train.py
"""
