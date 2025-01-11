import argparse
import os
import pickle

import numpy as np
import torch
from go2_env import Go2Env, ControlType
from rsl_rl.runners import OnPolicyRunner

import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'

import genesis as gs

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

def get_random_action(env_cfg):

    n = env_cfg["num_actions"]
    #pos0 = torch.tensor(env_cfg["default_joint_angles"])
    sigma = 0.01
    actions = torch.randn(n) * sigma
    #actions = pos0 + delta
    return actions


def decompose_obs(obs, env_cfg, obs_cfg):

    obs_dict = {}

    obs_dim = {
        "reach_dir_and_dist": 6,
        "finger_contact_force": 6,
        "links_lin_vel": 33,
        "links_ang_vel": 33,
        "links_projected_gravity": 33,
        "dof_pos": 9,
        "dof_vel": 9,
        "dof_force": 9,
        "actions": env_cfg["num_actions"],
    }

    start_idx = 0
    for name in obs_dim:
        if obs_cfg[name]:
            d = obs_dim[name]
            obs_dict[name] = obs[:,start_idx:start_idx+d]
            start_idx += d
    assert start_idx == obs.shape[-1]
    return obs_dict

def teacher_policy(obs, env_cfg, obs_cfg):

    print ("===== teacher_policy")
    obs_dict = decompose_obs(obs, env_cfg, obs_cfg)
    assert env_cfg["control_type"] == ControlType.CARTESIAN

    actions = torch.zeros(8)    
    actions[:3] += (obs_dict["reach_dir_and_dist"][0,:3] + obs_dict["reach_dir_and_dist"][0,3:])/2.0
    print (obs_dict)
    print ("actions=", actions)
    
    return actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="learn2grasp")
    parser.add_argument("--ckpt", type=int, default=100)
    parser.add_argument("--teacher", action='store_true')
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    log_dir = f"logs/{args.exp_name}"
    print ("log_dir=", log_dir)
    #from go2_train import env_cfg, obs_cfg, reward_cfg, train_cfg
    env_cfg, obs_cfg, reward_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    # reward_cfg["reward_scales"] = {}

    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        device=device,
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")

    if args.teacher:
        policy = lambda obs : teacher_policy(obs, env_cfg, obs_cfg)
    elif os.path.isfile(resume_path):
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=device)
    else:
        policy = None

    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            if policy is not None:
                actions = policy(obs)
            else:
                actions = get_random_action(env_cfg)
                
        
            obs, _, rews, dones, infos = env.step(actions)
            print ("rewards=", rews)


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking -v --ckpt 100
"""
