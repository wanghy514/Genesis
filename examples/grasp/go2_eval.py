import argparse
import os
import sys
import pickle

import numpy as np
import torch
from go2_env import Go2Env, ControlType, NUM_ACTIONS_OF_CTRL_TYPE, get_num_obs
from go2_train import workdir
from teacher import teacher_policy

import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'

import genesis as gs
from runner import RunnerWithTB as Runner

USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
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



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="learn2grasp")
    parser.add_argument("--ckpt", type=int, default=100)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--teacher", action='store_true')
    parser.add_argument("--verbose", default=False, action='store_true')
    args = parser.parse_args()

    
    if USE_CUDA:
        gs.init()
    else:
        gs.init(backend=gs.cpu)

    log_dir = os.path.join(workdir, f"{args.exp_name}")
    print ("log_dir=", log_dir)    
    env_cfg, obs_cfg, reward_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    # reward_cfg["reward_scales"] = {}

    if args.teacher:
        env_cfg["control_type"] = ControlType.JOINT_POS
        env_cfg["num_actions"] = NUM_ACTIONS_OF_CTRL_TYPE[env_cfg["control_type"]]
        assert obs_cfg["reach_dir_and_dist"] == True
        obs_cfg["num_obs"] = get_num_obs(obs_cfg, env_cfg)

    env = Go2Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        device=device,
        show_viewer=True,
        show_pinch_pos=False,
        verbose=args.verbose,
    )

    runner = Runner(env, train_cfg, log_dir, device=device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")


    # print (dir(env.franka))
    # import sys
    # sys.exit()

    if args.teacher:
        policy = lambda obs : teacher_policy(
            env.franka, 
            env.end_effector, 
            env.default_dof_pos, 
            env.episode_length_buf,
            obs, 
            env_cfg, 
            obs_cfg,
            device,
        )
    elif os.path.isfile(resume_path):
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=device)
    else:
        policy = None

    obs, _ = env.reset()
    all_returns = []
    returns = torch.zeros(args.num_envs)
    with torch.no_grad():
        while True:            
            if policy is not None:
                actions = policy(obs)
            else:
                actions = get_random_action(env_cfg)

            obs, rews, dones, infos = env.step(actions)
            for batch_idx in range(args.num_envs):
                if dones[batch_idx].item() == 0:
                    returns[batch_idx] += rews[batch_idx].item()
                else:
                    all_returns.append(returns[batch_idx].item())
                    returns[batch_idx] = 0.0                   
            print (" ============= all_returns = ", ["%.2f" % v for v in all_returns])
            print ("avg. rewards=", np.mean(all_returns))            

if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking -v --ckpt 100
"""
