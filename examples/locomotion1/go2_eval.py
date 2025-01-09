import argparse
import os
import pickle

import torch
from go2_env import Go2Env

import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'

import genesis as gs
from runner import RunnerWithTB as Runner

USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
    device = 'cuda:0'
else:
    device = 'cpu'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("--ckpt", type=int, default=100)
    args = parser.parse_args()
    
    if USE_CUDA:
        gs.init()
    else:
        gs.init(backend=gs.cpu)    

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
        device=device,
    )

    runner = Runner(env, train_cfg, log_dir, device=device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=device)

    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            #obs, _, rews, dones, infos = env.step(actions)
            obs, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking -v --ckpt 100
"""
