import argparse
import os
import sys
import pickle

import numpy as np
import torch
from go2_env import Go2Env, ControlType, NUM_ACTIONS_OF_CTRL_TYPE, get_num_obs
from go2_train import workdir

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

def straight_line_planner(qpos, qpos_goal, num_waypoints):

    step_size = 1.0/(num_waypoints-1)
    steps = np.arange(0.0, 1.0 + step_size/2.0, step_size)
    assert len(steps) == num_waypoints
    return [
        qpos * (1.0-s) + qpos_goal * s
        for s in steps
    ]
    


path0 = None
path1 = None
path2 = None
step_cnt = 0

def teacher_policy(franka, end_effector, default_dof_pos, obs, env_cfg, obs_cfg):

    global path0
    global path1    
    global path2    
    global step_cnt

    warm_up_steps = 20
    gripper_close_pos = -0.04

    assert env_cfg["control_type"] == ControlType.JOINT_POS
    actions = torch.zeros(9)    
    
    # Plan for lateral move
    if step_cnt == warm_up_steps:
        obs_dict = decompose_obs(obs, env_cfg, obs_cfg)
        waypoint0 = torch.concat([end_effector.get_pos()[0,:], end_effector.get_quat()[0,:]])        
        waypoint0[:2] += (obs_dict["reach_dir_and_dist"][0,:2] + obs_dict["reach_dir_and_dist"][0,3:5])/2.0                
        pos0 = franka.inverse_kinematics(
            link = end_effector,
            pos  = waypoint0[:3].unsqueeze(0),
            quat = waypoint0[3:].unsqueeze(0),
        )        
        assert pos0.shape[0] == 1        
        path0 = straight_line_planner( #franka.plan_path(
            qpos = obs_dict["dof_pos"] / obs_cfg["obs_scales"]["dof_pos"] + default_dof_pos,
            qpos_goal = pos0,
            num_waypoints = 100 - step_cnt,
        )
       
    # Plan for reach down
    if step_cnt == 100:
        obs_dict = decompose_obs(obs, env_cfg, obs_cfg)                
        waypoint1 = torch.concat([end_effector.get_pos()[0,:], end_effector.get_quat()[0,:]])        
        waypoint1[2:3] += (obs_dict["reach_dir_and_dist"][0,2] + obs_dict["reach_dir_and_dist"][0,5])/2.0        
        pos1 = franka.inverse_kinematics(
            link = end_effector,
            pos  = waypoint1[:3].unsqueeze(0),
            quat = waypoint1[3:].unsqueeze(0),
        )
        assert pos1.shape[0] == 1
        path1 = straight_line_planner( # franka.plan_path(
            qpos = obs_dict["dof_pos"] / obs_cfg["obs_scales"]["dof_pos"] + default_dof_pos,
            qpos_goal = pos1,
            num_waypoints = 150 - step_cnt,
        )
    
    # Plan for lift up
    if step_cnt == 170:

        obs_dict = decompose_obs(obs, env_cfg, obs_cfg)                
        path2 = straight_line_planner( # franka.plan_path(
            qpos = obs_dict["dof_pos"] / obs_cfg["obs_scales"]["dof_pos"] + default_dof_pos,
            qpos_goal = default_dof_pos,
            num_waypoints = 250 - step_cnt,
        )    

    if step_cnt < warm_up_steps:
        pass
    elif step_cnt < 100:        
        t = len(path0) - (100 - step_cnt)
        actions[:7] = (path0[t][0,:7] - default_dof_pos[:7])/env_cfg["action_scale"]
        actions[-2:] = 0
        
    elif step_cnt < 150:
        t = len(path1) - (150 - step_cnt)
        actions[:7] = (path1[t][0,:7] - default_dof_pos[:7])/env_cfg["action_scale"]
        actions[-2:] = 0
        
    elif step_cnt < 170: 
        actions[:7] = (path1[-1][0,:7] - default_dof_pos[:7])/env_cfg["action_scale"]
        actions[-2:] = gripper_close_pos        

    elif step_cnt < 250:        
        t = len(path2) - (250 - step_cnt)        
        actions[:7] = (path2[t][0,:7] - default_dof_pos[:7])/env_cfg["action_scale"]
        actions[-2:] = gripper_close_pos
        
    step_cnt += 1
    return actions


def main():

    global step_cnt

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="learn2grasp")
    parser.add_argument("--ckpt", type=int, default=100)
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
        num_envs=1,
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
        policy = lambda obs : teacher_policy(env.franka, env.end_effector, env.default_dof_pos, obs, env_cfg, obs_cfg)
    elif os.path.isfile(resume_path):
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=device)
    else:
        policy = None

    obs, _ = env.reset()
    all_returns = []
    returns = 0.0
    with torch.no_grad():
        while True:            
            if policy is not None:
                actions = policy(obs)
            else:
                actions = get_random_action(env_cfg)

            obs, rews, dones, infos = env.step(actions)
            if dones.item() == 0:
                returns += rews.item()
            else:
                all_returns.append(returns)
                returns = 0.0    
                step_cnt = 0
            print (" ============= all_returns = ", ["%.2f" % v for v in all_returns])
            print ("avg. rewards=", np.mean(all_returns))            

if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking -v --ckpt 100
"""
