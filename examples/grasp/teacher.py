import torch
import numpy as np

from go2_env import ControlType


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
    


def teacher_policy(
        franka, 
        end_effector, 
        default_dof_pos, 
        episode_length_buf, 
        obs, 
        env_cfg, 
        obs_cfg,
        device,
    ):    

    warm_up_steps = 20
    gripper_close_pos = -0.04

    assert env_cfg["control_type"] == ControlType.JOINT_POS

    step_cnt = episode_length_buf[0].cpu().item()
    assert torch.all(episode_length_buf == step_cnt)

    num_envs = len(episode_length_buf)

    actions = torch.zeros(num_envs, 9).to(device)
    
    # for batch_idx in range(num_envs):
        
    # Plan for lateral move
    if step_cnt == warm_up_steps:
        obs_dict = decompose_obs(obs, env_cfg, obs_cfg)
        waypoint0 = torch.concat([end_effector.get_pos(), end_effector.get_quat()], dim=1)
        waypoint0[:, :2] += (obs_dict["reach_dir_and_dist"][:,:2] + obs_dict["reach_dir_and_dist"][:,3:5])/2.0                
        pos0 = franka.inverse_kinematics(
            link = end_effector,
            pos  = waypoint0[:, :3],
            quat = waypoint0[:, 3:],
        )        
        teacher_policy.path0 = straight_line_planner(
            qpos = obs_dict["dof_pos"] / obs_cfg["obs_scales"]["dof_pos"] + default_dof_pos,
            qpos_goal = pos0,
            num_waypoints = 100 - step_cnt,
        )
    
    # Plan for reach down
    if step_cnt == 100:
        obs_dict = decompose_obs(obs, env_cfg, obs_cfg)                
        waypoint1 = torch.concat([end_effector.get_pos(), end_effector.get_quat()], dim=1)
        waypoint1[:, 2:3] += (obs_dict["reach_dir_and_dist"][:,2:3] + obs_dict["reach_dir_and_dist"][:,5:])/2.0        
        pos1 = franka.inverse_kinematics(
            link = end_effector,
            pos  = waypoint1[:, :3],
            quat = waypoint1[:, 3:],
        )        
        teacher_policy.path1 = straight_line_planner(
            qpos = obs_dict["dof_pos"] / obs_cfg["obs_scales"]["dof_pos"] + default_dof_pos,
            qpos_goal = pos1,
            num_waypoints = 150 - step_cnt,
        )
    
    # Plan for lift up
    if step_cnt == 170:

        obs_dict = decompose_obs(obs, env_cfg, obs_cfg)                
        teacher_policy.path2 = straight_line_planner(
            qpos = obs_dict["dof_pos"] / obs_cfg["obs_scales"]["dof_pos"] + default_dof_pos,
            qpos_goal = default_dof_pos.unsqueeze(0),
            num_waypoints = 250 - step_cnt,
        )    

    if step_cnt < warm_up_steps:
        pass
    elif step_cnt < 100:        
        t = len(teacher_policy.path0) - (100 - step_cnt)
        actions[:, :7] = (teacher_policy.path0[t][:, :7] - default_dof_pos[:7])/env_cfg["action_scale"]
        actions[:, -2:] = 0
        
    elif step_cnt < 150:
        t = len(teacher_policy.path1) - (150 - step_cnt)
        actions[:, :7] = (teacher_policy.path1[t][:, :7] - default_dof_pos[:7])/env_cfg["action_scale"]
        actions[:, -2:] = 0
        
    elif step_cnt < 170: 
        actions[:, :7] = (teacher_policy.path1[-1][:, :7] - default_dof_pos[:7])/env_cfg["action_scale"]
        actions[:, -2:] = gripper_close_pos        

    elif step_cnt < 250:        
        t = len(teacher_policy.path2) - (250 - step_cnt)        
        actions[:, :7] = (teacher_policy.path2[t][:, :7] - default_dof_pos[:7])/env_cfg["action_scale"]
        actions[:, -2:] = gripper_close_pos
        
    return actions
