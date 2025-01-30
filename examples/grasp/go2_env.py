
import torch
import numpy as np
import math
import os
import copy
os.environ['PYOPENGL_PLATFORM'] = 'glx'

import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat, random_quaternion
from enum import Enum


class ControlType(Enum):
    DISCRETE = 0
    CARTESIAN = 1
    JOINT_POS = 2
    JOINT_VEL = 3
    JOINT_FORCE = 4

NUM_ACTIONS_OF_CTRL_TYPE = {
    ControlType.DISCRETE: 15,
    ControlType.CARTESIAN: 8,
    ControlType.JOINT_POS: 9,
    ControlType.JOINT_VEL: 9,
    ControlType.JOINT_FORCE: 9,
}    

def get_num_obs(obs_cfg, env_cfg):

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
    return num_obs


CUBE_SIZE = 0.04

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

class Go2Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, device, show_viewer=False, show_pinch_pos=False, verbose=False):
        self.device = torch.device(device)

        self.num_envs = num_envs
        self.num_links = env_cfg["num_links"]
        self.num_fingers = env_cfg["num_fingers"]
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]

        self._verbose = verbose

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequence on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg

        self.obs_scales = copy.deepcopy(obs_cfg["obs_scales"])
        self.reward_scales = copy.deepcopy(reward_cfg["reward_scales"])
        self.show_pinch_pos = show_pinch_pos

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=self.num_envs),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plain
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        self.object_init_pos = (0.6, 0.0, CUBE_SIZE/2.0)
        self.cube = self.scene.add_entity(
            gs.morphs.Box(
                size = (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
                pos  = self.object_init_pos,
            )
        )
        if self.show_pinch_pos:
            self.pinch_cube = self.scene.add_entity(
                gs.morphs.Box(
                    size = (0.01, 0.01, 0.01),
                    pos  = (0.65, 0.0, 0.05),
                    fixed = True,
                )
            )

        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
        )

        assert self.franka.n_links == self.num_links
        if env_cfg["control_type"] in [ControlType.JOINT_POS, ControlType.JOINT_VEL, ControlType.JOINT_FORCE]:
            assert self.franka.n_dofs == self.num_actions

        # build
        self.scene.build(n_envs=num_envs)

        self.arm_dofs = np.arange(7)
        self.finger_dofs = np.arange(7, 9)
        self.all_dofs = [self.franka.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]
        
        # Set control gains
        # Note: the following values are tuned for achieving best behavior with Franka
        # Typically, each new robot would have a different set of parameters.
        # Sometimes high-quality URDF or XML file would also provide this and will be parsed.
        self.franka.set_dofs_kp(
            np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
        )
        self.franka.set_dofs_kv(
            np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
        )
        self.franka.set_dofs_force_range(
            np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
        )

        self.end_effector = self.franka.get_link('hand')

        # set to pre-grasp pose
        qpos = self.env_cfg["default_joint_angles"]
        self.franka.set_dofs_position(qpos)
        self._resample_object_location(torch.arange(self.num_envs, device=self.device))        
        

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            if self._verbose:
                print (" !!!!!!! reward scale recaling", name, self.reward_scales[name])
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # initialize buffers
        # self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        # self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.finger_contact_force = torch.zeros((self.num_envs, self.num_fingers, 3), device=self.device, dtype=gs.tc_float)
        self.finger_pos = torch.zeros((self.num_envs, self.num_fingers, 3), device=self.device, dtype=gs.tc_float)
        self.links_lin_vel = torch.zeros((self.num_envs, self.num_links, 3), device=self.device, dtype=gs.tc_float)
        self.links_ang_vel = torch.zeros((self.num_envs, self.num_links, 3), device=self.device, dtype=gs.tc_float)
        self.links_projected_gravity = torch.zeros((self.num_envs, self.num_links, 3), device=self.device, dtype=gs.tc_float)
        
        self.object_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)

        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, self.num_links, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        
        self.dof_pos = torch.zeros((self.num_envs, self.franka.n_dofs), device=self.device, dtype=gs.tc_float)       
        self.dof_vel = torch.zeros_like(self.dof_pos)
        self.dof_force = torch.zeros_like(self.dof_pos)

        self.last_dof_vel = torch.zeros_like(self.dof_vel)        

        self.default_dof_pos = torch.tensor(            
            self.env_cfg["default_joint_angles"],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.extras = dict()  # extra information for logging    


        if self.env_cfg["control_type"] == ControlType.DISCRETE:
            self.action_id_to_6dof_vec = torch.tensor([
                [-1,0,0,0,0,0],
                [1,0,0,0,0,0],
                [0,-1,0,0,0,0],
                [0,1,0,0,0,0],
                [0,0,-1,0,0,0],
                [0,0,1,0,0,0],
                [0,0,0,-1,0,0],
                [0,0,0,1,0,0],
                [0,0,0,0,-1,0],
                [0,0,0,0,1,0],
                [0,0,0,0,0,-1],
                [0,0,0,0,0,1],
                [0,0,0,0,0,0],
                [0,0,0,0,0,0],
            ], device=self.device, dtype=gs.tc_float)

            self.action_id_to_finger_force = torch.tensor([
                [0,0],
                [0,0],
                [0,0],
                [0,0],
                [0,0],
                [0,0],
                [0,0],
                [0,0],
                [0,0],
                [0,0],
                [0,0],
                [0,0],
                [0.5, 0.5],
                [-0.5, -0.5],
            ], device=self.device, dtype=gs.tc_float)  


    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions

        if self.env_cfg["control_type"] == ControlType.DISCRETE:
            
            J = self.franka.get_jacobian(self.end_effector) # shape = [num_envs, 6, 9]
            inv_J = torch.linalg.pinv(J)
            action_logits = exec_actions[:, :-1]
            action_id = torch.argmax(action_logits, dim=-1)
            action_velocity = exec_actions[:, -1] * self.env_cfg["action_scale"]
            action_velocity = torch.maximum(action_velocity, torch.zeros_like(action_velocity))

            #print (self.action_id_to_6dof_vec.shape, action_id.shape, action_velocity.shape)
            end_effector_dof_vel = torch.stack(
                [
                    self.action_id_to_6dof_vec[_id]
                    for _id in action_id
                ],
            ) 
            #print (end_effector_dof_vel.shape)
            end_effector_dof_vel *= action_velocity.unsqueeze(1)

            joint_dof_vel = torch.bmm(inv_J, end_effector_dof_vel.unsqueeze(2)).squeeze(2)            
            finger_force = torch.stack(
                [
                    self.action_id_to_finger_force[_id]
                    for _id in action_id
                ],
            )
            self.franka.control_dofs_velocity(joint_dof_vel[:,:-2], self.arm_dofs)
            self.franka.control_dofs_force(finger_force, self.finger_dofs)
            
            
        elif self.env_cfg["control_type"] == ControlType.CARTESIAN:
            J = self.franka.get_jacobian(self.end_effector) # shape = [num_envs, 6, 9]
            inv_J = torch.linalg.pinv(J)
            end_effector_dof_vel = exec_actions[:, :-2] * self.env_cfg["action_scale"]
            joint_dof_vel = torch.bmm(inv_J, end_effector_dof_vel.unsqueeze(2)).squeeze(2)            
            self.franka.control_dofs_velocity(joint_dof_vel[:,:-2], self.arm_dofs)

            finger_force = exec_actions[:, -2:] * self.env_cfg["action_scale"]
            self.franka.control_dofs_force(finger_force, self.finger_dofs)
            
        elif self.env_cfg["control_type"] == ControlType.JOINT_POS:
            target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos            
            self.franka.control_dofs_position(target_dof_pos, self.all_dofs)
        
        elif self.env_cfg["control_type"] == ControlType.JOINT_VEL:
            target_dof_vel = exec_actions * self.env_cfg["action_scale"]
            self.franka.control_dofs_velocity(target_dof_vel, self.all_dofs)

        elif self.env_cfg["control_type"] == ControlType.JOINT_FORCE:
            target_dof_force = exec_actions * self.env_cfg["action_scale"]
            self.franka.control_dofs_force(target_dof_force, self.all_dofs)            

        self.scene.step()
        self.episode_length_buf += 1
        
        

        # update buffers
        links_vel = self.franka.get_links_vel()
        links_ang = self.franka.get_links_ang()
        links_quat = self.franka.get_links_quat()
        links_inv_quat = inv_quat(links_quat)
        self.finger_pos[:] = self.franka.get_links_pos()[:,-2:,:]
        self.finger_pos[:,:,-1] = self.finger_pos[:,:,-1] - 0.045
        self.pinch_pos = torch.mean(self.finger_pos, dim=1)
        #self.pinch_pos[:, -1] = self.pinch_pos[:, -1]

        if self.show_pinch_pos:
            self.pinch_cube.set_pos(
                self.pinch_pos, zero_velocity=True, envs_idx=torch.arange(self.num_envs, device=self.device)
            )

        
        self.links_lin_vel[:] = transform_by_quat(links_vel, links_inv_quat).view(
            self.num_envs, self.num_links, 3
        )
        self.links_ang_vel[:] = transform_by_quat(links_ang, links_inv_quat).view(
            self.num_envs, self.num_links, 3
        )
        self.links_projected_gravity[:] = transform_by_quat(self.global_gravity, links_inv_quat).view(
            self.num_envs, self.num_links, 3
        )

        contact_force = self.franka.get_links_net_contact_force()
        self.finger_contact_force[:] = contact_force[:,-2:, :]

        # self.base_pos[:] = self.robot.get_pos()
        # self.base_quat[:] = self.robot.get_quat()
        # self.base_euler = quat_to_xyz(
        #    transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        # )
        # inv_base_quat = inv_quat(self.base_quat)
        # self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        # self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        # self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)

        # Dofs
        self.dof_pos[:] = self.franka.get_dofs_position()
        self.dof_vel[:] = self.franka.get_dofs_velocity()
        self.dof_force[:] = self.franka.get_dofs_force()

        # Object
        self.object_pos[:] = self.cube.get_pos()

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        #self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        #self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf[:] = 0.0        
        for name, reward_func in self.reward_functions.items():
            # print ("eval reward", name)
            rew = reward_func()
            #if self.num_envs == 1:

            if self._verbose:
                print (f" ++ add reward {name}, mean= {torch.mean(rew)}, scale {self.reward_scales[name]}")

            rew *= self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        
        if self._verbose:
            print ("self.rew_buf.mean() / self.dt=", torch.mean(self.rew_buf) / self.dt)
            print ("self.rew_buf.shape=", self.rew_buf.shape)

        # compute observations
        n = self.num_envs
        obs_list = []        

        if self.obs_cfg["reach_dir_and_dist"]:
            obs_list.extend([
               self.object_pos - self.finger_pos[:,0,:],
               self.object_pos - self.finger_pos[:,1,:]
            ])
        if self.obs_cfg["finger_contact_force"]:
            obs_list.append(
                self.finger_contact_force.view(n, -1) * self.obs_scales["contact_force"], # 2x3
            )
        if self.obs_cfg["links_lin_vel"]:
            obs_list.append(
                self.links_lin_vel.view(n, -1) * self.obs_scales["lin_vel"],              # 11x3
            )
        if self.obs_cfg["links_ang_vel"]:
            obs_list.append(
                self.links_ang_vel.view(n, -1) * self.obs_scales["ang_vel"],              # 11x3
            )        
        if self.obs_cfg["links_projected_gravity"]:
            obs_list.append(
                self.links_projected_gravity.view(n, -1),                                 # 11x3
            )
        if self.obs_cfg["dof_pos"]:
            obs_list.append(
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],       # 9
            )
        if self.obs_cfg["dof_vel"]:
            obs_list.append(
                self.dof_vel * self.obs_scales["dof_vel"],                                # 9
            )
        if self.obs_cfg["dof_force"]:
            obs_list.append(
                self.dof_force * self.obs_scales["dof_force"],                            # 9
            )
        if self.obs_cfg["actions"]:
            obs_list.append(
                self.actions.view(n, -1),                                                 
            )

        self.obs_buf[:] = torch.cat(obs_list, axis=-1)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:] # not used?

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.dof_force[envs_idx] = 0.0
        self.franka.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.all_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        self._resample_object_location(envs_idx)        

        # reset base
        self.links_ang_vel[envs_idx] = 0
        self.links_lin_vel[envs_idx] = 0
        
        #self.base_pos[envs_idx] = self.base_init_pos
        #self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        #self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        #self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        #self.base_lin_vel[envs_idx] = 0
        #self.base_ang_vel[envs_idx] = 0
        
        self.franka.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

    def _resample_object_location(self, envs_idx):
            
        obj_pos = torch.tensor(self.object_init_pos, device=self.device, dtype=gs.tc_float).repeat(
            len(envs_idx), 1
        )
        offset = torch.randn(len(envs_idx), 3).to(self.device)
        offset[:,0] *= 0.01
        offset[:,1] *= 0.1
        offset[:,-1] = 0
        self.cube.set_pos(
            obj_pos + offset,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        self.cube.set_quat(
            torch.from_numpy(random_quaternion(len(envs_idx))).to(self.device),
            zero_velocity=True, 
            envs_idx=envs_idx,
        )


    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    # ------------ reward functions----------------    

    def _reward_action_rate(self):
        # Penalize changes in actions
        reward = torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        # print ("action rate reward=", reward)
        return reward

    def _reward_lift_height(self):
        object_height = self.object_pos[:,2] - CUBE_SIZE/2.0
        reward = object_height / self.reward_cfg["target_height"]
        reward = torch.minimum(reward,  torch.ones_like(reward))

        if self._verbose:
            print ("-------------------------")
            print (f"lift height (mean / max)={torch.mean(object_height)} / {torch.max(object_height)}")            
            print (f"lift reward (mean / max)={torch.mean(reward)} / {torch.max(reward)}")            
        
        return reward
    
    def _reward_obj_dist(self):                
        dist = torch.norm(self.pinch_pos - self.object_pos, dim=1)
        dist_reward = torch.maximum(1.0 - 5 * dist, torch.zeros_like(dist))
        #if self.num_envs == 1:

        if self._verbose:
            print (f"dist (mean / min)={torch.mean(dist)} / {torch.min(dist)}")            
            print (f"dist reward (mean / max)={torch.mean(dist_reward)} / {torch.max(dist_reward)}")            
        
        return dist_reward
    
    def _reward_finger_pressure(self):
        # TODO measure contact force at inside of finger
        reward = torch.sum(torch.abs(self.finger_contact_force), dim=(1,2))
        # print ("finger pressure reward=", reward)
        return reward

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
