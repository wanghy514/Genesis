import torch
import numpy as np
import math
import os
os.environ['PYOPENGL_PLATFORM'] = 'glx'

import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class Go2Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, device, show_viewer=False):
        self.device = torch.device(device)

        self.num_envs = num_envs
        self.num_links = env_cfg["num_links"]
        self.num_fingers = env_cfg["num_fingers"]
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequence on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
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

        self.object_init_pos = (0.65, 0.0, 0.02)
        self.cube = self.scene.add_entity(
            gs.morphs.Box(
                size = (0.04, 0.04, 0.04),
                pos  = self.object_init_pos,
            )
        )        

        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
        )

        assert self.franka.n_links == self.num_links
        assert self.franka.n_dofs == self.num_actions

        # add robot
        # self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        # self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        # self.inv_base_init_quat = inv_quat(self.base_init_quat)
        # self.robot = self.scene.add_entity(
        #     gs.morphs.URDF(
        #         file="urdf/go2/urdf/go2.urdf",
        #         pos=self.base_init_pos.cpu().numpy(),
        #         quat=self.base_init_quat.cpu().numpy(),
        #     ),
        # )

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motor_dofs = [self.franka.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]
        # self.robot_motors_dof = np.arange(7)
        # self.robot_fingers_dof = np.arange(7, 9)

        # PD control parameters
        # self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        # self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motor_dofs)
        # set control gains
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

        # set to pre-grasp pose
        qpos = self.env_cfg["default_joint_angles"]
        self.franka.set_dofs_position(qpos)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
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

        
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.dof_force = torch.zeros_like(self.actions)

        self.last_dof_vel = torch.zeros_like(self.actions)

        #self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        #self.link_pos = torch.zeros((self.num_envs, self.num_links, 3), device=self.device, dtype=gs.tc_float)
        #self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        #self.link_quat = torch.zeros((self.num_envs, self.num_links, 4), device=self.device, dtype=gs.tc_float)

        self.default_dof_pos = torch.tensor(
            #[self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            self.env_cfg["default_joint_angles"],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.extras = dict()  # extra information for logging    

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos

        self.franka.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        links_vel = self.franka.get_links_vel()
        links_ang = self.franka.get_links_ang()
        links_quat = self.franka.get_links_quat()
        links_inv_quat = inv_quat(links_quat)
        self.finger_pos[:] = self.franka.get_links_pos()[:,-2:,:]
        
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
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        n = self.num_envs
        obs_list = []
        
        # print ("object_pos=", self.object_pos)
        # print ("finger_pos=", self.finger_pos)

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
                self.actions.view(n, -1),                                                 # 9
            )

        self.obs_buf[:] = torch.cat(obs_list, axis=-1)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:] # not used?

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_buf

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
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        self.cube.set_pos(
            torch.tensor(self.object_init_pos, device=self.device, dtype=gs.tc_float).repeat(
                len(envs_idx), 1
            ),
            zero_velocity=True,
            envs_idx=envs_idx,
        )

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

        #self._resample_commands(envs_idx) # TODO resample object location

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
        object_height = self.object_pos[:,2]
        reward = torch.minimum(object_height, self.reward_cfg["target_height"] * torch.ones_like(object_height))
        # print ("lift height reward=", reward)
        return reward 
    
    def _reward_obj_inv_dist(self):        
        dist0 = torch.norm(self.finger_pos[:,0,:] - self.object_pos, dim=1)
        dist1 = torch.norm(self.finger_pos[:,1,:] - self.object_pos, dim=1)
        inv_dist = 0.01 / (dist0 + dist1 + 1e-10)        
        return torch.maximum(inv_dist, self.reward_cfg["inv_dist_bound"] * torch.ones_like(inv_dist))
    
    def _reward_finger_pressure(self):
        # TODO measure contact force at inside of finger
        reward = torch.sum(torch.abs(self.finger_contact_force), dim=(1,2))
        # print ("finger pressure reward=", reward)
        return reward

