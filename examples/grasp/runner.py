import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.algorithms import PPO, DDPG
# from rsl_rl.modules import ActorCritic
from rsl_rl.modules.actor_critic import get_activation
#from rsl_rl.runners import OnPolicyRunner
from rsl_rl.runners.legacy_runner import LeggedGymRunner
from rsl_rl.env import VecEnv


# class ResNet(nn.Module):

#     def __init__(self, mlp_input_dim_a, hidden_dims, activation, num_actions, shortcut=True):
#         super(ResNet, self).__init__()
#         self.shortcut = shortcut
#         _layers = []
#         _layers.append(nn.Linear(mlp_input_dim_a, hidden_dims[0]))
#         _layers.append(activation)
#         for layer_index in range(len(hidden_dims)):
#             if layer_index == len(hidden_dims) - 1:
#                 _layers.append(nn.Linear(hidden_dims[layer_index], num_actions))
#             else:
#                 _layers.append(nn.Linear(hidden_dims[layer_index], hidden_dims[layer_index + 1]))
#                 _layers.append(activation)
#         self.layers = nn.Sequential(*_layers)

#         if shortcut:
#             self.linear = nn.Linear(mlp_input_dim_a, num_actions)
    
#     def forward(self, x):
#         if self.shortcut:
#             return self.layers(x) + self.linear(x)
#         else:
#             return self.layers(x)
    

# class ResActorCritic(ActorCritic):

#    def __init__(self,  num_actor_obs,
#                         num_critic_obs,
#                         num_actions,
#                         actor_hidden_dims=[256, 256, 256],
#                         critic_hidden_dims=[256, 256, 256],
#                         activation='elu',                        
#                         init_noise_std=1.0,
#                         actor_shortcut=False,
#                         critic_shortcut=False,
#                         **kwargs):
#         if kwargs:
#             print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
#         super(ActorCritic, self).__init__()

#         activation = get_activation(activation)

#         mlp_input_dim_a = num_actor_obs
#         mlp_input_dim_c = num_critic_obs

#         # Policy
#         self.actor = ResNet(mlp_input_dim_a, actor_hidden_dims, activation, num_actions, shortcut=actor_shortcut)

#         # Value function
#         self.critic = ResNet(mlp_input_dim_c, critic_hidden_dims, activation, 1, shortcut=critic_shortcut)

#         print(f"Actor MLP: {self.actor}")
#         print(f"Critic MLP: {self.critic}")

#         # Action noise
#         self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
#         self.distribution = None
#         # disable args validation for speedup
#         Normal.set_default_validate_args = False


class Runner(LeggedGymRunner):

    pass
    
    # def __init__(self,
    #         env: VecEnv,
    #         train_cfg,
    #         log_dir=None,
    #         device='cpu'):

    #     self.cfg=train_cfg["runner"]
    #     self.alg_cfg = train_cfg["algorithm"]
    #     self.policy_cfg = train_cfg["policy"]
    #     self.device = device
    #     self.env = env
    #     if self.env.num_privileged_obs is not None:
    #         num_critic_obs = self.env.num_privileged_obs 
    #     else:
    #         num_critic_obs = self.env.num_obs

        
    #     alg_class = eval(self.cfg["algorithm_class_name"])

    #     if True:
    #         # when using algorithm branch, actor critic networks are created inside alg_class()
    #         self.alg = alg_class(env, device=self.device, **self.alg_cfg)
    #     else:            
    #         actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
    #         actor_critic: ActorCritic = actor_critic_class( self.env.num_obs,
    #                                                         num_critic_obs,
    #                                                         self.env.num_actions,
    #                                                         **self.policy_cfg).to(self.device)            
    #         self.alg = alg_class(actor_critic, device=self.device, **self.alg_cfg)
    #     self.num_steps_per_env = self.cfg["num_steps_per_env"]
    #     self.save_interval = self.cfg["save_interval"]

    #     # init storage and model
    #     self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])

    #     # Log
    #     self.log_dir = log_dir
    #     self.writer = None
    #     self.tot_timesteps = 0
    #     self.tot_time = 0
    #     self.current_learning_iteration = 0

    #     _, _ = self.env.reset()
