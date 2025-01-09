import pickle

# Create cfgs
from go2_train import env_cfg, obs_cfg, reward_cfg, train_cfg

exp_name = "learn2grasp"


with open(f"logs/{exp_name}/cfgs.pkl", "wb") as fp:
    pickle.dump((env_cfg, obs_cfg, reward_cfg, train_cfg), fp)

print ("written to ", f"logs/{exp_name}/cfgs.pkl")

