import pickle

# Create cfgs
from go2_train import get_cfgs, parse_args

args = parse_args()
env_cfg, obs_cfg, reward_cfg, train_cfg = get_cfgs(args)

with open(f"logs/{args.exp_name}/cfgs.pkl", "wb") as fp:
    pickle.dump((env_cfg, obs_cfg, reward_cfg, train_cfg), fp)

print ("written to ", f"logs/{args.exp_name}/cfgs.pkl")

