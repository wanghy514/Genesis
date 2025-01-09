from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

from rsl_rl.runners.legacy_runner import LeggedGymRunner
from rsl_rl.env import VecEnv

class RunnerWithTB(LeggedGymRunner):

    def __init__(self,
            env: VecEnv,
            train_cfg,
            log_dir=None,
            device='cpu'):
        
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        
        self._learn_cb.append(RunnerWithTB.tb_log)

        self.log_dir = log_dir
        self.writer = None
        
    def learn(self, *args, **kwargs):
        if self.log_dir is not None and self.writer is None:
            self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)        

        super().learn(*args, **kwargs)

    def tb_log(self, stat):        
        mean_reward = sum(stat["returns"]) / len(stat["returns"]) if len(stat["returns"]) > 0 else 0.0
        self.writer.add_scalar("Train/mean_reward", mean_reward, stat["current_iteration"])        

    