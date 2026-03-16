from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import List
class MultiStageOneCycleLR(_LRScheduler):
    def __init__(self, optimizer: Optimizer, total_steps: int, warmup_ratio: float, cooldown_ratio: float, last_epoch: int = -1, verbose: bool = False):
        # 计算各个阶段的步数
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.cooldown_steps = int(total_steps * cooldown_ratio)
        
        # 计算保持阶段的步数
        self.constant_steps = total_steps - self.warmup_steps - self.cooldown_steps
        
        # 进行合法性检查
        if self.constant_steps < 0:
            raise ValueError("预热比例和退火比例的总和不能超过1。")

        # 计算退火阶段的起始步
        self.cooldown_start_step = self.warmup_steps + self.constant_steps

        # 新增状态：用于手动触发退火
        self.cooldown_triggered = False
        self.cooldown_trigger_step = None

        super().__init__(optimizer, last_epoch, verbose)

    def trigger_cooldown(self):
        if not self.cooldown_triggered:
            new_total_steps = self.last_epoch + self.cooldown_steps
            if self.verbose:
                print(f"INFO: Cooldown manually triggered at step {self.last_epoch}.")
                print(f"INFO: Original total_steps was {self.total_steps}. New total_steps is {new_total_steps}.")
            
            self.cooldown_triggered = True
            self.cooldown_trigger_step = self.last_epoch
            # 关键改动：更新总步数
            self.total_steps = new_total_steps

    def get_lr(self) -> List[float]:
        current_step = self.last_epoch

        if self.cooldown_triggered:
            manual_cooldown_duration = self.total_steps - self.cooldown_trigger_step

            step_in_manual_cooldown = current_step - self.cooldown_trigger_step

            if manual_cooldown_duration <= 0:
                return [0.0 for _ in self.base_lrs]
            
            cooldown_progress = min(float(step_in_manual_cooldown) / float(manual_cooldown_duration), 1.0)
            cooldown_factor = 0.5 * (1.0 + math.cos(math.pi * cooldown_progress))
            
            return [base_lr * cooldown_factor for base_lr in self.base_lrs]

        if self.warmup_steps > 0 and current_step < self.warmup_steps:
            warmup_factor = float(current_step + 1) / float(self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        elif current_step >= self.cooldown_start_step:
            step_in_cooldown = current_step - self.cooldown_start_step
            
            if self.cooldown_steps == 0:
                cooldown_progress = 1.0
            else:
                cooldown_progress = min(float(step_in_cooldown) / float(self.cooldown_steps), 1.0)
            
            cooldown_factor = 0.5 * (1.0 + math.cos(math.pi * cooldown_progress))
            return [base_lr * cooldown_factor for base_lr in self.base_lrs]

        else:
            return [base_lr for base_lr in self.base_lrs]