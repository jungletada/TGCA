
import torch


class PolyOptimizer(torch.optim.SGD):
    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr, weight_decay)
        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum
        self.__initial_lr = [group['lr'] for group in self.param_groups]

    def step(self, closure=None):
        if self.global_step < (0.5*self.max_step):
            lr_mult = (1 - self.global_step / (0.5*self.max_step)) ** self.momentum
            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        elif self.global_step < self.max_step:
             lr_mult = (1 - (self.global_step-(0.5*self.max_step)) / (self.max_step-(0.5*self.max_step))) ** self.momentum
             for i in range(len(self.param_groups)):
                 self.param_groups[i]['lr'] = 0.0007 * lr_mult
        super().step(closure)

        self.global_step += 1


class PolyOptimizer_cls(torch.optim.SGD):
    def __init__(self, params, lr, weight_decay, max_step, warmup_step=1500, momentum=0.9):
        super().__init__(params, lr, weight_decay)
        self.global_step = 0
        self.max_step = max_step
        self.warmup_step = warmup_step
        self.momentum = momentum
        self.__initial_lr = [group['lr'] for group in self.param_groups]
        assert max_step > warmup_step, f"max_step {max_step} should larger than warmup_step {warmup_step}"
        
    def step(self, closure=None):
        if self.global_step < self.warmup_step:
            lr_mult = self.global_step / self.warmup_step
        elif self.warmup_step <= self.global_step < self.max_step:
            scale = (self.global_step - self.warmup_step) / (self.max_step - self.warmup_step)
            lr_mult = (1 - scale) ** self.momentum
            
        for i in range(len(self.param_groups)):
            if i == 4:
                self.param_groups[i]['lr'] = self.__initial_lr[i]
            else:
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)
        self.global_step += 1


class PolyOptimizerAdamW(torch.optim.AdamW):
    def __init__(self, params, lr, max_step, warmup_step=1500, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, momentum=0.9):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.global_step = 0
        self.max_step = max_step
        self.warmup_step = warmup_step
        self.momentum = momentum
        self.__initial_lr = [group['lr'] for group in self.param_groups]

    def step(self, closure=None):
        if self.global_step < self.warmup_step:
            lr_mult = self.global_step / self.warmup_step
        elif self.warmup_step <= self.global_step < self.max_step:
            scale = (self.global_step - self.warmup_step) / (self.max_step - self.warmup_step)
            lr_mult = (1 - scale) ** self.momentum
            
        for i in range(len(self.param_groups)):
            if i == 4:
                self.param_groups[i]['lr'] = self.__initial_lr[i]
            else:
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult
        super().step(closure)
        self.global_step += 1