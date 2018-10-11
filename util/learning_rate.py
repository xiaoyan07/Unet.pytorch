import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MultiStepLR(object):
    def __init__(self, optimizer, steps, gamma):
        self.optimizer = optimizer
        self.steps = steps
        self.gamma = gamma

    def step(self, global_step):
        if global_step.value not in self.steps:
            return

        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']
            new_lr = lr * self.gamma
            param_group['lr'] = new_lr
            logging.info('step: {}\t'
                         'lr: {} -> {}'.format(global_step.value, lr, new_lr))
