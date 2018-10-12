import torch
from util import checkpoint_util as ckpt_utils
import os


class TrainPack(object):
    def __init__(self,
                 model_dir,
                 model,
                 loss,
                 optimizer,
                 lr_schedule,
                 save_per_steps=1000,
                 use_gpu=True):
        self.model = model
        self.loss = loss
        self.opt = optimizer
        self.lr = lr_schedule

        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        # todo: support checkpoint
        self.model_dir = model_dir
        self._save_per_steps = save_per_steps
        self._global_step = ckpt_utils.GlobalStep()

    def train(self, data_loader):
        """ train model by DataLoader

        Args:
            data_loader: a instance of torch.utils.data.DataLoader

        Returns:

        """
        for idx, (inputs, targets) in enumerate(data_loader):
            self.train_step(inputs, targets)

    def train_step(self, inputs, targets):
        self.zero_grad()
        self.compute_gradients(inputs, targets)
        self.apply_gradients()

    def train_in_multi_grads(self, data_loader, multiple=2):
        data_iterator = iter(data_loader)
        # todo
        while True:
            self.zero_grad()
            for i in range(multiple):
                data = next(data_iterator)
                inputs, targets = data
                self.compute_gradients(inputs, targets)
            self.apply_gradients()

    def compute_gradients(self, inputs, targets):
        self.model.train()
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        outputs = self.model(inputs)

        total_loss = self.loss(outputs, targets)

        total_loss.backward()

    def apply_gradients(self):
        """ apply gradients and update the state of optimizer, lr and global step
            clear gradients obtained before
        Returns:

        """
        self.opt.step()
        self.lr.step()

        self.opt.zero_grad()
        self._global_step.step()
        self.save_checkpoint_hook()

    def zero_grad(self):
        self.opt.zero_grad()

    def checkpoint_name(self, global_step):
        return 'model-{}.pth'.format(global_step)

    def save_checkpoint_hook(self):
        if self._global_step.value % self._save_per_steps == 0:
            trainstate = ckpt_utils.TrainState.from_instances(model=self.model,
                                                              optimizer=self.opt,
                                                              global_step=self._global_step)

            ckpt = ckpt_utils.CheckPoint(os.path.join(self.model_dir, self._global_step.value))
            ckpt.write(trainstate.state_dict())
