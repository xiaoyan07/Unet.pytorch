import torch


class TrainPack(object):
    def __init__(self,
                 model,
                 loss,
                 optimizer,
                 lr_schedule,
                 use_gpu=True):
        self.model = model
        self.loss = loss
        self.opt = optimizer
        self.lr = lr_schedule

        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        # todo: support checkpoint

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

        while True:
            self.zero_grad()
            for i in range(multiple):
                data = next(data_iterator)
                inputs, targets = data
                self.compute_gradients(inputs, targets)
            self.apply_gradients()

    def compute_gradients(self, inputs, targets):
        self.model.train()
        inputs.to(self.device)
        targets.to(self.device)

        outputs = self.model(inputs)

        total_loss = self.loss(outputs, targets)

        total_loss.backward()

    def apply_gradients(self):
        self.opt.step()
        self.lr.step()

        self.opt.zero_grad()

    def zero_grad(self):
        self.opt.zero_grad()
