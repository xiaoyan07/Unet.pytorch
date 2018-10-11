import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(data_loader, model_fn, loss_fn, optimizer, lr_schedule):

    for idx, (inputs, targets) in enumerate(data_loader):
        inputs.to(device)
        targets.to(device)

        optimizer.zero_grad()

        outputs = model_fn(inputs)

        total_loss = loss_fn(outputs, targets)

        total_loss.backward()

        optimizer.step()
        lr_schedule.step()
