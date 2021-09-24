import torch.optim as optim
import torch.nn as nn
import time
import torch


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(model, epoch, learning_rate, train_loader, device):
    optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    end = time.time()
    for batch, (x, y) in enumerate(train_loader):

        data_time.update(time.time() - end)
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        output = model(x)

        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        losses.update(loss.item(), x.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if batch % 2500 == 0:
            print(
                f'Epoch {epoch}, [{batch}/{len(train_loader)}], Time {batch_time.val:.3f} ({batch_time.avg:.3f}), Data {data_time.val:.3f} ({data_time.avg:.3f}), Loss {losses.val:.4f} ({losses.avg:.4f})')

    return losses


def validate(model, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for batch, (x, y) in enumerate(val_loader):
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            loss = criterion(output, y)

            output = output.float()
            loss = loss.float()

            losses.update(loss.item(), x.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch % 1250 == 0:
                print(
                    f'[{batch}/{len(val_loader)}], Time {batch_time.val:.3f} ({batch_time.avg:.3f}), Loss {losses.val:.4f} ({losses.avg:.4f})')
