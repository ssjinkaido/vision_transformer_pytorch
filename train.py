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


def train(model, epoch, train_loader, optimizer, device, scaler):
    criterion = nn.CrossEntropyLoss()
    batch_time = AverageMeter()
    losses = AverageMeter();
    model.train()
    end = time.time()
    train_loss = 0
    correct = 0
    total = 0
    for batch, (x, y) in enumerate(train_loader):

        x = x.to(device)
        y = y.to(device)
        batch_size = y.size(0)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            output = model(x)
            loss = criterion(output, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # loss.backward()
        # optimizer.step()
        losses.update(loss.item(), batch_size)
        _, predicted = output.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

        batch_time.update(time.time() - end)
        end = time.time()

        if batch % 1200 == 0 and batch != 0:
            #           print(f'Epoch {epoch}, [{batch}/{len(train_loader)}], Time {batch_time.avg:.3f}, Loss {losses.avg:.4f}, Acc {accuracies.avg:.4f}')
            print(
                f'Epoch {epoch}, [{batch}/{len(train_loader)}], Time {batch_time.avg:.3f}, Loss {(losses.avg):.4f}, Acc {((correct / total) * 100):.4f}')


def validate(model, val_loader, scheduler, device):
    criterion = nn.CrossEntropyLoss()
    batch_time = AverageMeter()
    losses = AverageMeter()
    model.eval()
    correct = 0
    total = 0
    end = time.time()
    with torch.no_grad():
        for batch, (x, y) in enumerate(val_loader):
            x = x.to(device)
            y = y.to(device)
            batch_size = y.size(0)
            output = model(x)
            loss = criterion(output, y)

            losses.update(loss.item(), batch_size)
            _, predicted = output.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch % 1200 == 0 and batch != 0:
                print(
                    f'[{batch}/{len(val_loader)}], Time {batch_time.avg:.3f}, Loss {(losses.avg):.4f}, Acc {(correct / total) * 100:.4f}')

    return losses.avg
