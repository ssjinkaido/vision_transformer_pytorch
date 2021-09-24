from vision_transformer import VisionTransformer
from train import train, validate
from prepare_dataset import train_loader, val_loader
import torch
from tqdm import tqdm
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCHS = 50
LEARNING_RATE = 1e-3
vision_transformer = VisionTransformer(image_size=32, num_classes=10)


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main():
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    model = vision_transformer.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3,
                                                           threshold=0.0001, threshold_mode='abs', min_lr=1e-8,
                                                           eps=1e-08)
    for i in tqdm(range(EPOCHS)):
        train(model, i, train_loader, optimizer, device, scaler)
        avg_test_loss = validate(model, val_loader, scheduler, device)
        scheduler.step(avg_test_loss)


if __name__ == '__main__':
    seed_everything(42)
    main()
