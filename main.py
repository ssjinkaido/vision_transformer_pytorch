from vision_transformer import VisionTransformer
from train import train, validate
from prepare_dataset import train_loader, val_loader
import torch
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCHS = 50
LEARNING_RATE = 1e-3
vision_transformer = VisionTransformer(image_size=32, num_classes=10)

def seed_everything(seed):
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
    for i in tqdm(range(EPOCHS)):
        model = vision_transformer.to(device)
        train(model, i, LEARNING_RATE, train_loader, device)
        validate(model, val_loader, device)

if __name__ == '__main__':
    seed_everything(42)
    main()
