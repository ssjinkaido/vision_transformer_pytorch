from vision_transformer import VisionTransformer
from train import train, validate
from prepare_dataset import train_loader, val_loader
import torch
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCHS = 50
LEARNING_RATE = 1e-3
vision_transformer = VisionTransformer(image_size=32, num_classes=10)


def main():
    for i in tqdm(range(EPOCHS)):
        model = vision_transformer.to(device)
        train(model, i, LEARNING_RATE, train_loader, device)
        validate(model, val_loader, device)

if __name__ == '__main__':
    main()
