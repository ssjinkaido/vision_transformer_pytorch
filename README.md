# vision_transformer_pytorch

The paper of vision transformer is here: https://arxiv.org/pdf/2010.11929.pdf \
Most of the code are borrowed from here : https://github.com/YousefGamal220/Vision-Transformers \
I try to re implement the multiheaded attention since it the code I borrowed is a bit hard to understand, also I try to rewrite the train function to make it looks cleaner

## Project structure

The project has the following structure:
- `main.py` file used to train and validate
- `train.py` file contains training and validation functions
- `prepare_dataset.py` file used to download dataset(CIFAR10) and load them into DataLoader
- `vision_transformer.py` file contains vision transformer model