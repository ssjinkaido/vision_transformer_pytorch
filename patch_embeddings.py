import torch.nn as nn
import torch


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, input_channels=3, embedding_dims=768):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.embedding_dims = embedding_dims
        self.number_of_patches = (self.image_size // self.patch_size) ** 2
        self.projection = nn.Conv2d(self.input_channels, self.embedding_dims, kernel_size=self.patch_size,
                                    stride=self.patch_size)

    def forward(self, x):
        # x_shape = (batches, input_channels, image_size, image_size)
        projection = self.projection(x)  # (batches, embedding_dims, sqrt(n_patches), sqrt(n_patches))
        print(projection.shape)
        projection = projection.flatten(2)  # shape(batches, embedding_dims, n_patches
        print(projection.shape)
        projection = projection.transpose(1, 2)  # shape (n_samples, n_patches, embedding_dim)
        print(projection.shape)
        return projection


if __name__ == '__main__':
    x = torch.randn(32, 3, 384, 384)
    patch_embedding = PatchEmbedding(768, 16, 3, 768)
    x = patch_embedding(x)
    # (32,577,768)
