import torch
import torch.nn as nn
from patch_embeddings import PatchEmbedding
from block import Block


class VisionTransformer(nn.Module):
    def __init__(self, image_size=384, patch_size=16, input_channels=3, num_classes=10, embedding_dims=1024, depth=24,
                 num_heads=16, include_bias=True, dropout_p=0.5, attention_p=0.5):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, input_channels,
                                              embedding_dims)
        self.cls = nn.Parameter(torch.zeros(1, 1, embedding_dims))
        self.positional_embeddings = nn.Parameter(torch.zeros(1, 1 + self.patch_embedding.number_of_patches,
                                                              embedding_dims))
        self.pos_drop = nn.Dropout(dropout_p)
        self.blocks = nn.ModuleList(
            [Block(embedding_dims, num_heads, include_bias, dropout_p, attention_p) for _ in range(depth)])
        self.norm = nn.LayerNorm(embedding_dims, eps=1e-6)
        self.head = nn.Linear(embedding_dims, num_classes)

    def forward(self, x):
        n_samples = x.shape[0]
        x = self.patch_embedding(x)
        cls = self.cls.expand(n_samples, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.positional_embeddings

        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)

        cls_final = x[:, 0]

        x = self.head(cls_final)

        return x
