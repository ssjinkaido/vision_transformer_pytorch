import torch
import torch.nn as nn
import math


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
        projection = projection.flatten(2)  # shape(batches, embedding_dims, n_patches
        projection = projection.transpose(1, 2)  # shape (n_samples, n_patches, embedding_dim)
        return projection


class AttentionModel(nn.Module):
    def __init__(self, dim, num_heads, include_bias, attention_dropout=0.1, projection_dropout=0.1):
        super(AttentionModel, self).__init__()
        '''
            dim: Input/Output dimensions
            num_heads: number of heads of the attention
            include_bias: bool variable to include bias or not for query, key, and value of the attention
            attention_dropout: probability of dropout for the attention 
            projection_dropout: robability of dropout for the projection (Patch Embedding Layer)
        '''
        self.dim = dim
        self.num_heads = num_heads
        self.include_bias = include_bias
        self.attention_dropout = attention_dropout
        self.projection_dropout = projection_dropout

        self.head_dim = self.dim // self.num_heads

        self.values = nn.Linear(self.dim, self.dim)
        self.queries = nn.Linear(self.dim, self.dim)
        self.keys = nn.Linear(self.dim, self.dim)

        self.linear_layer = nn.Linear(self.dim, self.dim * 3, bias=self.include_bias)
        self.projection = nn.Linear(self.dim, self.dim)
        self.attention_drop = nn.Dropout(self.attention_dropout)
        self.projection_drop = nn.Dropout(self.projection_dropout)

    def forward(self, x):
        # shape(32, num_patches+1, embedding_dim) (32,577,768)
        N = x.shape[0]

        queries = self.queries(x)
        keys = self.keys(x)
        values = self.values(x)

        values = values.reshape(N, -1, self.num_heads, self.head_dim)
        queries = queries.reshape(N, -1, self.num_heads, self.head_dim)
        keys = keys.reshape(N, -1, self.num_heads, self.head_dim)

        values = values.transpose(2, 1)
        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)

        key_transpose = keys.transpose(-2, -1)  # (num_samples, num_heads, head_dim, n_patches + 1)
        score = (queries @ key_transpose) / math.sqrt(self.head_dim)
        attention = nn.Softmax(dim=-1)(score)  # shape(n_samples, n_heads, n_patches + 1, n_patches +1)
        attention = self.attention_drop(attention)
        weighted_average = attention @ values  # shape(32,12,577,64)
        weighted_average_transpose = weighted_average.transpose(1, 2)  # shape(32,577,12,64)
        weighted_average_flat = weighted_average_transpose.flatten(2)  # shape(32,577,768)
        output = self.projection(weighted_average_flat)  # shape(32,577,768)
        output = self.projection_drop(output)
        return output


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout_p=0.1):
        super(MLP, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.layer1 = nn.Linear(self.in_features, self.hidden_features)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(self.hidden_features, self.out_features)
        self.drop = nn.Dropout(dropout_p)

    def forward(self, x):
        linear1 = self.layer1(x)
        gelu = self.gelu(linear1)
        gelu = self.drop(gelu)
        linear2 = self.linear2(gelu)
        output = self.drop(linear2)
        return output


class Block(nn.Module):
    def __init__(self, dim, num_heads, include_bias=True, projection_dropout=0.1, attention_dropout=0.1):
        super(Block, self).__init__()
        self.attention = AttentionModel(dim, num_heads, include_bias, attention_dropout, projection_dropout)
        self.mlp = MLP(dim, dim * 4, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        x = x + self.norm1(self.attention(x))
        x = x + self.norm2(self.mlp(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, image_size=32, patch_size=2, input_channels=3, num_classes=10, embedding_dims=768, depth=12,
                 num_heads=12, include_bias=True, dropout_p=0.1, attention_p=0.1):
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
