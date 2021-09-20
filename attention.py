import torch.nn as nn
import torch


class AttentionModel(nn.Module):
    def __init__(self, dim, num_heads, include_bias, attention_dropout=0.1, projection_dropout=0.1):
        super(AttentionModel, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.include_bias = include_bias
        self.attention_dropout = attention_dropout
        self.projection_dropout = projection_dropout

        self.head_dim = dim // num_heads

        self.values = nn.Linear(self.dim, self.dim)
        self.queries = nn.Linear(self.dim, self.dim)
        self.keys = nn.Linear(self.dim, self.dim)

        self.projection = nn.Linear(dim, dim)
        self.attention_drop = nn.Dropout(self.attention_dropout)
        self.projection_drop = nn.Dropout(self.projection_dropout)

    def forward(self, x):
        # shape(32, num_patches+1, embedding_dim) (32,257,768)
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
        # shape query, key, value (32,12,257,64)

        key_transpose = keys.transpose(-2, -1)  # (num_samples, num_heads, head_dim, n_patches + 1)
        score = (queries @ key_transpose) / math.sqrt(self.head_dim)
        attention = Softmax(dim=-1)(score) # shape(n_samples, n_heads, n_patches + 1, n_patches +1)
        attention = self.attention_drop(attention)
        weighted_average = attention @ values # shape(32,12,257,64)
        weighted_average_transpose = weighted_average.transpose(1, 2) # shape(32,257,12,64)
        weighted_average_flat = weighted_average_transpose.flatten(2) # shape(32,257,768)
        output = self.projection(weighted_average_flat) # shape(32,257,768)
        output = self.projection_drop(output)
        return output


if __name__ == '__main__':
    x = torch.randn(32, 257, 768)
    attention = AttentionModel(768, 12, True)
    x = attention(x)
    print(x)
    # (32,257,768)
