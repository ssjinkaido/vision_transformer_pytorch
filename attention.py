import torch.nn as nn
import torch


class AttentionModel(nn.Module):
    def __init__(self, dim, num_heads, include_bias, attention_dropout=0.5, projection_dropout=0.5):
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

        self.head_dim = dim // num_heads

        self.values = nn.Linear(self.head_dim, self.head_dim)
        self.queries = nn.Linear(self.head_dim, self.head_dim)
        self.keys = nn.Linear(self.head_dim, self.head_dim)

        self.scale = self.head_dim ** -0.5
        self.linear_layer = nn.Linear(dim, dim * 3, bias=include_bias)
        self.projection = nn.Linear(dim, dim)
        self.attention_drop = nn.Dropout(self.attention_dropout)
        self.projection_drop = nn.Dropout(self.projection_dropout)

    def forward(self, queries, keys, values):
        # shape(32, num_patches+1, embedding_dim) (32,577,768)
        N = queries.shape[0]

        values = values.reshape(N, -1, self.num_heads, self.head_dim)
        queries = queries.reshape(N, -1, self.num_heads, self.head_dim)
        keys = keys.reshape(N, -1, self.num_heads, self.head_dim)

        values = self.values(values)
        queries = self.queries(queries)
        keys = self.keys(keys)

        values = values.transpose(2, 1)
        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        # shape query, key, value (32,12,577,64)

        key_transpose = keys.transpose(-2, -1)  # (num_samples, num_heads, head_dim, n_patches + 1)
        score = (queries @ key_transpose) * self.scale
        score = score/ (self.head_dim **(1/2))
        attention = score.softmax(dim=-1) # shape(n_samples, n_heads, n_patches + 1, n_patches +1)
        attention = self.attention_drop(attention)
        weighted_average = attention @ values # shape(32,12,577,64)
        weighted_average_transpose = weighted_average.transpose(1, 2) # shape(32,577,12,64)
        weighted_average_flat = weighted_average_transpose.flatten(2) # shape(32,577,768)
        output = self.projection(weighted_average_flat) # shape(32,577,768)
        output = self.projection_drop(output)
        return output


if __name__ == '__main__':
    x = torch.randn(32, 577, 768)
    attention = AttentionModel(768, 12, True)
    x = attention(x)
    print(x)
    # (32,577,768)
