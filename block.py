import torch.nn as nn
from attention import AttentionModel
from mlp import MLP


class Block(nn.Module):
    def __init__(self, dim, num_heads, include_bias=True, projection_dropout=0.5, attention_dropout=0.5):
        super(Block, self).__init__()
        self.attention = AttentionModel(dim, num_heads, include_bias, attention_dropout, projection_dropout)
        self.mlp = MLP(dim, dim * 4, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        attention_output = self.attention(x, x, x)
        sum_attention = self.norm1(x + attention_output)
        output = self.norm2(sum_attention + self.mlp(sum_attention))
        return output
