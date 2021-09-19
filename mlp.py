import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout_p=0.5):
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


if __name__ == '__main__':
    x = torch.randn(32, 577, 768)
    mlp = MLP(768, 768 * 4, 768, 0.5)
    x = mlp(x)
    print(x.shape)
