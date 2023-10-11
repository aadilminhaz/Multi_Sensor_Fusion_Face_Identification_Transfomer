import torch.nn as nn

#import MultiheadAttention 
#import MLP
from MultiheadAttention import MultiheadAttention
from MLP import MLP


class Encoder(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio = 4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=12-6)
        self.attention = MultiheadAttention(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(in_features= dim, hidden_features= hidden_features, out_features=dim)   #Final Layer of Encoder

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x