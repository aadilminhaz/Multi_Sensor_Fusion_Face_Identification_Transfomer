import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fcLayer1 = nn.Linear(in_features, hidden_features)
        self.activation = nn.GELU()   #need to research on that, and see if it can be improved
        self.fcLayer2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.fcLayer1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fcLayer2(x)
        x = self.drop(x)
        return x