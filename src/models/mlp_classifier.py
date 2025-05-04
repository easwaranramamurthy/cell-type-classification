import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, num_classes, d_model=512,hidden_dim=32):
        super(MLP, self).__init__()
        self.feed_fwd = nn.Linear(in_features=d_model, out_features=hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.feed_fwd_2 = nn.Linear(in_features=hidden_dim, out_features=num_classes)


    def forward(self, input):
        return self.feed_fwd_2(self.sigmoid(self.feed_fwd(input)))