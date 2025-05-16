import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(
        self, num_classes: int, d_model: int = 512, hidden_dim: int = 16, dropout: float=0.1
    ) -> None:
        """Initializes a multi-layer perceptron classifier with a single hidden layer with GELU activations
        Args:
            num_classes (int): number of output logits
            d_model (int, optional): number of input features. Defaults to 512.
            hidden_dim (int, optional): number of hidden layer units. Defaults to 16.
            dropout (float, optional): dropout probability for hidden layer. Defaults to 0.1.
        """

        super(MLP, self).__init__()
        self.feed_fwd = nn.Linear(in_features=d_model, out_features=hidden_dim)
        self.sigmoid = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.feed_fwd_2 = nn.Linear(in_features=hidden_dim, out_features=num_classes)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP

        Args:
            input (torch.Tensor): input features

        Returns:
            torch.Tensor: output logits of the MLP
        """
        return self.feed_fwd_2(self.dropout(self.sigmoid(self.feed_fwd(input))))
