import torch.nn as nn
import torch


class SelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        d_q_k_v: int = 128,
        num_heads: int = 6,
    ):
        """Initializes a dot product self attention block

        Args:
            d_model (int, optional): Dimension of the token embeddings. Defaults to 512.
            d_q_k_v (int, optional): Dimension of the querie, key, and value vectors. Defaults to 128.
            num_heads (int, optional): Number of heads in multi-head attention. Defaults to 6.
        """
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.d_q_k_v = d_q_k_v
        self.num_heads = num_heads

        self.query = nn.Linear(in_features=d_model, out_features=d_q_k_v * num_heads)
        self.key = nn.Linear(in_features=d_model, out_features=d_q_k_v * num_heads)
        self.value = nn.Linear(in_features=d_model, out_features=d_q_k_v * num_heads)
        self.softmax = nn.Softmax(dim=-1)
        self.output = nn.Linear(in_features=num_heads * d_q_k_v, out_features=d_model)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Forward pass through the self attention module

        Args:
            embedding (torch.Tensor): Input embeddings

        Returns:
            torch.Tensor: Output of self attention
        """
        batch_size = embedding.shape[0]
        num_tokens = embedding.shape[1]

        queries = torch.reshape(
            self.query(embedding),
            (batch_size, self.num_heads, num_tokens, self.d_q_k_v),
        )
        keys = torch.reshape(
            self.key(embedding),
            (batch_size, self.num_heads, num_tokens, self.d_q_k_v),
        )
        values = torch.reshape(
            self.value(embedding),
            (batch_size, self.num_heads, num_tokens, self.d_q_k_v),
        )

        # print(queries.shape, keys.shape)
        token_weights_pre_softmax = torch.einsum("bhqd, bhkd->bhqk", queries, keys)
        token_weights_pre_softmax_scaled = torch.div(
            token_weights_pre_softmax, torch.sqrt(torch.tensor(self.d_q_k_v))
        )
        softmaxed_weights = self.softmax(token_weights_pre_softmax_scaled)

        # print(softmaxed_weights.shape, values.shape)
        value_weights = torch.einsum("bhtv,bhvd->bhtd", softmaxed_weights, values)

        value_weights_concat = value_weights.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.num_heads * self.d_q_k_v)
        attention_output = self.output(value_weights_concat)

        return attention_output
