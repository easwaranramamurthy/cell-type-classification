from models.self_attention import SelfAttention
from models.mlp_classifier import MLP
import torch.nn as nn
import torch


class Transformer(nn.Module):
    def __init__(
        self,
        num_layers: int = 4,
        vocab_size: int = 20000,
        num_tokens: int = 2048,
        d_model: int = 512,
        d_q_k_v: int = 128,
        num_heads: int = 6,
        num_classes: int = 4,
        hidden_dim: int = 16,
    ) -> None:
        """Implements a full bidirectional transformer model which uses a CLS token to perform classification

        Args:
            num_layers (int, optional): Number of self attention layers in network. Defaults to 6.
            vocab_size (int, optional): Vocabulary size. Defaults to 20000.
            num_tokens (int, optional): Input context length. Defaults to 2048.
            d_model (int, optional): Dimension of the token embeddings. Defaults to 512.
            d_q_k_v (int, optional): Dimension of the query, key, and value vectors. Defaults to 128.
            num_heads (int, optional): Number of heads in each self attention layer. Defaults to 6.
            num_classes (int, optional): Number of output classes. Defaults to 4.
            hidden_dim (int, optional): Dimension of the hidden layer in the <CLS> MLP classifier. Defaults to 16

        """
        super(Transformer, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

        self.attn_ops = nn.ModuleList()
        self.first_layernorms = nn.ModuleList()
        self.feed_fwd_layers = nn.ModuleList()
        self.feed_fwd_activations = nn.ModuleList()
        self.second_layernorms = nn.ModuleList()

        for i in range(num_layers):
            self.attn_ops.add_module(f'attn_{i}', SelfAttention(num_tokens, d_model, d_q_k_v, num_heads).to(device='mps'))
            self.first_layernorms.add_module(f'ln1_{i}', nn.LayerNorm(d_model).to(device='mps'))
            self.feed_fwd_layers.add_module(f'fwd_{i}', nn.Linear(in_features=d_model, out_features=d_model).to(device='mps'))
            self.feed_fwd_activations.add_module(f'fwd_activation_{i}', nn.Sigmoid().to(device='mps'))
            self.second_layernorms.add_module(f'ln2_{i}', nn.LayerNorm(d_model).to(device='mps'))
       
        self.final_mlp = MLP(num_classes=num_classes, d_model=d_model, hidden_dim=hidden_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer model

        Args:
            input (torch.Tensor): input token indexes into vocab

        Returns:
            torch.Tensor: output logits from the final MLP classification
        """

        embedding = self.embed(input)
        # # print(f"Embedding {embedding.shape}")

        for (attn_op,
             first_layernorm,
             feed_fwd_layer,
             feed_fwd_activation,
             second_layernorm) in zip(self.attn_ops,
                                      self.first_layernorms,
                                      self.feed_fwd_layers,
                                      self.feed_fwd_activations,
                                      self.second_layernorms):
            attn_output = attn_op(embedding)
            # print(f"Attention output {attn_output.shape}")
            residual_added = embedding + attn_output
            # print(f"Residual added to attention output: {residual_added.shape}")
            normed_embedding = first_layernorm(residual_added)
            # print(f"Layer normalized embedding {normed_embedding.shape}")
            feed_fwd_output = feed_fwd_activation(feed_fwd_layer(normed_embedding))
            # print(f"Feed forward output {feed_fwd_output.shape}")
            residual_added_feed_fwd_output = normed_embedding + feed_fwd_output
            # print(f"Residual added to feed forward {residual_added_feed_fwd_output.shape}")
            embedding = second_layernorm(residual_added_feed_fwd_output)
            # print(f"Final embedding {embedding.shape}")
        # print(embedding.shape)
        # computing logits on the first token - CLS
        logits = self.final_mlp(embedding[:, 0, :])
        # print(f"Logits {logits.shape}")
        return logits
