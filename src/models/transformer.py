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
    ) -> None:
        """Implements a full bidirectional transformer model which uses a CLS token to perform classification

        Args:
            num_layers (int, optional): Number of self attention layers in network. Defaults to 6.
            vocab_size (int, optional): Vocabulary size. Defaults to 20000.
            num_tokens (int, optional): Input context length. Defaults to 2048.
            d_model (int, optional): Dimension of the token embeddings. Defaults to 512.
            d_q_k_v (int, optional): Dimension of the querie, key, and value vectors. Defaults to 128.
            num_heads (int, optional): Number of heads in each self attention layer. Defaults to 6.
            num_classes (int, optional): Number of output classes. Defaults to 4.
        """
        super(Transformer, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.attn_layers = [
            SelfAttention(num_tokens, d_model, d_q_k_v, num_heads)
            for _ in range(num_layers)
        ]
        # TODO: make sure this is applying layer norm over the correct dimension
        self.attn_layer_norms = [nn.LayerNorm(d_model) for _ in range(num_layers)]
        self.feed_fwd_layers = [
            nn.Linear(in_features=d_model, out_features=d_model)
            for _ in range(num_layers)
        ]
        self.feed_fwd_activations = [nn.Sigmoid() for _ in range(num_layers)]
        # TODO: make sure this is applying layer norm over the correct dimension
        self.feed_fwd_layer_norms = [nn.LayerNorm(d_model) for _ in range(num_layers)]
        self.final_mlp = MLP(num_classes=num_classes)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer model

        Args:
            input (torch.Tensor): input token indexes into vocab

        Returns:
            torch.Tensor: output logits from the final MLP classification
        """

        embedding = self.embed(input)
        # # print(f"Embedding {embedding.shape}")
        for (
            attn_layer,
            attn_layer_norm,
            feed_fwd_layer,
            feed_fwd_activation,
            feed_fwd_layer_norm,
        ) in zip(
            self.attn_layers,
            self.attn_layer_norms,
            self.feed_fwd_layers,
            self.feed_fwd_activations,
            self.feed_fwd_layer_norms,
        ):
            attn_output = attn_layer(embedding)
            # print(f"Attention output {attn_output.shape}")
            residual_added = embedding + attn_output
            # print(f"Residual added to attention output: {residual_added.shape}")
            normed_embedding = attn_layer_norm(residual_added)
            # print(f"Layer normalized embedding {normed_embedding.shape}")
            feed_fwd_output = feed_fwd_activation(feed_fwd_layer(normed_embedding))
            # print(f"Feed forward output {feed_fwd_output.shape}")
            residual_added_feed_fwd_output = normed_embedding + feed_fwd_output
            # print(f"Residual added to feed forward {residual_added_feed_fwd_output.shape}")
            embedding = feed_fwd_layer_norm(residual_added_feed_fwd_output)
            # print(f"Final embedding {embedding.shape}")
        print(embedding.shape)
        # computing logits on the first token - CLS
        logits = self.final_mlp(embedding[:, 0, :])
        # print(f"Logits {logits.shape}")
        return logits
