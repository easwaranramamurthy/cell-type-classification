from self_attention import SelfAttention
from mlp_classifier import MLP
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self,
                 num_layers=6,
                 num_tokens = 2048,
                 d_model=512,
                 d_q_k_v=128,
                 num_heads=6):
        super(SelfAttention, self).__init__()
        self.embed = nn.Embedding(embedding_dim=d_model)
        self.attn_layers = [SelfAttention(num_tokens, d_model, d_q_k_v, num_heads) for _ in range(num_layers)]
        self.attn_layer_norms = [nn.LayerNorm(d_model) for _ in range(num_layers)]
        self.feed_fwd_layers = [nn.Linear(in_features=d_model,out_features=d_model) for _ in range(num_layers)]
        self.feed_fwd_activations = [nn.Sigmoid() for _ in range(num_layers)]
        self.feed_fwd_layer_norms = [nn.LayerNorm(d_model) for _ in range(num_layers)]
        self.final_mlp = MLP()



    def forward(self, input):
        embedding = self.embed(input)
        for attn_layer, attn_layer_norm, feed_fwd_layer, feed_fwd_activation, feed_fwd_layer_norm in zip(self.attn_layers, self.attn_layer_norms, self.feed_fwd_layers, self.feed_fwd_activations, self.feed_fwd_layer_norms):
            attn_output = attn_layer(embedding)
            residual_added = embedding+attn_output
            normed_embedding = attn_layer_norm(residual_added)
            feed_fwd_output = feed_fwd_activation(feed_fwd_layer(normed_embedding))
            residual_added_feed_fwd_output = normed_embedding+feed_fwd_output
            embedding = feed_fwd_layer_norm(residual_added_feed_fwd_output)
        logits = self.final_mlp(embedding[:,0,:,:])
        return logits