import torch.nn as nn
import torch

class SelfAttention(nn.Module):
    def __init__(self, num_tokens = 2048, d_model=512, d_q_k_v=128, num_heads=6):
        super(SelfAttention, self).__init__()
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.d_q_k_v = d_q_k_v
        self.num_heads = num_heads

        self.query = nn.Linear(in_features=d_model,out_features=d_q_k_v*num_heads)
        self.key = nn.Linear(in_features=d_model, out_features=d_q_k_v*num_heads)    
        self.value = nn.Linear(in_features=d_model, out_features=d_q_k_v*num_heads)
        self.softmax = nn.Softmax(dim=2)
        self.output = nn.Linear(in_features=num_heads*d_q_k_v, out_features=d_model)


    def forward(self, embedding):
        batch_size = embedding.shape[0]
        queries = torch.reshape(self.query(embedding),(batch_size,self.num_tokens,self.num_heads,self.d_q_k_v))
        keys = torch.reshape(self.key(embedding),(batch_size,self.num_tokens,self.num_heads,self.d_q_k_v))
        values = self.value(embedding)
        values = torch.reshape(values,(batch_size,self.num_tokens,self.num_heads,self.d_q_k_v))

        # print(f"Queries {queries.shape}")
        # print(f"Keys {keys.shape}")
        # print(f"Keys transposed: {torch.transpose(keys,2,3).shape}")
        # q_k_multiplied = torch.matmul(queries, torch.transpose(keys,2,3))
        # print(f"QK multipled shape {q_k_multiplied.shape}")

        token_weights_pre_softmax = torch.einsum('bthd, buhd->btuh', queries, keys)
        token_weights_pre_softmax_scaled = torch.div(token_weights_pre_softmax, torch.sqrt(torch.tensor(self.d_model)))
        softmaxed_weights = self.softmax(token_weights_pre_softmax_scaled)
        value_weights = torch.einsum('btth,bthd->bthd', softmaxed_weights, values)
        value_weights_concat = torch.reshape(value_weights, (batch_size, self.num_tokens, self.num_heads*self.d_q_k_v))
        attention_output = self.output(value_weights_concat)
        return attention_output