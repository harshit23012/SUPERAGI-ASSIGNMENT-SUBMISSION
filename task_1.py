import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size not divisible by number of heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, queries, mask):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Attention mechanism
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden_size):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden_size)
        self.fc2 = nn.Linear(ff_hidden_size, embed_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, ff_hidden_size):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, ff_hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        # Apply layer normalization before the self-attention mechanism
        query_norm = self.norm1(query)
        key_norm = self.norm1(key)
        value_norm = self.norm1(value)

        attention = self.attention(value_norm, key_norm, query_norm, mask)

        # Add skip connection, followed by layer norm
        x = attention + query
        forward = self.feed_forward(self.norm2(x))

        # Add skip connection, followed by layer norm
        out = forward + x
        return out

class GPT2(nn.Module):
    def __init__(self, embed_size, num_layers, heads, vocab_size, max_length, ff_hidden_size, dropout):
        super(GPT2, self).__init__()
        self.embed_size = embed_size
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_size, heads, dropout, ff_hidden_size) for _ in range(num_layers)
        ])
        self.pos_encoding = PositionalEncoding(embed_size, max_length)
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_size, vocab_size)  # Output layer

    def forward(self, x, mask):
        x = self.token_embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for block in self.transformer_blocks:
            x = block(x, x, x, mask)

        logits = self.fc_out(x)
        return logits



# Custom weight initialization function
def _init_weights(module, n_layers):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # Apply the modified initialization for linear and embedding layers
        # involved in the residual connections
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

# GPT-2 125M model parameters
embed_size = 768  # Embedding size
num_layers = 12  # Number of transformer blocks
heads = 12  # Number of heads in multi-head attention
vocab_size = 50257  # Vocabulary size
max_length = 1024  # Maximum sequence length
ff_hidden_size = 3072  # Feedforward hidden layer size
dropout = 0.1  # Dropout rate

# Create an instance of the GPT-2 model
gpt2_125m_model = GPT2(embed_size, num_layers, heads, vocab_size, max_length, ff_hidden_size, dropout)
# Apply the custom weight initialization
gpt2_125m_model.apply(lambda module: _init_weights(module, num_layers))

# Sample input for testing
# Note: In a real scenario, you would process your input text to be compatible with the model's input requirements.
sample_input = torch.randint(0, vocab_size, (1, max_length))  # Randomly generated sample input
sample_mask = None  # No mask is applied in this demonstration

# Run a sample prediction
sample_output = gpt2_125m_model(sample_input, sample_mask)

# Displaying the shape of the output to demonstrate the model's functionality
sample_output.shape



