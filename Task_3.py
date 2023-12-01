import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super(RotaryPositionalEmbedding, self).__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0.0, dim, 2.0) / dim))
        t = torch.arange(max_len).type_as(inv_freq)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        self.register_buffer('sin', freqs.sin())
        self.register_buffer('cos', freqs.cos())

    def forward(self, x):
        n, _, device = x.shape[1], x.shape[2], x.device
        sin, cos = self.sin[:n, :].to(device), self.cos[:n, :].to(device)
        return torch.einsum('bnd,nd->bnd', x, cos) + torch.einsum('bnd,nd->bnd', x.flip(-1), sin)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        # Initialize RotaryPositionalEmbedding here
        self.rotary_pos_emb = RotaryPositionalEmbedding(self.head_dim)

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

        # Apply Rotary Positional Embedding
        queries = self.rotary_pos_emb(queries)
        keys = self.rotary_pos_emb(keys)

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



class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_size, heads, groups):
        super(GroupedQueryAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.groups = groups
        self.head_dim = embed_size // heads
        assert self.head_dim * heads == embed_size, "Embedding size not divisible by number of heads"
        assert heads % groups == 0, "Number of heads must be divisible by number of groups"

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

        # Group the heads
        values, keys = self._group_heads(values), self._group_heads(keys)
        queries = self._group_heads(queries)

        # Calculate attention
        energy = torch.einsum("nqghd,nkghd->nghqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.groups * self.head_dim
        )

        # Concatenate and pass through the final linear layer
        out = self.fc_out(out)
        return out

    def _group_heads(self, x):
        # Reshape the input to group heads
        new_head_dim = self.head_dim * (self.heads // self.groups)
        return x.reshape(*x.shape[:-2], self.groups, new_head_dim)


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

class SlidingWindowAttention(nn.Module):
    def __init__(self, embed_size, heads, window_size, dilate=False):
        super(SlidingWindowAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.window_size = window_size
        self.dilate = dilate
        self.head_dim = embed_size // heads

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, queries, mask):
        N, value_len, key_len, query_len = values.size(0), values.size(1), keys.size(1), queries.size(1)

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # Apply sliding window and optional dilation
        if self.dilate:
            step = self.window_size // 2
            values, keys = values[:, ::step, :, :], keys[:, ::step, :, :]

        # Compute attention scores
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / math.sqrt(self.embed_size), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out


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
    def __init__(self, embed_size, heads, dropout, ff_hidden_size, window_size, use_grouped_attention=False):
        super(TransformerBlock, self).__init__()

        self.use_grouped_attention = use_grouped_attention
        if use_grouped_attention:
            self.attention = GroupedQueryAttention(embed_size, heads, groups=4)  # You can adjust the number of groups
        else:
            self.attention = SlidingWindowAttention(embed_size, heads, window_size, dilate=False)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, ff_hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        # Apply layer normalization before the self-attention mechanism
        query_norm = self.norm1(query)
        key_norm = self.norm1(key)
        value_norm = self.norm1(value)

        # Add skip connection, followed by layer norm
        attention_out = self.attention(value_norm, key_norm, query_norm, mask)
        x = attention_out + query
        x = self.norm2(x)

        forward = self.feed_forward(x)

        # Add skip connection, followed by layer norm
        out = forward + x
        return out


class GPT2(nn.Module):
    def __init__(self, embed_size, num_layers, heads, vocab_size, max_length, ff_hidden_size, dropout, use_grouped_attention=False):
        super(GPT2, self).__init__()
        self.embed_size = embed_size
        self.transformer_blocks = nn.ModuleList([
    TransformerBlock(embed_size, heads, dropout, ff_hidden_size, window_size=5, use_grouped_attention=use_grouped_attention) for _ in range(num_layers)
])
        self.pos_encoding = PositionalEncoding(embed_size, max_length)
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.rotary_pos_emb = RotaryPositionalEmbedding(embed_size // heads)

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
import os
import torch.distributed as dist

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'  # Can be the IP address of the master node
    os.environ['MASTER_PORT'] = '12355'  # Any free port can be used
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


import os
import torch
import torch.distributed as dist

# Set the MASTER_ADDR and MASTER_PORT environment variables at the start of the script
os.environ.setdefault('MASTER_ADDR', 'localhost')  # Default to 'localhost' if not set
os.environ.setdefault('MASTER_PORT', '12355')      # Default to '12355' if not set

def setup(rank, world_size):
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """ Cleanup distributed training setup """
    dist.destroy_process_group()

def train(rank, world_size, fsdp=False):
    setup(rank, world_size)

    # Define dataset and dataloader
    dataset = MyDataset()
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Initialize model
    model = GPT2(embed_size, num_layers, heads, vocab_size, max_length, ff_hidden_size, dropout, use_grouped_attention=False)

    # Move model to GPU
    device = torch.device("cuda:{}".format(rank))
    model.to(device)

    if fsdp:
        # Wrap model with Fully Sharded Data Parallel
        model = FSDP(model)
    else:
        # Wrap model with Distributed Data Parallel
        model = DDP(model, device_ids=[rank])

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if rank == 0:
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}")

    cleanup()

def main():
    # Configuration for single GPU or distributed setup
    world_size = torch.cuda.device_count()
    fsdp = True  # Set to True to use Fully Sharded Data Parallel

    if world_size > 1:
        # Multi-GPU (Distributed Training)
        torch.multiprocessing.spawn(train, args=(world_size, fsdp), nprocs=world_size, join=True)
    else:
        # Single GPU
        train(0, 1, fsdp)

if __name__ == "__main__":
    main()


