
import torch.nn as nn
import ModelComponents as mc

# class for a single Encoder Block
class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: mc.MultiHeadAttention, feed_forward_block: mc.FeedForwardBlock, dropout: float):
        super().__init__()

        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([mc.ResiudalConnection(dropout) for _ in range (2)])
        
    def forward(self, x, src_mask):

        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))

        x = self.residual_connections[1](x, self.feed_forward_block)

        return x
    
class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()

        self.layers = layers
        self.norm = mc.LayerNormalization()

    def forward(self, x, mask):

        for layer in self.layers:

            x = layer(x, mask)
        return x
