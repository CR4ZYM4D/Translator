import torch
import torch.nn as nn
import ModelComponents as mc

# class for a single Decoder Block

class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: mc.MultiHeadAttention, cross_attention_block: mc.MultiHeadAttention, feed_forward_block: mc.FeedForwardBlock, dropout: float):
        super().__init__()

        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block

        self.residual_connections = nn.ModuleList([mc.ResiudalConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, target_mask):

        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, target_mask))

        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))

        x= self.residual_connections[2](x, self.feed_forward_block)

        return x

# class for the Decoder  

class Decoder(nn.Module):

    def __init__(self, layers:nn.ModuleList):
        super().__init__()

        self.layers = layers
        self.norm = mc.LayerNormalization()

    def forward(self, x, encoder_output, src_mask, target_mask):

        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)
        return self.norm(x)

# class for the Linear Layer after the decoder    

class ProjectionLayer(nn.Module):

    def __init__(self, model_dimension: int, vocab_size: int):
        super().__init__()

        self.proj = nn.Linear(model_dimension, vocab_size)

    def forward(self, x):

        return torch.log_softmax(self.proj(x), dim =-1)  