
import torch.nn as nn
import ModelComponents as mc 
import Encoder as e
import Decoder as d

class Transformer(nn.Module):

    def __init__(self, encoder: e.Encoder, decoder: d.Decoder, src_embedding: mc.InputEmbedding, target_embedding: mc.InputEmbedding, src_position: mc.PositionalEncoding, target_position: mc.PositionalEncoding, projection_layer: d.ProjectionLayer):

        super().__init__()

        self.encoder = encoder
        self.decoder= decoder
        self.src_embedding = src_embedding
        self.target_embedding = target_embedding
        self.src_position = src_position
        self.target_position = target_position
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):

        src = self.src_embedding(src)
        src = self.src_position(src)

        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, target, src_mask, target_mask):

        target = self.target_embedding(target)
        target = self.target_position(target)

        return self.decoder(target, encoder_output, src_mask, target_mask)
    
    def project(self, x):

        return self.projection_layer(x)
    
# function to build the transformer and initialize all the encoder decoder etc.

def buildTransformer(src_vocab_size: int, target_vocab_size: int, src_sequence_length: int, target_sequence_length: int, model_dimension: int = 512, heads: int = 8, blocks: int = 6, dropout: float = 0.1, d_ff: int = 2048)->Transformer: 

    src_embedding = mc.InputEmbedding(model_dimension, src_vocab_size)

    target_embedding = mc.InputEmbedding(model_dimension, target_vocab_size)

    src_position = mc.PositionalEncoding(model_dimension, src_sequence_length, dropout)

    target_position = mc.PositionalEncoding(model_dimension, target_sequence_length, dropout)

    encoder_blocks = []

    for _ in range(blocks):

        encoder_self_attention_block = mc.MultiHeadAttention(model_dimension, heads, dropout)

        encoder_feed_forward_block = mc.FeedForwardBlock(model_dimension, d_ff, dropout)

        encoder_blocks.append(e.EncoderBlock(encoder_self_attention_block, encoder_feed_forward_block, dropout))
        
    decoder_blocks = []

    for _ in range(blocks):

        decoder_self_attention_block = mc.MultiHeadAttention(model_dimension, heads, dropout)

        decoder_cross_attention_block = mc.MultiHeadAttention(model_dimension, heads, dropout)

        decoder_feed_forward_block = mc.FeedForwardBlock(model_dimension, d_ff, dropout)

        decoder_blocks.append(d.DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, decoder_feed_forward_block, dropout))

    encoder = e.Encoder(nn.ModuleList(encoder_blocks))
    decoder = d.Decoder(nn.ModuleList(decoder_blocks))
    projection_layer = d.ProjectionLayer(model_dimension, target_vocab_size)

    transformer = Transformer(encoder, decoder, src_embedding, target_embedding, src_position, target_position, projection_layer)

    for p in transformer.parameters():

        if p.dim()>1:

            nn.init.xavier_uniform_(p)

    return transformer

