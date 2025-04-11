import torch
import torch.nn as nn
import math

# class to create and pass the input embeddings of each word in the sequence creating the tensor 
class InputEmbedding(nn.Module):

    def __init__(self, model_dimension: int, vocab_size: int):

        super().__init__()
        self.model_dimension = model_dimension
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, model_dimension)

    def forward(self, x):

        return self.embedding(x)*math.sqrt(self.model_dimension)

# class to calculate and add the positional encodings to the sequence 
class PositionalEncoding(nn.Module):

    def __init__(self, model_dimension: int, sequence_length: int, dropout: float):
        
        super().__init__()
        self.model_dimension = model_dimension
        self.sequence_length = sequence_length
        self.dropout = nn.Dropout(dropout)

        positional_encoding = torch.zeros(sequence_length,model_dimension)

        positions = torch.arange(0, sequence_length, dtype= torch.float).unsqueeze(1)

        divisor = torch.exp(torch.arange(0, model_dimension, 2).float() / model_dimension * -math.log(10000))

        positional_encoding[: , 0::2] = torch.sin(positions * divisor)

        positional_encoding[: , 1::2] = torch.cos(positions*divisor)

        positional_encoding.unsqueeze(0)

        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x):
        
        x = x + self.positional_encoding[:, :x.shape[1], :].requires_grad_(False)

        return self.dropout(x)
    
#class to return the normalized tensor as per x at j (Xj) = ((Xj-mean)/ root(variance +eps))*alpha + beta 

class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6):
        super().__init__()

        self.alpha = nn.Parameter(torch.ones(1))    
        self.beta = nn.Parameter(torch.zeros(1))

        self.eps = eps

    def forward(self, x):

        mean = x.mean(dim =-1, keepdim = True)
        std = x.std(dim =-1, keepdim = True)

        return self.alpha * (x - mean)/(std + self.eps) + self.beta
    


