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
    
# class that sends the tensor in to a FFN

class FeedForwardBlock(nn.Module):

    def __init__(self, model_dimension: int, d_ff: int, dropout: float):
        super().__init__()

        self.layer1 = nn.Linear(model_dimension, d_ff)
        self.layer2 = nn.Linear(d_ff, model_dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        return self.layer2(self.dropout(torch.relu(self.layer1(x))))
    
# class for the Multi-Head Attention Block which will be used for both self Multi-Head Attention of encoder and cross Multi-Head Attention of decoder

class MultiHeadAttention(nn.Module):

    def __init__(self, model_dimension: int, heads: int, dropout: float):
        super().__init__()

        self.model_dimension = model_dimension
        self.heads = heads
        self.dropout = nn.Dropout(dropout)

        self.d_k = model_dimension//heads

        self.Wq = nn.Linear(model_dimension, model_dimension)
        self.Wk = nn.Linear(model_dimension, model_dimension)
        self.Wv = nn.Linear(model_dimension, model_dimension)

        self.Wo = nn.Linear(model_dimension, model_dimension)

    @staticmethod
    def attention(query, key, value, mask, dropout):

        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1))/math.sqrt(d_k)

        if mask is not None:
            
            attention_scores.masked_fill_(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim = -1)

        if dropout is not None:

            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, query, key, value, mask):

        query = self.Wq(query)
        key = self.Wk(key)
        value = self.Wv(value)

        query = query.view(query.shape[0], query.shape[1], self.heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.heads, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contigous().view(x.shape[0], -1, self.heads* self.d_k)

        return self.Wo(x)
    
# class for the residual/direct connections between blocks

class ResiudalConnection(nn.Module):

    def __init__(self, dropout: float):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):

        return x + self.dropout(sublayer(self.norm(x)))
    
    


    