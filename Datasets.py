import torch 
import torch.nn as nn
from torch.utils.data import Dataset

class LanguageDataset(Dataset):

    def __init__(self, dataset, tokenizer_src, tokenizer_target, language_src, language_target, sequence_length):
        super().__init__()

        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_target = tokenizer_target
        self.language_src = language_src
        self.language_target = language_target
        self.sequence_length = sequence_length

        self.sos_token = torch.Tensor([tokenizer_src.token_to_id(['[SOS]'])], dtype = torch.int64)
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id(['[EOS]'])], dtype = torch.int64)
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id(['[PAD]'])], dtype = torch.int64)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        
        src_target_pair = self.dataset[index]
        src_text = src_target_pair['translation'][self.language_src]
        target_text = src_target_pair['translation'][self.language_target]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_target.encode(target_text).ids

        num_enc_padding_tokens = self.sequence_length - len(src_text) - 2
        num_dec_padding_tokens = self.sequence_length - len(target_text) - 1

        if num_dec_padding_tokens<0 or num_enc_padding_tokens < 0:

            raise ValueError ("Sentence too long")
        
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * num_enc_padding_tokens, dtype = torch.int64)
            ]
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                torch.tensor([self.pad_token] * num_dec_padding_tokens, dtype = torch.int64)
            ]
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype= torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * num_dec_padding_tokens, dtype = torch.int64)
            ]
        )

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causalMask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "target_text": target_text
        }
    
def causalMask(size):

    mask = torch.triu(torch.ones(1,size,size), diagonal =1).type(torch.int)

    return mask == 0

