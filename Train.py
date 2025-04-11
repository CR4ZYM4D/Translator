import torch.nn as nn
from torch.utils.data import random_split, Dataset, DataLoader

from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

# method to get all the sentences for the language for which tokenizer has to be made

def getAllSentences(dataset, language):

    for item in dataset:
        yield item['translation'][language]

# method to fetch the tokenizer or build a new one if not there
 
def getOrBuildTokenizer(config, dataset, language):

    tokenizer_path = Path(config['tokenizer_file'].format(language))

    if not Path.exists(tokenizer_path):

        tokenizer = Tokenizer(WordLevel(unk_token = '[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency = 2)

        tokenizer.train_from_iterator(getAllSentences(dataset, language), trainer)

        tokenizer.save(str(tokenizer_path))

    else:

        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

# method to get the dataset

def getDataset(config):

    raw_dataset = load_dataset("opus_books", f'{config['language_src']}-{config['language_target']}', split = 'train')

    tokenizer_src = getOrBuildTokenizer(config, raw_dataset, config['language_src'])
    tokenizer_target = getOrBuildTokenizer(config, raw_dataset, config['language_target'])

    training_set_size = (int)(len(raw_dataset) * 0.9)
    testing_set_size = len(raw_dataset) - training_set_size

    raw_training_set, raw_testing_set = random_split(raw_dataset,[training_set_size, testing_set_size])