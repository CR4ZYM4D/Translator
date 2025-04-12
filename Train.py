import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split, Dataset, DataLoader

from Transformer import buildTransformer
import Datasets
from config import getWeightFilePath, getConfig

from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from tqdm import tqdm
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

# method to get and split the dataset

def getDataset(config):

    raw_dataset = load_dataset("opus_books", f"{config['language_src']}-{config['language_target']}", split = 'train')

    tokenizer_src = getOrBuildTokenizer(config, raw_dataset, config['language_src'])
    tokenizer_target = getOrBuildTokenizer(config, raw_dataset, config['language_target'])

    training_set_size = (int)(len(raw_dataset) * 0.9)
    testing_set_size = len(raw_dataset) - training_set_size

    raw_training_set, raw_testing_set = random_split(raw_dataset,[training_set_size, testing_set_size])

    training_set = Datasets.LanguageDataset(raw_training_set, tokenizer_src, tokenizer_target, config['language_src'], config['language_target'], config['sequence_length'])

    testing_set = Datasets.LanguageDataset(raw_testing_set, tokenizer_src, tokenizer_target, config['language_src'], config['language_target'], config['sequence_length'])

    max_src_len = 0
    max_target_len = 0

    for item in raw_dataset:

        src_tokens = tokenizer_src.encode(item['translation'][config['language_src']]).ids
        target_tokens = tokenizer_target.encode(item['translation'][config['language_target']]).ids

        max_src_len = max(max_src_len, len(src_tokens)) 
        max_target_len = max(max_target_len, len(target_tokens))

    training_dataloader = DataLoader(training_set, batch_size = config['batch_size'], shuffle = True) 
    testing_dataloader = DataLoader(testing_set, batch_size = 1, shuffle = True)

    return training_dataloader, testing_dataloader, tokenizer_src, tokenizer_target 

def getModel(config, src_vocab_size, target_vocab_size):

    model = buildTransformer(src_vocab_size, target_vocab_size, config['sequence_length'], config['sequence_length'])

    return model

def trainModel(config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Path(config['model_folder']).mkdir( parents= True, exist_ok =True)

    training_dataloader, testing_dataloader, tokenizer_src, tokenizer_target = getDataset(config)

    model = getModel(config, tokenizer_src.get_vocab_size(), tokenizer_target.get_vocab_size()).to(device)

    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr = config['learning_rate'], eps = 1e-9)

    initial_epoch = 0
    global_step = 0

    if config['preload']:

        model_filename = getWeightFilePath(config, config['preload'])

        state = torch.load(model_filename)

        initial_epoch = state['epoch'] + 1

        optimizer.load_state_dict(state['optimizer_state_dict'])

        global_step = state['global_step']

    loss_function = nn.CrossEntropyLoss(ignore_index = tokenizer_src.token_to_id('[PAD]'), label_smoothing= 0.1).to(device)

    for epoch in range (initial_epoch, config['num_epochs']):

        model.train()

        batch_iterator = tqdm(training_dataloader, desc = f"processing epoch {epoch: 02d}")

        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)

            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)

            decoder_output = model.decode(encoder_output, decoder_input, encoder_mask, decoder_mask)

            projection_output = model.project(decoder_output)

            label = batch['label'].to(device)

            loss = loss_function(projection_output.view(-1, tokenizer_target.get_vocab_size()), label.view(-1))

            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            writer.add_scalar("loss", loss.item(), global_step)

            writer.flush()

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()

            global_step += 1
        
        model_filename = getWeightFilePath(config, f'{epoch:02d}')

        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename
        )

if __name__ == '__main__':

    config = getConfig()

    model = trainModel(config)
