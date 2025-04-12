from pathlib import Path

def getConfig():

    return{

        'batch_size': 8,
        'num_epochs': 20,
        'learning_rate': 10**-4,
        'sequence_length': 400,
        'model_dimension': 512,
        'language_src': "en",
        'language_target': "fr",
        'model_folder': "weights",
        'model_filename': "tmodel_",
        'preload': None,
        'tokenizer_file': "tokenizer_{0}.json",
        'experiment_name': "runs/tmodel",
    }

def getWeightFilePath(config, epoch: str):

    model_folder = config['model_folder']
    model_basename = config['model_name']
    model_filename = f"{model_basename}{epoch}.pt"

    return str(Path('.')/model_folder/model_filename)