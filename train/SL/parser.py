import os
import json
import datetime
from time import time

import torch
from torch.autograd import Variable
import torch.nn as nn
from utils.config import load_config
from utils.vocab import create_vocab

use_cuda = torch.cuda.is_available()

#TODO: It will be good to have similar preprocssing as for the CL.

def preprocess_config(args):
    """Function to process the arguments and returns the relevant arguments for training.

    Parameters
    ----------
    args : type
        check config.json for all the dictionary keys

    Returns
    -------
    ensemble_args : dict. Arguments for all the modules.
    dataset_args : dict. Arguments for Dataset
    optimizer_args : dict. Arghuments for the optimizer
    exp_config : dict. Arguments for saving models, logging etc...

    """
    # TODO Also load arguments for visualisation

    config = load_config(args.config)

    # Create vocab.json
    if config['dataset']['new_vocab'] or not os.path.isfile(os.path.join(args.data_dir, config['data_paths']['vocab_file'])):
        create_vocab(
            data_dir=args.data_dir,
            data_file=config['data_paths']['train'],
            min_occ=config['dataset']['min_occ'])

    with open(os.path.join(args.data_dir, config['data_paths']['vocab_file'])) as file:
        vocab = json.load(file)
    word2i = vocab['word2i']
    vocab_size = len(word2i)
    word_pad_token = word2i['<padding>']

    # Experiments_args
    exp_config = config['exp_config']
    exp_config['ts'] = str(datetime.datetime.fromtimestamp(time()).strftime('%Y_%m_%d_%H_%M'))

    ensemble_args = dict()

    # Encoder args
    ensemble_args['encoder'] = config['encoder']
    ensemble_args['encoder']['vocab_size'] = vocab_size
    ensemble_args['encoder']['word_embedding_dim'] = config['embeddings']['word_embedding_dim']
    ensemble_args['encoder']['word_pad_token'] = word_pad_token
    ensemble_args['encoder']['decider'] = config['exp_config']['decider']

    # Guesser args
    ensemble_args['guesser'] = config['guesser']
    ensemble_args['guesser']['no_categories'] = config['embeddings']['no_categories']
    ensemble_args['guesser']['obj_pad_token'] = config['embeddings']['obj_pad_token']
    ensemble_args['guesser']['obj_categories_embedding_dim'] = config['embeddings']['obj_categories_embedding_dim']
    # ensemble_args['guesser']['encoder_hidden_dim'] = config['encoder']['hidden_dim']


    # QGen args
    if exp_config['qgen'] == 'qgen_cap':
        ensemble_args['qgen'] = config['qgen_cap']
        ensemble_args['qgen']['qgen'] = 'qgen_cap'
        ensemble_args['qgen']['encoder_hidden_dim'] = config['encoder']['scale_to']
    else:
        ensemble_args['qgen'] = config['qgen']
        ensemble_args['qgen']['qgen'] = 'qgen'
    ensemble_args['qgen']['max_tgt_length'] = config['dataset']['max_q_length']
    ensemble_args['qgen']['vocab_size'] = vocab_size
    ensemble_args['qgen']['word_embedding_dim'] = config['embeddings']['word_embedding_dim']
    ensemble_args['qgen']['word_pad_token'] = word_pad_token
    ensemble_args['qgen']['visual_features_dim'] = config['encoder']['visual_features_dim']
    ensemble_args['qgen']['start_token'] = word2i['<start>']

    # Decider
    ensemble_args['decider'] = config['decider']
    ensemble_args['decider']['type'] = config['exp_config']['decider']

    # Dataset
    dataset_args = config['dataset']
    dataset_args['data_dir'] = args.data_dir
    dataset_args['data_paths'] = config['data_paths']
    dataset_args['my_cpu'] = args.my_cpu

    # Optimizer_args
    optimizer_args = config['optimizer']
    optimizer_args['my_cpu'] = args.my_cpu

    if exp_config['logging']:
        exp_config['exp_name'] = args.exp_name

    return ensemble_args, dataset_args, optimizer_args, exp_config
