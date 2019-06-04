import numpy as np
import datetime
import json
import argparse
import os
import multiprocessing
from time import time
from shutil import copy2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

from utils.vocab import create_vocab
from utils.eval import calculate_accuracy
from utils.wrap_var import to_var
from train.CL.parser import preprocess_config
from utils.gameplayutils import *
from utils.model_loading import load_model

from utils.datasets.CL.RndObjSampDataset import RndObjSampDataset # For Guesser Accuracy training
from utils.datasets.CL.QGenDataset import QGenDataset #For SL QGen training
from utils.datasets.GamePlay.GamePlayDataset import GamePlayDataset # For validation

from models.Oracle import Oracle
from models.Ensemble import Ensemble
# from models.N2N.CNN import ResNet

from train.CL.gameplay import gameplay_fwpass
from train.CL.qgen import qgen_fwpass
# TODO Make this capitalised everywhere to inform it is a global variable
use_cuda = torch.cuda.is_available()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", type=str, default="data", help='Data Directory')
    parser.add_argument("-config", type=str, default="config/CL/config.json", help=' General config file')
    parser.add_argument("-ens_config", type=str, default="config/CL/ensemble.json", help=' Ensemble config file')
    parser.add_argument("-or_config", type=str, default="config/CL/oracle.json", help=' Oracle config file')
    parser.add_argument("-modulo", type=int, default=1, help='This flag will cause the guesser to be updated every modulo number of epochs. If this flag is on then automatically epoch flag will overridden to 0') # TODO update help
    parser.add_argument("-exp_name", type=str, help='Experiment Name')
    parser.add_argument("-bin_name", type=str, default='', help='Name of the trained model file')
    parser.add_argument("-eval_newobj", action='store_true', help='To evaluate new object score for the model')
    parser.add_argument("-my_cpu", action='store_true', help='To select number of workers for dataloader. CAUTION: If using your own system then make this True')
    parser.add_argument("-breaking", action='store_true',
                        help='To Break training after 5 batch, for code testing purpose')
    parser.add_argument("-dataparallel", action='store_true', help='This for model files which     were saved with Dataparallel')

    args = parser.parse_args()
    print(args.exp_name)

    use_dataparallel = args.dataparallel
    breaking = args.breaking

    ensemble_args, dataset_args, optimizer_args, exp_config, oracle_args, word2i = preprocess_config(args)

    pad_token= word2i['<padding>']

    torch.manual_seed(exp_config['seed'])
    if use_cuda:
        torch.cuda.manual_seed_all(exp_config['seed'])

    float_tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    if exp_config['logging']:
        log_dir = exp_config['logdir']+str(args.exp_name)+exp_config['ts']+'/'
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        copy2(args.config, log_dir)
        copy2(args.ens_config, log_dir)
        copy2(args.or_config, log_dir)
        with open(log_dir+'args.txt', 'w') as f:
            f.write(str(vars(args))) # converting args.namespace to dict

    if exp_config['save_models']:
        model_dir = exp_config['save_models_path'] + args.bin_name + exp_config['ts'] + '/'
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        # This is again duplicate just for bookkeeping multiple times
        copy2(args.config, model_dir)
        copy2(args.ens_config, model_dir)
        copy2(args.or_config, model_dir)
        with open(model_dir+'args.txt', 'w') as f:
            f.write(str(vars(args))) # converting args.namespace to dict

    model = Ensemble(**ensemble_args)
    model = load_model(model, ensemble_args['bin_file'], use_dataparallel=use_dataparallel)
    # model.eval()

    oracle = Oracle(
        no_words            = oracle_args['vocab_size'],
        no_words_feat       = oracle_args['embeddings']['no_words_feat'],
        no_categories       = oracle_args['embeddings']['no_categories'],
        no_category_feat    = oracle_args['embeddings']['no_category_feat'],
        no_hidden_encoder   = oracle_args['lstm']['no_hidden_encoder'],
        mlp_layer_sizes     = oracle_args['mlp']['layer_sizes'],
        no_visual_feat      = oracle_args['inputs']['no_visual_feat'],
        no_crop_feat        = oracle_args['inputs']['no_crop_feat'],
        dropout             = oracle_args['lstm']['dropout'],
        inputs_config       = oracle_args['inputs'],
        scale_visual_to     = oracle_args['inputs']['scale_visual_to']
        )

    oracle = load_model(oracle, oracle_args['bin_file'], use_dataparallel=use_dataparallel)
    oracle.eval()

    softmax = nn.Softmax(dim=-1)

    #For Guesser
    guesser_loss_function = nn.CrossEntropyLoss() #For Guesser
    # For QGen.
    _cross_entropy = nn.CrossEntropyLoss(ignore_index=0)

    #TODO: Decider

    if use_dataparallel:
        encoder_optim = optim.Adam(model.module.encoder.parameters(), optimizer_args['lr'])
        decider_optim = optim.Adam(model.module.decider.parameters(), optimizer_args['lr'])
        guesser_optim = optim.Adam(model.module.guesser.parameters(), optimizer_args['lr'])
        qgen_optim = optim.Adam(model.module.qgen.parameters(), optimizer_args['lr'])
    else:
        encoder_optim = optim.Adam(model.encoder.parameters(), optimizer_args['lr'])
        decider_optim = optim.Adam(model.decider.parameters(), optimizer_args['lr'])
        guesser_optim = optim.Adam(model.guesser.parameters(), optimizer_args['lr'])
        qgen_optim = optim.Adam(model.qgen.parameters(), optimizer_args['lr'])

    #Guesser dataset based on the Random Object selection
    dataset_guesser = RndObjSampDataset(split='train', **dataset_args)
    #QGen dataset using the GT data
    dataset_qgen = QGenDataset(split='train', **dataset_args)

    #Validation data on the gameplay data
    dataset_val_gp = GamePlayDataset(split='val', **dataset_args)

    # TODO visualisation intit

    for epoch in range(optimizer_args['no_epochs']):

        start = time()
        print('epoch', epoch)
        # Condition for guesser and QGen

        gu_dataloader = DataLoader(
        dataset= dataset_guesser,
        batch_size=optimizer_args['batch_size'],
        shuffle=True,
        num_workers= 1 if optimizer_args['my_cpu'] else multiprocessing.cpu_count()//2,
        pin_memory= use_cuda,
        drop_last=False)

        qgen_dataloader = DataLoader(
        dataset= dataset_qgen,
        batch_size=optimizer_args['batch_size'],
        shuffle=True,
        num_workers= 1 if optimizer_args['my_cpu'] else multiprocessing.cpu_count()//2,
        pin_memory= use_cuda,
        drop_last=False)

        gp_dataloader = DataLoader(
        dataset=dataset_val_gp,
        batch_size=optimizer_args['batch_size'],
        shuffle=False, # If using this code for RL training make shuffle true
        num_workers= 1 if optimizer_args['my_cpu'] else multiprocessing.cpu_count()//2,
        pin_memory= use_cuda,
        drop_last=False)

        modulo_value = (epoch%args.modulo == 0)
        if modulo_value:
            train_dataloader = qgen_dataloader
        else:
            train_dataloader = gu_dataloader

        if args.eval_newobj:
            train_dataloader = gu_dataloader
            modulo_value = False
            if epoch > 4:
                break

        # cmd logging
        train_qgen_loss = float_tensor()
        train_guesser_loss = float_tensor()
        training_guesser_accuracy = list()
        val_gameplay_accuray = list()

        for split, dataloader in zip(['train', 'val'], [train_dataloader, gp_dataloader]):

            if split == 'train':
                model.train()
            else:
                model.eval()

            if args.eval_newobj and split == 'val':
                break

            for i_batch, sample in enumerate(dataloader):

                if i_batch > 5 and breaking:
                    print('Breaking after processing 4 batch')
                    break

                for k, v in sample.items():
                    if torch.is_tensor(v):
                        sample[k] = to_var(v)

                if modulo_value and split == 'train':

                    qgen_out = qgen_fwpass(q_model= model, inputs= sample, use_dataparallel= use_dataparallel)

                    word_logits_loss = _cross_entropy(qgen_out.view(-1, 4901), sample['target_q'].view(-1)) #TODO remove this hardcoded number

                    # Backprop
                    encoder_optim.zero_grad()
                    qgen_optim.zero_grad()
                    word_logits_loss.backward()
                    encoder_optim.step()
                    qgen_optim.step()

                    train_qgen_loss = torch.cat([train_qgen_loss, word_logits_loss.data])
                    # print(word_logits_loss)
                    # print(train_qgen_loss)
                else:
                    # Guesser GamePlay training
                    # TODO better this one
                    sample['pad_token'] = pad_token
                    exp_config['max_no_qs'] = dataset_args['max_no_qs']
                    exp_config['use_dataparallel'] = use_dataparallel
                    guesser_logits = gameplay_fwpass(q_model= model, o_model= oracle, inputs= sample, exp_config= exp_config, word2i= word2i)

                    guesser_loss = guesser_loss_function(guesser_logits*sample['objects_mask'].float(), sample['target_obj'])
                    guesser_accuracy = calculate_accuracy(softmax(guesser_logits)*sample['objects_mask'].float(), sample['target_obj'])

                    if split == 'train':
                        if not args.eval_newobj:
                            encoder_optim.zero_grad()
                            guesser_optim.zero_grad()
                            guesser_loss.backward()
                            encoder_optim.step()
                            guesser_optim.step()

                        train_guesser_loss = torch.cat([train_guesser_loss, guesser_loss.data])
                        training_guesser_accuracy.append(guesser_accuracy)
                    else:
                        val_gameplay_accuray.append(guesser_accuracy)

        if exp_config['save_models'] and not args.eval_newobj:
            model_file = os.path.join(model_dir, ''.join(['model_ensemble_addnTrain_', args.bin_name,'_E_', str(epoch)]))
            torch.save(model.state_dict(), model_file)

        print("Epoch %03d, Time taken %.3f"%(epoch, time()-start))
        if modulo_value:
            print("Training Loss:: QGen %.3f"%(torch.mean(train_qgen_loss)))
        else:
            print("Training Guesser:: Loss %.3f, Accuracy %.5f"%(torch.mean(train_guesser_loss), np.mean(training_guesser_accuracy)))
        print("Validation GP Accuracy:: %.5f"%(np.mean(val_gameplay_accuray)))
        if exp_config['save_models'] and not args.eval_newobj:
            print("Saved model to %s" % (model_file))
        print('-----------------------------------------------------------------')
        # GamePlay validation score
