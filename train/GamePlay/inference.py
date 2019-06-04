import numpy as np
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

from utils.vocab import create_vocab
from utils.eval import calculate_accuracy
from utils.model_loading import load_model
from utils.gameplayutils import *
from train.GamePlay.parser import preprocess_config

from utils.datasets.GamePlay.GameplayN2NResNetDataset import GameplayN2NResNetDataset
from utils.datasets.GamePlay.GamePlayDataset import GamePlayDataset

from models.Oracle import Oracle
from models.Ensemble import Ensemble
from models.CNN import ResNet



# TODO Make this capitalised everywhere to inform it is a global variable
use_cuda = torch.cuda.is_available()

#TODO: Move this code from the train folder

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", type=str, default="data", help='Data Directory')
    parser.add_argument("-config", type=str, default="config/GamePlay/config.json", help=' General config file')
    parser.add_argument("-ens_config", type=str, default="config/GamePlay/ensemble.json", help=' Ensemble config file')
    parser.add_argument("-or_config", type=str, default="config/GamePlay/oracle.json", help=' Oracle config file')
    parser.add_argument("-exp_name", type=str, help='Experiment Name')
    parser.add_argument("-my_cpu", action='store_true', help='To select number of workers for dataloader. CAUTION: If using your own system then make this True')
    parser.add_argument("-breaking", action='store_true',
                        help='To Break training after 5 batch, for code testing purpose')
    parser.add_argument("-resnet", action='store_true', help='This flag will cause the program to use the image features from the ResNet forward pass instead of the precomputed ones.')
    parser.add_argument("-dataparallel", action='store_true', help='This for model files which were saved with Dataparallel')
    parser.add_argument("-log_enchidden", action='store_true', help='This flag saves the encoder hidden state. WARNING!!! This might cause the resulting json file to blow up!')

    # --------Arguments from config.json that can be overridden here. Similar changes have to be made in the util file and not here--------------------
    parser.add_argument("-batch_size", type=int, help='Batch size for the gameplay')
    parser.add_argument("-load_bin_path", type=str, help='Bin file path for the saved model. If this is not given then one provided in ensemble.json will be taken ')

    args = parser.parse_args()
    print(args.exp_name)
    use_dataparallel = args.dataparallel
    breaking = args.breaking

    # Load the Arguments and Hyperparamters
    ensemble_args, dataset_args, optimizer_args, exp_config, oracle_args, word2i, i2word, catid2str = preprocess_config(args)

    pad_token= word2i['<padding>']

    torch.manual_seed(exp_config['seed'])
    if use_cuda:
        torch.cuda.manual_seed_all(exp_config['seed'])

    if exp_config['logging']:
        log_dir = exp_config['logdir']+str(args.exp_name)+exp_config['ts']+'/'
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        copy2(args.config, log_dir)
        copy2(args.ens_config, log_dir)
        copy2(args.or_config, log_dir)
        with open(log_dir+'args.txt', 'w') as f:
            f.write(str(vars(args))) # converting args.namespace to dict

    model = Ensemble(**ensemble_args)
    model = load_model(model, ensemble_args['bin_file'], use_dataparallel=use_dataparallel)
    model.eval()

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

    print(model)
    print(oracle)

    if args.resnet:
        cnn = ResNet()

        if use_cuda:
            cnn.cuda()
            cnn = DataParallel(cnn)
        cnn.eval()

    softmax = nn.Softmax(dim=-1)

    if args.resnet:
        dataset_val = GameplayN2NResNetDataset(split='val', **dataset_args)
        dataset_test = GameplayN2NResNetDataset(split='test', **dataset_args)
    else:
        dataset_val = GamePlayDataset(split='val', **dataset_args)
        dataset_test = GamePlayDataset(split='test', **dataset_args)

    for split, dataset in zip(exp_config['splits'], [dataset_val, dataset_test]):
        print(split)
        eval_log = dict()

        dataloader = DataLoader(
        dataset=dataset,
        batch_size=optimizer_args['batch_size'],
        shuffle=False, # This is made False to check model performance at batch level.
        num_workers= 1 if optimizer_args['my_cpu'] else multiprocessing.cpu_count()//2,
        pin_memory= use_cuda,
        drop_last=False)

        total_no_batches = len(dataloader)
        accuracy = list()
        decider_perc = list()
        start = time()

        for i_batch, sample in enumerate(dataloader):

            # Breaking condition
            if breaking and i_batch>5:
                print('Breaking after 5')
                break

            # Get Batch
            for k, v in sample.items():
                if torch.is_tensor(v):
                    sample[k] = to_var(v, True)

            if args.resnet:
                img_features, avg_img_features = cnn(to_var(sample['image'].data, True))
            else:
                avg_img_features = sample['image']

            batch_size = avg_img_features.size(0)

            history = to_var(torch.LongTensor(batch_size, 200).fill_(pad_token))
            history[:,0] = sample['history']
            history_len = sample['history_len']

            decisions = to_var(torch.LongTensor(batch_size).fill_(0))
            mask_ind = torch.nonzero(1-decisions).squeeze()
            _enc_mask = mask_ind

            if exp_config['logging']:
                #Logging for 10 questions+ begining start token
                decision_probs = to_var(torch.zeros((batch_size, dataset_args['max_no_qs']+1, 1)))
                all_guesser_probs = to_var(torch.zeros((batch_size, dataset_args['max_no_qs']+1,20)))
                if args.log_enchidden:
                    #Logging for 10 questions + begining start token
                    enc_hidden_logging = to_var(torch.zeros((batch_size, dataset_args['max_no_qs']+1, ensemble_args['encoder']['scale_to'])))

            for q_idx in range(dataset_args['max_no_qs']):

                if use_dataparallel and use_cuda:
                    encoder_hidden = model.module.encoder(history=history[mask_ind], visual_features=avg_img_features[mask_ind], history_len=history_len[mask_ind])
                    decision = model.module.decider(encoder_hidden=encoder_hidden)
                else:
                    encoder_hidden = model.encoder(history=history[mask_ind], visual_features=avg_img_features[mask_ind], history_len=history_len[mask_ind])
                    decision = model.decider(encoder_hidden=encoder_hidden)

                ########## Logging Block ################
                if exp_config['logging'] and q_idx==0:
                    if use_dataparallel and use_cuda:
                        tmp_guesser_logits = model.module.guesser(encoder_hidden=encoder_hidden, spatials=sample['spatials'], objects=sample['objects'], regress= False)
                    else:
                        tmp_guesser_logits = model.guesser(encoder_hidden=encoder_hidden, spatials=sample['spatials'], objects=sample['objects'], regress= False)

                    tmp_guesser_prob = softmax(tmp_guesser_logits * sample['objects_mask'].float())

                    for guesser_i, i in enumerate(mask_ind.data.tolist()):
                        all_guesser_probs[i, q_idx, :] = tmp_guesser_prob[guesser_i]

                if args.log_enchidden and exp_config['logging']:
                    for enc_i, i in enumerate(mask_ind.data.tolist()):
                        enc_hidden_logging[i, q_idx, :] = encoder_hidden[enc_i]
                ##########################################

                if exp_config['decider_enabled']:
                    _decision = softmax(decision).max(-1)[1].squeeze()
                else:
                    _decision = to_var(torch.LongTensor(decision.size(0)).fill_(0))

                ########## Logging Block ################
                if exp_config['logging'] and exp_config['decider_enabled']:
                    _decision_probs = softmax(decision)[:,:,1]
                    for dec_i, i in enumerate(mask_ind.data.tolist()):
                        decision_probs[i, q_idx, :] = _decision_probs[dec_i]
                ##########################################

                decisions[mask_ind] = _decision
                _enc_mask = torch.nonzero(1-_decision).squeeze()
                mask_ind = torch.nonzero(1-decisions).squeeze()

                if len(mask_ind)==0:
                    break

                if use_dataparallel and use_cuda:
                    qgen_out = model.module.qgen.sampling(src_q=sample['src_q'][mask_ind], encoder_hidden=encoder_hidden[_enc_mask], visual_features=avg_img_features[mask_ind], greedy=True, beam_size=1)
                else:
                    qgen_out = model.qgen.sampling(src_q=sample['src_q'][mask_ind], encoder_hidden=encoder_hidden[_enc_mask], visual_features=avg_img_features[mask_ind], greedy=True, beam_size=1)

                # The below method is dropped because of the way the new QGen is trained
                # new_question_lengths = ((qgen_out != word2i["?"]).sum(dim=1) + 1).long()
                new_question_lengths = get_newq_lengths(qgen_out, word2i["?"])

                answer_predictions = oracle(
                    qgen_out,
                    sample['target_cat'][mask_ind],
                    sample['target_spatials'][mask_ind],
                    None,
                    avg_img_features[mask_ind],
                    new_question_lengths
                    )

                answer_tokens = anspred2wordtok(answer_predictions, word2i)

                history[mask_ind], history_len[mask_ind] = append_dialogue(
                    dialogue=history[mask_ind],
                    dialogue_length=history_len[mask_ind],
                    new_questions=qgen_out,
                    question_length=new_question_lengths,
                    answer_tokens=answer_tokens,
                    pad_token= pad_token)

                if dataset_args['max_no_qs']-1 == q_idx:
                    if exp_config['decider_enabled']:
                        if use_dataparallel and use_cuda:
                            encoder_hidden = model.module.encoder(history=history[mask_ind], visual_features=avg_img_features[mask_ind], history_len=history_len[mask_ind])
                            decision = model.module.decider(encoder_hidden=encoder_hidden)
                        else:
                            encoder_hidden = model.encoder(history=history[mask_ind], visual_features=avg_img_features[mask_ind], history_len=history_len[mask_ind])
                            decision = model.decider(encoder_hidden=encoder_hidden)
                        _decision = softmax(decision).max(-1)[1].squeeze()
                        decisions[mask_ind] = _decision

                        ########## Logging Block ################
                        _decision_probs = softmax(decision)[:,:,1]
                        if exp_config['logging']:
                            for dec_i, i in enumerate(mask_ind.data.tolist()):
                                decision_probs[i, q_idx+1, :] = _decision_probs[dec_i]

                    if args.log_enchidden and exp_config['logging']:
                        for enc_i, i in enumerate(mask_ind.data.tolist()):
                            enc_hidden_logging[i, q_idx, :] = encoder_hidden[enc_i]
                        ##########################################

                ########## Logging Block ################
                if exp_config['logging']:
                    if use_dataparallel and use_cuda:
                        encoder_hidden = model.module.encoder(history=history, visual_features=avg_img_features, history_len=history_len)
                        tmp_guesser_logits = model.module.guesser(encoder_hidden=encoder_hidden, spatials=sample['spatials'], objects=sample['objects'], regress= False)
                    else:
                        encoder_hidden = model.encoder(history=history, visual_features=avg_img_features, history_len=history_len)
                        tmp_guesser_logits = model.guesser(encoder_hidden=encoder_hidden, spatials=sample['spatials'], objects=sample['objects'], regress= False)

                    tmp_guesser_prob = softmax(tmp_guesser_logits * sample['objects_mask'].float())

                    if exp_config['logging']:
                        for guesser_i, i in enumerate(mask_ind.data.tolist()):
                            all_guesser_probs[i, q_idx+1, :] = tmp_guesser_prob[guesser_i]
                ##########################################


            if use_dataparallel and use_cuda:
                encoder_hidden = model.module.encoder(history=history, visual_features=avg_img_features, history_len=history_len)
                guesser_logits = model.module.guesser(encoder_hidden=encoder_hidden, spatials=sample['spatials'], objects=sample['objects'], regress= False)
            else:
                encoder_hidden = model.encoder(history=history, visual_features=avg_img_features, history_len=history_len)
                guesser_logits = model.guesser(encoder_hidden=encoder_hidden, spatials=sample['spatials'], objects=sample['objects'], regress= False)

            # Uncomment this to see the entire game dialogue
            # dials = dialtok2dial(history, i2word)
            # for dial in dials:
            #     print(dial)
            #     print(dial.count('?'))
            # raise

            batch_accuracy = calculate_accuracy(softmax(guesser_logits*sample['objects_mask'].float()), sample['target_obj'])
            accuracy.append(batch_accuracy)
            decider_perc.append(torch.sum(decisions.data)/decisions.size(0))
            print("(%03d/%03d) Accuracy: Batch %.4f, Total %.4f"%(i_batch, total_no_batches, batch_accuracy, np.mean(accuracy)))
            # raise

        ########## Logging Block ################
            if exp_config['logging']:
                no_qs_list = list()

                dials = dialtok2dial(history, i2word)
                guesser_probs = softmax(guesser_logits*sample['objects_mask'].float())
                guesses = guesser_probs.max(-1)[1]

                for bidx in range(batch_size):
                    eval_log[sample['game_id'][bidx]] = dict()
                    eval_log[sample['game_id'][bidx]]['split'] = split
                    eval_log[sample['game_id'][bidx]]['true_dialogue'] = str()
                    eval_log[sample['game_id'][bidx]]['gen_dialogue'] = dials[bidx]
                    eval_log[sample['game_id'][bidx]]['no_qs'] = dials[bidx].count('?')
                    no_qs_list.append(dials[bidx].count('?'))
                    eval_log[sample['game_id'][bidx]]['decision'] = decisions[bidx].data[0]
                    eval_log[sample['game_id'][bidx]]['decider_guess_prob'] = decision_probs[bidx].data.tolist()
                    eval_log[sample['game_id'][bidx]]['image'] = sample['image_file'][bidx]
                    eval_log[sample['game_id'][bidx]]['flickr_url'] = sample['image_url'][bidx]
                    eval_log[sample['game_id'][bidx]]['target_id'] = sample['target_obj'][bidx].data[0]
                    eval_log[sample['game_id'][bidx]]['target_cat_id'] = sample['target_cat'][bidx].data[0]
                    eval_log[sample['game_id'][bidx]]['target_cat_str'] = catid2str[str(sample['target_cat'][bidx].data[0])]
                    eval_log[sample['game_id'][bidx]]['guess_id'] = guesses[bidx].data[0]
                    eval_log[sample['game_id'][bidx]]['guess_probs'] = guesser_probs[bidx].data.tolist()
                    eval_log[sample['game_id'][bidx]]['obj_list'] = sample['objects'][bidx].data[0]
                    eval_log[sample['game_id'][bidx]]['guess_cat_id'] = sample['objects'][bidx][guesses[bidx]].data[0]
                    eval_log[sample['game_id'][bidx]]['guess_cat_str'] = catid2str[str(sample['objects'][bidx][guesses[bidx]].data[0])]
                    if args.log_enchidden:
                        eval_log[sample['game_id'][bidx]]['enc_hidden'] = enc_hidden_logging[bidx].data.tolist()
                    eval_log[sample['game_id'][bidx]]['all_guess_probs'] = all_guesser_probs[bidx].data.tolist()
                    # eval_log[sample['game_id'][bidx]][]

        if exp_config['logging']:
            # TODO Need a better name for the file.
            file_name = log_dir+split+'_GPinference_'+str(args.exp_name)+'_'+exp_config['ts']+'.json'
            with open(file_name, 'w') as f:
                json.dump(eval_log, f)
            print(file_name)
        ##########################################

        print('Time taken', time()-start)
        print(split+' accuracy', np.mean(accuracy))
        print(split+' decision percentage', np.mean(decider_perc))
        if exp_config['logging']:
            print(split+' average no qs', np.mean(no_qs_list))
