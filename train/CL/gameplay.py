import torch
import torch.nn as nn

from utils.wrap_var import to_var
from utils.gameplayutils import *

use_cuda = torch.cuda.is_available()

def gameplay_fwpass(q_model, o_model, inputs, exp_config, word2i, train= True):
    """Assumption: Takes in models and batch level input to give the guesser logits.

    Parameters
    ----------
    q_model : type
        Description of parameter `q_model`.
    o_model : type
        Description of parameter `o_model`.
    inputs : type
        Description of parameter `inputs`.

    Returns
    -------
    guesser_logits: type
        Description of returned object.

    """
    softmax = nn.Softmax(dim=-1)

    use_dataparallel = exp_config['use_dataparallel']
    avg_img_features = inputs['image']

    batch_size = avg_img_features.size(0)

    history = to_var(torch.LongTensor(batch_size, 200).fill_(inputs['pad_token']))
    history[:,0] = inputs['history']
    history_len = inputs['history_len']

    decisions = to_var(torch.LongTensor(batch_size).fill_(0))
    mask_ind = torch.nonzero(1-decisions).squeeze()
    _enc_mask = mask_ind

    for q_idx in range(exp_config['max_no_qs']):

        if use_dataparallel and use_cuda:
            encoder_hidden = q_model.module.encoder(history=history[mask_ind], visual_features=avg_img_features[mask_ind], history_len=history_len[mask_ind])
            if exp_config['decider'] == 'decider_seq':
                if q_idx == 0:
                    decision, rnn_state = q_model.module.decider.basicforward(encoder_hidden=encoder_hidden, q_idx=q_idx)
                else:
                    decision, rnn_state = q_model.module.decider.basicforward(encoder_hidden=encoder_hidden, q_idx=q_idx, rnn_state=(rnn_state[0][:,_enc_mask.data,:], rnn_state[1][:,_enc_mask.data,:]))
            else:
                decision = q_model.module.decider(encoder_hidden=encoder_hidden)
        else:
            encoder_hidden = q_model.encoder(history=history[mask_ind], visual_features=avg_img_features[mask_ind], history_len=history_len[mask_ind])
            if exp_config['decider'] == 'decider_seq':
                if q_idx == 0:
                    decision, rnn_state = q_model.decider.basicforward(encoder_hidden=encoder_hidden, q_idx=q_idx)
                else:
                    decision, rnn_state = q_model.decider.basicforward(encoder_hidden=encoder_hidden, q_idx=q_idx, rnn_state=(rnn_state[0][:,_enc_mask.data,:], rnn_state[1][:,_enc_mask.data,:]))
            else:
                decision = q_model.decider(encoder_hidden=encoder_hidden)


        if exp_config['decider_enabled']:
            _decision = softmax(decision).max(-1)[1].squeeze()
        else:
            _decision = to_var(torch.LongTensor(decision.size(0)).fill_(0))

        decisions[mask_ind] = _decision
        _enc_mask = torch.nonzero(1-_decision).squeeze()
        mask_ind = torch.nonzero(1-decisions).squeeze()

        if len(mask_ind)==0:
            break

        if use_dataparallel and use_cuda:
            qgen_out = q_model.module.qgen.sampling(src_q=inputs['src_q'][mask_ind], encoder_hidden=encoder_hidden[_enc_mask], visual_features=avg_img_features[mask_ind], greedy=True, beam_size=1)
        else:
            qgen_out = q_model.qgen.sampling(src_q=inputs['src_q'][mask_ind], encoder_hidden=encoder_hidden[_enc_mask], visual_features=avg_img_features[mask_ind], greedy=True, beam_size=1)

        new_question_lengths = get_newq_lengths(qgen_out, word2i["?"])

        answer_predictions = o_model(
            qgen_out,
            inputs['target_cat'][mask_ind],
            inputs['target_spatials'][mask_ind],
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
            pad_token= inputs['pad_token'])


    history.detach_()
    avg_img_features.detach_()
    history_len.detach_()

    if use_dataparallel and use_cuda:
        encoder_hidden = q_model.module.encoder(history=history, visual_features=avg_img_features, history_len=history_len)
        guesser_logits = q_model.module.guesser(encoder_hidden=encoder_hidden, spatials=inputs['spatials'], objects=inputs['objects'], regress= False)
    else:
        encoder_hidden = q_model.encoder(history=history, visual_features=avg_img_features, history_len=history_len)
        guesser_logits = q_model.guesser(encoder_hidden=encoder_hidden, spatials=inputs['spatials'], objects=inputs['objects'], regress= False)


    return guesser_logits
