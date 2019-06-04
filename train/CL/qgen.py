import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import dropout
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

use_cuda = torch.cuda.is_available()

def qgen_fwpass(q_model, inputs, use_dataparallel):
    """Short summary.

    Parameters
    ----------
    q_model : type
        Description of parameter `q_model`.
    inputs : type
        Description of parameter `inputs`.

    Returns
    -------
    type
        Description of returned object.

    """
    history, history_len = inputs['history'], inputs['history_len']
    lengths = inputs['tgt_len']
    visual_features = dropout(inputs['image'], p=0.5, training=True)
    src_q = inputs['src_q']

    if use_dataparallel and use_cuda:
        encoder_hidden = q_model.module.encoder(history=history, visual_features=visual_features, history_len=history_len)
        qgen_out = q_model.module.qgen(src_q=src_q, encoder_hidden=encoder_hidden, visual_features=visual_features, lengths=lengths)
    else:
        encoder_hidden = q_model.encoder(history=history, visual_features=visual_features, history_len=history_len)
        qgen_out = q_model.qgen(src_q=src_q, encoder_hidden=encoder_hidden, visual_features=visual_features, lengths=lengths)

    return qgen_out
