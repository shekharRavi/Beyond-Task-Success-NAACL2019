import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.Decider import Decider
from models.Guesser import Guesser
from models.QGen import QGenSeq2Seq
from models.QGenImgCap import QGenImgCap
from models.Encoder import Encoder

"""
Putting all the models together
"""
class Ensemble(nn.Module):
    """docstring for Ensemble."""
    def __init__(self, **kwargs):
        super(Ensemble, self).__init__()
        """Short summary.

        Parameters
        ----------
        **kwargs : dict
            'encoder' : Arguments for the encoder module
            'qgen' : Arguments for the qgen module
            'guesser' : Arguments for the guesser module
            'regressor' : Arguments for the regressor module
            'decider' : Arguments for the decider module

        """

        self.ensemble_args = kwargs

        # TODO: use get_attr to get different versions of the same model. For example QGen

        self.encoder = Encoder(**self.ensemble_args['encoder'])

        # Qgen selection
        #For the NAACL 2019, we used Seq2Seq one
        if self.ensemble_args['qgen']['qgen'] == 'qgen_cap':
            self.qgen = QGenImgCap(**self.ensemble_args['qgen'])
        else:
            self.qgen = QGenSeq2Seq(**self.ensemble_args['qgen'])

        self.guesser = Guesser(**self.ensemble_args['guesser'])

        self.decider = Decider(**self.ensemble_args['decider'])

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, **kwargs):
        """Short summary.

        Parameters
        ----------
        **kwargs : dict
            'history' : The dialogue history. Shape :[Bx max_src_length]
            'history_len' : The length of the dialogue history. Shape [Bx1]
            'src_q' : The input word sequence for the QGen
            'tgt_len' : Length of the target question
            'visual_features' : The avg pool layer from ResNet 152
            'spatials' : Spatial features for the guesser. Shape [Bx20x8]
            'objects' : List of objects for guesser. Shape [Bx20]
            'mask_select' : Bool. Based on the decider target, either QGen or Guesser is used

        Returns
        -------
        ensemble_out : dict
            'decider_out' : predicted decision
            'guesser_out' : log probabilities of the objects
            'qgen_out' : predicted next question

        """
        history, history_len = kwargs['history'], kwargs['history_len']
        lengths = kwargs['tgt_len']
        visual_features = self.dropout(kwargs['visual_features'])
        src_q = kwargs['src_q']
        spatials = kwargs['spatials']
        objects = kwargs['objects']
        mask_select = kwargs['mask_select']

        encoder_hidden = self.encoder(history=history, history_len=history_len, visual_features=visual_features)
        decider_out = self.decider(encoder_hidden=encoder_hidden)

        if mask_select:
            guesser_out = self.guesser(encoder_hidden= encoder_hidden, spatials= spatials, objects= objects, regress= False)

            return decider_out, guesser_out
        else:
            qgen_out = self.qgen(src_q=src_q, encoder_hidden=encoder_hidden, visual_features=visual_features, lengths=lengths)

            return decider_out, qgen_out
