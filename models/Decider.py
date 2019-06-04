import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
"""
For different types of Deciders.
"""

class Decider(nn.Module):

    def __init__(self, **kwargs):
        super(Decider, self).__init__()
        """
        Parameters
        ----------
        **kwargs : dict
            'arch': Architecture of the decider. Architecture should be given as [input,hidden1,...,output]

        """

        self.decider_args = kwargs
        self.mlp = nn.Sequential()

        idx = 0
        for i in range(len(self.decider_args['arch'])-1):
            self.mlp.add_module(str(idx), nn.Linear(self.decider_args['arch'][i], self.decider_args['arch'][i+1]))
            idx += 1
            if i<len(self.decider_args['arch'])-2:
                self.mlp.add_module(str(idx), nn.ReLU())
                idx += 1

    def forward(self, **kwargs):
        """Decider(MLP).

        Parameters
        ----------
        **kwargs : dict
            'encoder_hidden': output from the encoder. Shape: [Bx1xhidden_size]

        Returns
        -------
        output : dict
            'decision' : decision scores w/o softmax

        """

        encoder_hidden = kwargs['encoder_hidden']

        decision = self.mlp(encoder_hidden)

        return decision

