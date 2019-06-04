import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import log_softmax
from utils.wrap_var import to_var
"""
Guesser
"""

use_cuda = torch.cuda.is_available()

class Guesser(nn.Module):
    """
    Assumption that encoder hidden is given which is then used for dot product with other elements.
    """
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        kwargs : dict
            'no_categories' : Total number of categories in the dataset
            'obj_categories_embedding_dim' : Dimension of the object category embedding
            'pad_token' : Pad token for object categories
            'layer_sizes' : Architecture of the mlp. Architecture should be given as [inp, hidden1,..., output]

        """
        super(Guesser, self).__init__()

        self.guesser_args = kwargs

        self.obj_categories_embedding = nn.Embedding(self.guesser_args['no_categories'], self.guesser_args['obj_categories_embedding_dim'], padding_idx=self.guesser_args['obj_pad_token'])

        self.mlp = nn.Sequential()

        idx = 0
        for i in range(len(self.guesser_args['layer_sizes'])-1):
            self.mlp.add_module(str(idx), nn.Linear(self.guesser_args['layer_sizes'][i], self.guesser_args['layer_sizes'][i+1]))
            idx += 1
            self.mlp.add_module(str(idx), nn.ReLU())
            idx += 1

    def forward(self, **kwargs):
        """GW baseline Guesser with encoder hidden state as the input instead of standalone LSTM.

        Parameters
        ----------
        **kwargs : dict
            'encoder_hidden' : output from the encoder. Shape: [Bx1xhidden_size]
            'spatials' : spatial features for all the objects. Shape: [Bx20x8]
            'objects' : list of objects category ids. Shape: [Bx20]
            if regress:
            'regress' : bool. If True returns the target object category embedding
            'target_cat' : Index of the target category embedding

        Returns
        -------
        output : dict
            'logits' : log softmax of the predictions made by Guesser

        """

        regress  = kwargs['regress']
        #Not used in the NAACL'19 paper
        if regress:
            target_cat = kwargs['target_cat']
            return self.obj_categories_embedding(target_cat)
            #TODO add documentation for this. Regression not supported for now.

        encoder_hidden = kwargs['encoder_hidden'].squeeze(1)
        spatials = kwargs['spatials']
        objects = kwargs['objects']

        batch_size = encoder_hidden.size(0)

        objects_embedding = self.obj_categories_embedding(objects)

        mlp_in = torch.cat([objects_embedding, spatials.float()], dim=2).view(-1, self.guesser_args['obj_categories_embedding_dim']+8)

        mlp_out = self.mlp(mlp_in)

        mlp_out = mlp_out.view(batch_size, -1, encoder_hidden.size(1))

        logits = to_var(torch.Tensor(mlp_out.size(0), mlp_out.size(1)))

        # TODO make this batch wise. Check the solution given by Gabe later

        for bid in range(mlp_out.size(0)):
            logits[bid] = torch.matmul(mlp_out[bid], encoder_hidden[bid])

        return logits
