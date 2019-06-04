import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils.wrap_var import to_var

use_cuda = torch.cuda.is_available()

class Encoder(nn.Module):
    """docstring for EncoderBasic."""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__()
        """Short summary.

        Parameters
        ----------
        kwargs : dict
            'vocab_size' : size of the vocabulary
            'word_embedding_dim' : Dimension of the word embeddings
            'hidden_dim' : Dimension of hidden state of the encoder LSTM
            'word_pad_token' : Pad token in the vocabulary
            'num_layers' : Number of layers of the encoder LSTM
            'visual_features_dim' : Dimension of avg pool layer of the Resnet 152
            'scale_to' : Used to scale the concatenated visual features and LSTM hidden state to be used as input to next modules
            'decider' : Depending on decider the return from the forward pass changes

        """

        self.encoder_args = kwargs

        self.word_embeddings = nn.Embedding(self.encoder_args['vocab_size'], self.encoder_args['word_embedding_dim'], padding_idx=self.encoder_args['word_pad_token'])

        self.rnn = nn.LSTM(self.encoder_args['word_embedding_dim'], self.encoder_args['hidden_dim'], num_layers=self.encoder_args['num_layers'], batch_first=True)

        # Looking for better variable name here
        self.scale_to = nn.Linear(self.encoder_args['hidden_dim']+self.encoder_args['visual_features_dim'], self.encoder_args['scale_to'])

        # Using tanh to keep the input to all other modules to be between 1 and -1
        self.tanh = nn.Tanh()

    def forward(self, **kwargs):
        """Short summary.

        Parameters
        ----------
        kwargs : dict
            'visual_features' : avg pool layer of the Resnet 152
            'history' : Dialogue history
            'history_len' : Length of dialogue history for pack_padded_sequence

        Returns
        -------
        output : dict
            'encoder_hidden' : final output from Encoder for all other modules as input

        """

        history, history_len = kwargs['history'], kwargs['history_len']

        visual_features = kwargs['visual_features']

        batch_size = history.size(0)

        if isinstance(history_len, Variable):
            history_len = history_len.data

        history_len, ind = torch.sort(history_len, dim=0, descending=True)

        history = history[ind]

        history_embedding = self.word_embeddings(history)
        packed_history = pack_padded_sequence(history_embedding, list(history_len), batch_first=True)

        if self.encoder_args['decider'] == 'decider_seq':
            history_q_lens = kwargs['history_q_lens'][ind]
            history_q_lens = history_q_lens-1 #Because the index starts from 0
            packed_hidden, (_hidden, _) = self.rnn(packed_history, hx=None)
            # The _hidden is not in batch first format
            _hidden = _hidden.transpose(1,0)

            if kwargs['mask_select']:
                hidden_padded, _ = pad_packed_sequence(packed_hidden, batch_first = True)
                history_hiddens = to_var(torch.zeros(hidden_padded.size(0), 11, hidden_padded.size(-1)))
                # 11 beacuse that is the max number of questions+1(start_token)
                for i in range(history_q_lens.size(0)):
                    for j, idx in enumerate(list(history_q_lens[i].data)):
                        if idx == -1:
                            break
                        history_hiddens[i, j,:] = hidden_padded[i, idx-1, :]

                _, revert_ind = ind.sort()
                history_hiddens = history_hiddens[revert_ind.cuda() if use_cuda else revert_ind]
                _hidden = _hidden[revert_ind.cuda() if use_cuda else revert_ind]
                encoder_hidden = self.tanh(self.scale_to(torch.cat([_hidden, visual_features.unsqueeze(1)], dim=2)))
                visual_features = visual_features.unsqueeze(1).repeat(1,11,1)
                decider_input = self.tanh(self.scale_to(torch.cat([history_hiddens, visual_features], dim=2)))

                return encoder_hidden, decider_input
            else:
                # undo the sorting
                _, revert_ind = ind.sort()
                _hidden = _hidden[revert_ind.cuda() if use_cuda else revert_ind]
                encoder_hidden = self.tanh(self.scale_to(torch.cat([_hidden, visual_features.unsqueeze(1)], dim=2)))

                return encoder_hidden
        else:
            _, (_hidden, _) = self.rnn(packed_history, hx=None)
            # The _hidden is not in batch first format
            _hidden = _hidden.transpose(1,0)

            # undo the sorting
            _, revert_ind = ind.sort()
            _hidden = _hidden[revert_ind.cuda() if use_cuda else revert_ind]

            # TODO: Visual Attention in the next level of complexity.

            # This is similar to Best of both worlds paper
            encoder_hidden = self.tanh(self.scale_to(torch.cat([_hidden, visual_features.unsqueeze(1)], dim=2)))

            return encoder_hidden
