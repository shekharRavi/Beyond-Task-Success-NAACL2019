import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from utils.wrap_var import to_var

use_cuda = torch.cuda.is_available()

#This is not used in the NAACL'19 work

class QGenImgCap(nn.Module):
    """docstring for QGenImgCap."""
    def __init__(self, **kwargs):
        super(QGenImgCap, self).__init__()

        self.qgen_args = kwargs

        self.word_embedding = nn.Embedding(self.qgen_args['vocab_size'], self.qgen_args['word_embedding_dim'], padding_idx=self.qgen_args['word_pad_token'])

        if self.qgen_args['visual']:
            self.scale_visual_to = nn.Linear(self.qgen_args['visual_features_dim'],                     self.qgen_args['scale_visual_to'])

            self.rnn = nn.LSTM(self.qgen_args['word_embedding_dim']+self.qgen_args['scale_visual_to']+self.qgen_args['encoder_hidden_dim'], self.qgen_args['hidden_dim'], num_layers=self.qgen_args['num_layers'], batch_first=True)
        else:
            self.rnn = nn.LSTM(self.qgen_args['word_embedding_dim']+self.qgen_args['encoder_hidden_dim'], self.qgen_args['hidden_dim'], num_layers=self.qgen_args['num_layers'], batch_first=True)

        #TODO: make it get_attr for option of GRU

        self.to_logits = nn.Linear(self.qgen_args['hidden_dim'], self.qgen_args['vocab_size'])

        self.start_token = self.qgen_args['start_token']
        self.ReLU = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, **kwargs):

        src_q, encoder_hidden, visual_features = kwargs['src_q'], kwargs['encoder_hidden'], kwargs['visual_features']

        lengths = kwargs['lengths']

        batch_size = encoder_hidden.size(0)

        # get word embeddings for input tokens
        src_q_embedding = self.word_embedding(src_q)

        encoder_hidden = encoder_hidden.squeeze(1)

        encoder_hidden_batch = encoder_hidden.repeat(1, self.qgen_args['max_tgt_length']).view(batch_size, self.qgen_args['max_tgt_length'], encoder_hidden.size(1))

        if self.qgen_args['visual']:
            proj_visual_features = self.ReLU(self.scale_visual_to(visual_features))

            # copy visual features for each input token
            proj_visual_features_batch = proj_visual_features.repeat(1, self.qgen_args['max_tgt_length']).view(batch_size, self.qgen_args['max_tgt_length'], proj_visual_features.size(1))

            input = torch.cat([src_q_embedding, proj_visual_features_batch, encoder_hidden_batch], dim=2)
        else:
            input = torch.cat([src_q_embedding, encoder_hidden_batch], dim=2)

        # RNN forward pass
        # Aapparently hidden and cell have to be in num_layers x batch_size x hidden_dim
        rnn_hiddens, _ = self.rnn(input, hx=None)

        rnn_hiddens.contiguous()

        word_logits = self.to_logits(rnn_hiddens.view(-1, self.qgen_args['hidden_dim'])).view(batch_size, self.qgen_args['max_tgt_length'], self.qgen_args['vocab_size'])

        # Packing them to act as an alternative to masking
        # packed_word_logits = pack_padded_sequence(word_logits, list(lengths.data), batch_first=True)[0]

        return word_logits

    def basicforward(self, embedding, rnn_state):
        """Short summary.

        Parameters
        ----------
        embedding :
        rnn_state :

        Returns
        -------
        logits:
        rnn_state:
        """
        rnn_hiddens, rnn_state = self.rnn(embedding, rnn_state)
        rnn_hiddens.contiguous()
        logits = self.to_logits(rnn_hiddens.view(-1, self.qgen_args['hidden_dim'])).view(embedding.size(0), 1, self.qgen_args['vocab_size'])

        return logits, rnn_state

    def sampling(self, **kwargs):
        """Short summary.

        Parameters
        ----------
        **kwargs : dict

        Returns
        -------

        """
        greedy = kwargs.get('greedy', False)
        beam = kwargs.get('beam_size', 1)
        if not greedy:
            temp = kwargs.get('temp', 1)

        if beam > 1:
            # return self.beam_search(**kwargs)
            raise NotImplementedError

        encoder_hidden, visual_features = kwargs['encoder_hidden'], kwargs['visual_features']

        batch_size = encoder_hidden.size(0)

        start_tokens = to_var(torch.LongTensor(batch_size, 1).fill_(self.start_token))
        start_embedding = self.word_embedding(start_tokens)

        if self.qgen_args['visual']:
            proj_visual_features = self.ReLU(self.scale_visual_to(visual_features)).unsqueeze(1)
            _embedding = torch.cat([start_embedding, proj_visual_features, encoder_hidden], dim=2)
        else:
            _embedding = torch.cat([start_embedding, encoder_hidden], dim=2)

        cell = to_var(torch.zeros(self.qgen_args['num_layers'], batch_size, self.qgen_args['hidden_dim']))
        hidden = to_var(torch.zeros(self.qgen_args['num_layers'], batch_size, self.qgen_args['hidden_dim']))
        rnn_state = (hidden, cell)

        sampled_q = []
        for i in range(self.qgen_args['max_tgt_length']):
            if i>0:
                word_embedding = self.word_embedding(word_id)
                if self.qgen_args['visual']:
                    _embedding = torch.cat([word_embedding, proj_visual_features, encoder_hidden], dim=2)
                else:
                    _embedding = torch.cat([word_embedding, encoder_hidden], dim=2)
            logits, rnn_state = self.basicforward(embedding=_embedding, rnn_state=rnn_state)

            if greedy:
                word_id = self.softmax(logits).max(-1)[1]
            else:
                # Make sure that temp is between 0.1-1 for good results. temp=0 will throw a error.
                word_id = torch.multinomial(self.softmax(logits/temp).squeeze(1), 1)

            sampled_q.append(word_id)

        sampled_q = torch.cat(sampled_q, dim=1)

        return sampled_q
