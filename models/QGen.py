import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from utils.wrap_var import to_var

use_cuda = torch.cuda.is_available()


class QGenSeq2Seq(nn.Module):
    """
    QGen hidden state is initialised by the scaled encoder output.
    The input at every time step is the word embedding concatenated
    with the visual features scaled down to 512?
    """

    def __init__(self, **kwargs):
        super(QGenSeq2Seq, self).__init__()
        """
        Parameters
        ----------
        kwargs : dict
        'vocab_size' : vocabulary size
        'embedding_dim' : dimension of the word embeddings
        'word_pad_token' : Padding token in the vocab
        'num_layers' : Number of layers in the LSTM
        'hidden_dim' : Hidden state dimension
        'visual_features_dim' : Dimension of the visual features
        'scale_visual_to' : Dimension to reduce the visual features to.
        """

        self.qgen_args = kwargs

        self.word_embedding = nn.Embedding(
            self.qgen_args['vocab_size'],
            self.qgen_args['word_embedding_dim'],
            padding_idx=self.qgen_args['word_pad_token']
        )

        self.scale_visual_to = nn.Linear(self.qgen_args['visual_features_dim'], self.qgen_args['scale_visual_to'])

        self.rnn = nn.LSTM(
            self.qgen_args['word_embedding_dim'] + self.qgen_args['scale_visual_to'],
            self.qgen_args['hidden_dim'],
            num_layers=self.qgen_args['num_layers'],
            batch_first=True
        )

        # TODO: make it get_attr for option of GRU

        self.to_logits = nn.Linear(self.qgen_args['hidden_dim'], self.qgen_args['vocab_size'])

        self.start_token = self.qgen_args['start_token']
        self.ReLU = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs : dict
        'visual_features': 2048 dim avg pool layer of ResNet 152 [Bx2048]
        'src_q': The input word sequence during teacher forcing training part. Shape: [Bx max_q_length]
        'encoder_hidden': output from the encoder. Shape: [Bx1xhidden_size]
        'lengths': target length for masking to calculate the loss

        Returns
        -------
        output : dict
        'packed_word_logits': predicted words
        """

        src_q, encoder_hidden, visual_features = kwargs['src_q'], kwargs['encoder_hidden'], kwargs['visual_features']

        lengths = kwargs['lengths']

        batch_size = encoder_hidden.size(0)

        # concatenating encoder hidden and visual features and scaling to required QGen hidden size
        hidden = encoder_hidden.transpose(1, 0)
        proj_visual_features = self.ReLU(self.scale_visual_to(visual_features))

        cell = to_var(torch.zeros(self.qgen_args['num_layers'], batch_size, self.qgen_args['hidden_dim']))

        # copy visual features for each input token
        proj_visual_features_batch = proj_visual_features.repeat(1, self.qgen_args['max_tgt_length']).view(
            batch_size,
            self.qgen_args['max_tgt_length'],
            proj_visual_features.size(1)
        )

        # get word embeddings for input tokens
        src_q_embedding = self.word_embedding(src_q)

        input = torch.cat([src_q_embedding, proj_visual_features_batch], dim=2)

        # RNN forward pass
        # Aapparently hidden and cell have to be in num_layers x batch_size x hidden_dim
        rnn_hiddens, _ = self.rnn(input, (hidden, cell))

        rnn_hiddens.contiguous()

        word_logits = self.to_logits(rnn_hiddens.view(-1, self.qgen_args['hidden_dim'])).view(
            batch_size,
            self.qgen_args['max_tgt_length'],
            self.qgen_args['vocab_size']
        )

        # Packing them to act as an alternative to masking
        # packed_word_logits = pack_padded_sequence(word_logits, list(lengths.data), batch_first=True)[0]

        return word_logits

    # TODO Beam Search and Gumbel Smapler.

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
        logits = self.to_logits(rnn_hiddens.view(-1, self.qgen_args['hidden_dim'])).view(embedding.size(0), 1,
                                                                                         self.qgen_args['vocab_size'])

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
        return_logits_and_hidden_states = kwargs.get('return_logits_and_hidden_states', False)
        beam = kwargs.get('beam_size', 1)

        if not greedy:
            temp = kwargs.get('temp', 1)

        if beam > 1:
            return self.beam_search(**kwargs)

        encoder_hidden, visual_features = kwargs['encoder_hidden'], kwargs['visual_features']

        batch_size = encoder_hidden.size(0)

        start_tokens = to_var(torch.LongTensor(batch_size, 1).fill_(self.start_token))

        hidden = encoder_hidden.transpose(1, 0)
        proj_visual_features = self.ReLU(self.scale_visual_to(visual_features)).unsqueeze(1)

        cell = to_var(torch.zeros(self.qgen_args['num_layers'], batch_size, self.qgen_args['hidden_dim']))

        start_embedding = self.word_embedding(start_tokens)

        _embedding = torch.cat([start_embedding, proj_visual_features], dim=2)

        rnn_state = (hidden, cell)

        sampled_q_logits = []
        sampled_q_tokens = []
        decoder_hidden_states = []
        for i in range(self.qgen_args['max_tgt_length']):
            if i > 0:
                word_embedding = self.word_embedding(word_id)
                _embedding = torch.cat([word_embedding, proj_visual_features], dim=2)
            decoder_hidden_states.append(rnn_state[0].squeeze(0))
            logits, rnn_state = self.basicforward(embedding=_embedding, rnn_state=rnn_state)

            if greedy:
                word_prob, word_id = self.softmax(logits).max(-1)
            else:
                # Make sure that temp is between 0.1-1 for good results. temp=0 will throw an error.
                probabilities = self.softmax(logits / temp).squeeze()
                m = torch.distributions.Categorical(probabilities)
                tmp_token = m.sample()
                word_prob = m.log_prob(tmp_token).view(batch_size, 1)
                word_id = tmp_token.long().view(batch_size, 1)

            sampled_q_logits.append(word_prob)
            sampled_q_tokens.append(word_id)

        sampled_q_logits = torch.cat(sampled_q_logits, dim=1)
        sampled_q_tokens = torch.cat(sampled_q_tokens, dim=1)

        if greedy:
            sampled_q_logits = torch.log(sampled_q_logits)

        if return_logits_and_hidden_states:
            return sampled_q_tokens, sampled_q_logits, decoder_hidden_states
        else:
            return sampled_q_tokens

    