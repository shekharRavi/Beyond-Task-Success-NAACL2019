import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


use_cuda = torch.cuda.is_available()

class Oracle(nn.Module):
    """docstring for Oracle"""
    def __init__(self, no_words, no_words_feat, no_categories, no_category_feat, no_hidden_encoder,
                 mlp_layer_sizes, no_visual_feat, no_crop_feat, dropout, inputs_config,
                 scale_visual_to=None):
        super(Oracle, self).__init__()

        self.no_words               = no_words
        self.no_words_feat          = no_words_feat
        self.no_hidden_encoder      = no_hidden_encoder
        self.mlp_layer_sizes        = mlp_layer_sizes
        self.no_categories          = no_categories
        self.no_category_feat       = no_category_feat
        self.n_spatial_feat         = 8
        self.no_visual_feat         = no_visual_feat
        self.no_crop_feat           = no_crop_feat
        self.inputs_config          = inputs_config
        self.scale_visual_to        = scale_visual_to

        self.word_embeddings = nn.Embedding(self.no_words, self.no_words_feat)
        self.obj_categories_embedding = nn.Embedding(self.no_categories, self.no_category_feat)

        self.encoder = nn.LSTM(self.no_words_feat, self.no_hidden_encoder, num_layers=1, dropout=dropout, batch_first=True)

        if self.scale_visual_to != 0:
            if type(self.scale_visual_to) == int:
                self.scale_visual = nn.Linear(self.no_visual_feat, scale_visual_to)
                self.scale_crop = nn.Linear(self.no_crop_feat, scale_visual_to)
            elif type(self.scale_visual_to) == list:
                self.scale_visual = nn.Linear(self.no_visual_feat, scale_visual_to[0])
                self.scale_crop = nn.Linear(self.no_crop_feat, scale_visual_to[1])

        # configure MLP
        self.no_mlp_inputs = 0
        if self.inputs_config['question']:
            self.no_mlp_inputs += self.encoder.hidden_size
        if self.inputs_config['obj_categories']:
            self.no_mlp_inputs += self.no_category_feat
        if self.inputs_config['spatial']:
            self.no_mlp_inputs += self.n_spatial_feat
        if self.inputs_config['visual']:
            if self.scale_visual_to == 0:
               self.no_mlp_inputs += self.no_visual_feat
            else: # When visual feature is scalled use the scaled feature length
              self.no_mlp_inputs += self.scale_visual_to
        if self.inputs_config['crop']:
            self.no_mlp_inputs += self.no_crop_feat

        self.mlp_layer_sizes = [self.no_mlp_inputs] + self.mlp_layer_sizes
        self.mlp = nn.Sequential()
        idx = 0
        for i in range(len(self.mlp_layer_sizes)-1):
            self.mlp.add_module(str(idx), nn.Linear(self.mlp_layer_sizes[i], self.mlp_layer_sizes[i+1]))
            idx += 1
            if i < len(mlp_layer_sizes)-1:
                self.mlp.add_module(str(idx), nn.ReLU())
                idx += 1
            else:
                 self.mlp.add_module(str(idx), nn.LogSoftmax(dim=-1))


    def forward(self, questions, obj_categories, spatials, crop_features, visual_features, lengths):

        bs = questions.size(0)

        # sort input by quesiton length
        if isinstance(lengths, Variable):
            lengths = lengths.data

        lengths, ind    = torch.sort(lengths, 0, descending= True)
        questions       = questions[ind]
        if self.inputs_config['obj_categories']:
            obj_categories  = obj_categories[ind]
        if self.inputs_config['spatial']:
            spatials        = spatials[ind]
        if self.inputs_config['visual']:
            visual_features = visual_features[ind]
        if self.inputs_config['crop']:
            crop_features   = crop_features[ind]

        # prepare LSTM input
        questions_embedding  = self.word_embeddings(questions)
        packed_question = pack_padded_sequence(questions_embedding, list(lengths), batch_first = True)
        if self.inputs_config['obj_categories']:
            obj_categories_embeddding = self.obj_categories_embedding(obj_categories)

        outputs, _ = self.encoder(packed_question, hx=None)

        # get the outputs of the encoder of timepoint when the last word token has been feed to the encoder.
        # I.e. the outputs created by the paddings will be ignored.
        output_padded = pad_packed_sequence(outputs, batch_first = True)

        if use_cuda:
            I = lengths.view(bs, 1, 1).cuda()
            I = Variable(I.expand(questions_embedding.size(0), 1, self.encoder.hidden_size)-1).cuda()
        else:
            I = torch.LongTensor(lengths).view(bs, 1, 1)
            I = Variable(I.expand(questions_embedding.size(0), 1, self.encoder.hidden_size)-1)

        out = torch.gather(output_padded[0], 1, I).squeeze(1)

        if self.inputs_config['visual'] and self.scale_visual_to:
            visual_features = self.scale_visual(visual_features)
        if self.inputs_config['crop'] and self.scale_visual_to:
            crop_features = self.scale_crop(crop_features)

        if self.inputs_config['question']:
            mlp_in = out
        if self.inputs_config['obj_categories']:
            mlp_in = torch.cat([mlp_in, obj_categories_embeddding],1)
        if self.inputs_config['spatial']:
            mlp_in = torch.cat([mlp_in, spatials],1)
        if self.inputs_config['visual']:
            mlp_in = torch.cat([mlp_in, visual_features],1)
        if self.inputs_config['crop']:
            mlp_in = torch.cat([mlp_in, crop_features],1)

        predictions = self.mlp(mlp_in)

        # undo sorting
        _, revert_ind = ind.sort()

        predictions = predictions[revert_ind.cuda() if use_cuda else revert_ind]

        return predictions
