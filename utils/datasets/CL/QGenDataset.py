import os
import json
import random
import numpy as np
import h5py
from PIL import Image
# from utils.datasets.N2N.gameplay.prepro import create_data_file
from torch.utils.data import Dataset

from utils.datasets.CL.prepro import create_data_file, create_qgen_data_file
from utils.create_subset import create_subset


class QGenDataset(Dataset):
    """docstring for QGenDataset."""
    def __init__(self,  split, **kwargs):
        super(QGenDataset, self).__init__()
        self.data_args = kwargs

        visual_feat_file = os.path.join(self.data_args['data_dir'],self.data_args['data_paths']['ResNet']['image_features'] )
        visual_feat_mapping_file  = os.path.join(self.data_args['data_dir'],self.data_args['data_paths']['ResNet']['img2id'] )
        self.vf = np.asarray(h5py.File(visual_feat_file, 'r')[split+'_img_features'])

        with open(visual_feat_mapping_file, 'r') as file_v:
            self.vf_mapping = json.load(file_v)[split+'2id']

        data_file_name = 'n2n_' + split + '_successful_qgen_data.json'

        if self.data_args['new_data'] or not os.path.isfile(os.path.join(self.data_args['data_dir'], data_file_name)):
            create_qgen_data_file(data_dir=self.data_args['data_dir'], data_file=self.data_args['data_paths'][split], data_args=self.data_args, vocab_file_name=self.data_args['data_paths']['vocab_file'], split=split)

        if self.data_args['my_cpu']:
            if not os.path.isfile(os.path.join(self.data_args['data_dir'], 'subset_'+data_file_name.split('.')[0]+'.json')):
                create_subset(data_dir=self.data_args['data_dir'], dataset_file_name=data_file_name, split=split)

        if self.data_args['my_cpu']:
            with open(os.path.join(self.data_args['data_dir'], 'subset_'+data_file_name.split('.')[0]+'.json'), 'r') as f:
                self.data = json.load(f)
        else:
            with open(os.path.join(self.data_args['data_dir'], data_file_name), 'r') as f:
                self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not type(idx) == str:
            idx = str(idx)

        image_file = self.data[idx]['image_file']
        # load image features
        visual_feat_id = self.vf_mapping[image_file]
        visual_feat = self.vf[visual_feat_id]
        ImgFeat = visual_feat

        _data = dict()
        _data['history'] = np.asarray(self.data[idx]['history'])
        _data['history_len'] = self.data[idx]['history_len']
        _data['src_q'] = np.asarray(self.data[idx]['src_q'])
        _data['target_q'] = np.asarray(self.data[idx]['target_q'])
        _data['tgt_len'] = self.data[idx]['tgt_len']
        _data['image'] = ImgFeat
        _data['game_id'] = self.data[idx]['game_id']
        _data['image_url'] = self.data[idx]['image_url']

        return _data
