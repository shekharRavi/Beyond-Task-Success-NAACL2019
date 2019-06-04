import os
import json
import numpy as np
import h5py
from PIL import Image
from utils.datasets.SL.prepro import create_data_file
from torch.utils.data import Dataset
from torchvision import transforms

from utils.create_subset import create_subset

class N2NDataset(Dataset):
    def __init__(self, split='train', **kwargs):
        # data_dir, data_file, data_args, vocab_file_name,
        self.data_args = kwargs

        visual_feat_file = os.path.join(self.data_args['data_dir'],self.data_args['data_paths']['ResNet']['image_features'] )
        visual_feat_mapping_file  = os.path.join(self.data_args['data_dir'],self.data_args['data_paths']['ResNet']['img2id'] )
        self.vf = np.asarray(h5py.File(visual_feat_file, 'r')[split+'_img_features'])

        with open(visual_feat_mapping_file, 'r') as file_v:
            self.vf_mapping = json.load(file_v)[split+'2id']

        tmp_key = split + "_process_file"

        if tmp_key in self.data_args['data_paths']:
            data_file_name = self.data_args['data_paths'][tmp_key]
        else:
            if self.data_args['successful_only']:
                data_file_name = 'n2n_'+split+'_successful_data.json'
            else:
                data_file_name = 'n2n_'+split+'_all_data.json'

        if self.data_args['new_data'] or not os.path.isfile(os.path.join(self.data_args['data_dir'], data_file_name)):
            create_data_file(data_dir=self.data_args['data_dir'], data_file=self.data_args['data_paths'][split], data_args=self.data_args, vocab_file_name=self.data_args['data_paths']['vocab_file'], split=split)

        if self.data_args['my_cpu']:
            if not os.path.isfile(os.path.join(self.data_args['data_dir'], 'subset_'+split+'.json')):
                create_subset(data_dir=self.data_args['data_dir'], dataset_file_name=data_file_name, split=split)

        if self.data_args['my_cpu']:
            with open(os.path.join(self.data_args['data_dir'], 'subset_'+split+'.json'), 'r') as f:
                self.n2n_data = json.load(f)
        else:
            with open(os.path.join(self.data_args['data_dir'], data_file_name), 'r') as f:
                self.n2n_data = json.load(f)

    def __len__(self):
        return len(self.n2n_data)

    def __getitem__(self, idx):

        if not type(idx) == str:
            idx = str(idx)

        image_file = self.n2n_data[idx]['image_file']
        # load image features
        visual_feat_id = self.vf_mapping[image_file]
        visual_feat = self.vf[visual_feat_id]
        ImgFeat = visual_feat

        _data = dict()
        _data['image'] = ImgFeat
        _data['history'] = np.asarray(self.n2n_data[idx]['history'])
        _data['history_len'] = self.n2n_data[idx]['history_len']
        _data['src_q'] = np.asarray(self.n2n_data[idx]['src_q'])
        _data['target_q'] = np.asarray(self.n2n_data[idx]['target_q'])
        _data['tgt_len'] = self.n2n_data[idx]['tgt_len']
        _data['decider_tgt'] = int(self.n2n_data[idx]['decider_tgt'])
        _data['objects'] = np.asarray(self.n2n_data[idx]['objects'])
        _data['objects_mask'] = np.asarray(1-np.equal(self.n2n_data[idx]['objects'], np.zeros(len(self.n2n_data[idx]['objects']))))
        _data['spatials'] = np.asarray(self.n2n_data[idx]['spatials'])
        _data['target_obj'] = self.n2n_data[idx]['target_obj']
        _data['target_cat'] = self.n2n_data[idx]['target_cat']
        _data['game_id'] = self.n2n_data[idx]['game_id']
        _data['bboxes'] = np.asarray(self.n2n_data[idx]['bboxes'])
        _data['image_url'] = self.n2n_data[idx]['image_url']

        return _data
