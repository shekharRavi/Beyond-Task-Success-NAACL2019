import h5py
from PIL import Image
import json
import torch
import numpy as np
from time import time
from models.CNN import ResNet
from torchvision import transforms
from train.N2N.utils import to_var
from os import listdir


def extract_features(split, img_dir, model, my_cpu = False):

    img_list = listdir(img_dir+split+'/')

    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(( 0.485, 0.456, 0.406 ),
                                 ( 0.229, 0.224, 0.225 ))])

    if my_cpu:
        avg_img_features = np.zeros((5, 2048))
    else:
        avg_img_features = np.zeros((len(img_list), 2048))

    name2id = dict()

    for i in range(len(img_list)):
        if i>=5 and my_cpu:
            break
        ImgTensor = transform(Image.open(img_dir+split+'/'+img_list[i]).convert('RGB'))
        ImgTensor = to_var(ImgTensor.view(1,3,224,224))
        conv_features, feat = model(ImgTensor)
        avg_img_features[i] = feat.cpu().data.numpy()
        name2id[img_list[i]] = i

    return avg_img_features, name2id


if __name__ == '__main__':
    start = time()
    print('Start')
    splits = ['train','val', 'test']

    my_cpu = True
    # TODO: Remove these hard coded parts
    if my_cpu:
        img_dir = '/home/aashigpu/TEST_CARTESIUS/avenkate/N2N/data/'
    else:
        img_dir = '/home/aashish/Documents/ProjectAI/data/GuessWhat/'

    model = ResNet()
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    feat_h5_file = h5py.File('ResNet_avg_image_features.h5', 'w')
    json_data = dict()
    for split in splits:
        print(split)
        avg_img_features, name2id = extract_features(split, img_dir, model, my_cpu)
        feat_h5_file.create_dataset(name=split+'_img_features', dtype='float32', data=avg_img_features)
        json_data[split+'2id'] = name2id
    feat_h5_file.close()

    with open('ResNet_avg_image_features2id.json', 'w') as f:
            json.dump(json_data, f)

    print('Image Features extracted.')
    print('Time taken: ', time()-start)
