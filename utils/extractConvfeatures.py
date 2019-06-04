import h5py
from PIL import Image
import json
import torch
import torch.nn as nn
import numpy as np
from time import time
import gzip
from torchvision.models import resnet
from torchvision import transforms
from train.N2N.utils import preprocess_config, to_var
from os import listdir


def extract_features(split, img_dir, model, stage, my_cpu):
    img_list = listdir(img_dir+split+'/')

    transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(( 0.485, 0.456, 0.406 ),
                                     ( 0.229, 0.224, 0.225 ))])

    multiplier = int(2**(4-stage)) # This calculation was done for ResNet-152

    if my_cpu:
        conv_img_features = np.zeros((5, 2048//multiplier, 7*multiplier, 7*multiplier))
    else:
        conv_img_features = np.zeros((len(img_list), 2048//multiplier, 7*multiplier, 7*multiplier))
    name2id = dict()

    for i in range(len(img_list)):
        if i>=5 and my_cpu:
            break
        ImgTensor = transform(Image.open(img_dir+split+'/'+img_list[i]).convert('RGB'))
        ImgTensor = to_var(ImgTensor.view(1,3,224,224))
        feat = model(ImgTensor)
        conv_img_features[i] = feat.cpu().data.numpy()
        name2id[img_list[i]] = i

    return conv_img_features, name2id

def build_model(stage=3):
  cnn = resnet.resnet152(pretrained=True)
  layers = [
    cnn.conv1,
    cnn.bn1,
    cnn.relu,
    cnn.maxpool,
  ]
  for i in range(stage):
    name = 'layer%d' % (i + 1)
    layers.append(getattr(cnn, name))
  model = torch.nn.Sequential(*layers)
  model.cuda()
  model.eval()
  return model

if __name__ == '__main__':
    start = time()

    # Allowed are 2,3, and 4. Which gives feature maps of 28, 14 and 7 dimensions.
    stage = 3

    my_cpu = False

    print('Start')
    #TODO: Remove these hard coded parts
    if my_cpu:
        #img_dir = '/home/aashish/Documents/ProjectAI/data/GuessWhat/'
        img_dir = '/home/aashigpu/TEST_CARTESIUS/avenkate/N2N/data/'
        data_dir = '/nfs/scratch/aashigpu/GW/'
    else:
        img_dir = '/home/aashigpu/TEST_CARTESIUS/avenkate/N2N/data/'
        # Change this path according to your server storage. make sure you more than 120GB free space
        data_dir = '/nfs/scratch/aashigpu/GW/'

    splits = ['train','val', 'test']

    model = build_model(stage)

    feat_h5_file = h5py.File(data_dir+'ResNet_conv_%s_image_features.h5'%(stage), 'w')

    json_data = dict()
    for split in splits:
        print(split)
        conv_img_features, name2id = extract_features(split, img_dir, model, stage, my_cpu)
        feat_h5_file.create_dataset(name=split+'_img_features', dtype='float32', data=conv_img_features)
        json_data[split+'2id'] = name2id
    feat_h5_file.close()

    with open(data_dir+'ResNet_conv_%s_image_features2id.json'%(stage), 'w') as f:
            json.dump(json_data, f)

    print('Image Features extracted.')
    print('Time taken: ', time()-start)
