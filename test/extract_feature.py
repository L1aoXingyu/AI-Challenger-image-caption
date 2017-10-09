import argparse
import os
import pickle

import h5py
import numpy as np
import torch
import tqdm
from torch import nn
from torch.autograd import Variable
from torchvision import \
    models  # resnet152, densnet161, vgg19, inception_resnet_v2

from extract_dataset import get_loader

feature_loader = get_loader(batch_size=32)


def get_image_name():
    image_name = [i for _, img_name in feature_loader for i in img_name]

    if not os.path.exists('./img_name.pickle'):
        with open('img_name.pickle', 'ab+') as f:
            pickle.dump(image_name, f)
    print("Finish label extraction!")


def get_feature(model_name):
    if model_name == 'vgg':
        model = models.vgg19(pretrained=True)

        ext_model = model.features
        ext_model.add_module('avg_pool', nn.AvgPool2d(7))

    elif model_name == 'resnet':
        model = models.resnet152(pretrained=True)
        ext_model = nn.Sequential(*list(model.children())[:-1])

    elif model_name == 'densenet':
        model = models.densenet161(pretrained=True)
        ext_model = nn.Sequential(*list(model.children())[:-1])
        ext_model.add_module('avg_pool', nn.AvgPool2d(7))

    # to do:
    # add inception-resnet-v2

    ext_model = ext_model.cuda()
    ext_model = ext_model.train()

    img_ft = []

    for img, _ in tqdm.tqdm(feature_loader):
        img = Variable(img.cuda(), volatile=True)
        feature = ext_model(img)
        feature = feature.view(feature.size(0), feature.size(1))
        img_ft.append(feature)

    img_ft = torch.cat(img_ft)
    img_ft = img_ft.data.cpu().numpy()
    with h5py.File(model_name + "_test.hd5f", 'w') as h:
        h.create_dataset(model_name, data=img_ft)

    print('Finish ' + model_name + " feature extraction!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=str, help='model name')
    opt = parser.parse_args()
    print(opt)
    # get_image_name()
    get_feature(opt.m)


if __name__ == '__main__':
    main()
