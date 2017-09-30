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

feature_loader = get_loader(batch_size=256, shuffle=False)


def get_label():
    img_label = [label for _, label in feature_loader]

    img_label = [j for i in img_label for j in i]
    if not os.path.exists('./feature_label.pickle'):
        with open('feature_label.pickle', 'ab+') as f:
            pickle.dump(img_label, f)

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
        feature = feature.cpu().data.numpy()
        feature = np.repeat(feature, 5, 0)
        img_ft.append(feature)

    img_ft = np.concatenate(img_ft, 0)
    with h5py.File(model_name + ".hd5f", 'w') as h:
        h.create_dataset(model_name, data=img_ft)

    print('Finish ' + model_name + " feature extraction!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=str, help='model name')
    opt = parser.parse_args()
    print(opt)
    get_label()
    # get_feature(opt.m)


if __name__ == '__main__':
    main()
