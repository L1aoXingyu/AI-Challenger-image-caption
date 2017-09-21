import torch
import tqdm
from torchvision import models  # resnet152, densnet161, vgg19, inception_resnet_v2
import numpy as np
from torch.autograd import Variable
from torch import nn
from extract_dataset import get_loader
import pickle
import argparse
import os


def get_feature(model_name, feature_loader):
    if model_name == 'vgg':
        model = models.vgg19(pretrained=True)

        ext_model = nn.Sequential(*list(model.features))
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
    ext_model = ext_model.eval()

    img_ft = []
    img_label = []

    for img, label in tqdm.tqdm(feature_loader):
        img = Variable(img.cuda(), volatile=True)
        feature = ext_model(img)
        feature = feature.view(feature.size(0), feature.size(1))
        feature = feature.cpu().data.numpy()
        feature = np.repeat(feature, 5, 0)
        img_ft.append(feature)
        img_label.append(label)

    img_ft = np.concatenate(img_ft, 0)
    img_label = [j for i in img_label for j in i]
    if not os.path.exists('./feature_label.pickle'):
        with open('feature_label.pickle', 'ab+') as f:
            pickle.dump(img_label, f)
    with open(model_name + '_feature.pickle', 'ab+') as f:
        pickle.dump(img_ft, f)

    print('Finish ' + model_name + " feature extraction!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=str, help='model name')
    parser.add_argument("--bs", default=32, help="batch size")
    opt = parser.parse_args()
    print(opt)
    feature_loader = get_loader(batch_size=opt.bs, shuffle=False)
    get_feature(opt.m, feature_loader=feature_loader)


if __name__ == '__main__':
    main()