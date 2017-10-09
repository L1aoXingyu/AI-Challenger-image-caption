__author__ = 'sherlock'

import os
import time
import json

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

from feature_dataset import Vocabulary, get_dataloader
from ft_model import feature_model

EOS_WORD = "<s>"
vocab = Vocabulary("../word2idx.pickle", "../idx2word.pickle")

def predict(model, dataloader):
    predict_dict = []
    for feature, name in dataloader:
        feature = Variable(feature.cuda(), volatile=True)
        output = model.sample(feature, vocab.word2idx[EOS_WORD])
        pred = vocab.arr_to_text(output)
        predict_dict.append({"image_id": name[0].split('.')[0], "caption": pred})

    with open("submission.json", "w") as f:
        json.dump(predict_dict, f)

def main():
    dataloader = get_dataloader(
        feature=["resnet", "vgg", "densenet"], batch_size=1, shuffle=False)

    ft_model = feature_model(
        in_feature=2048+512+2208,  # 2048, 512, 2208
        vocab_size=vocab.total_word,
        n_class=vocab.n_class,
        embed_dim=512,
        hidden_size=512,
        num_layers=2)

    ft_model = ft_model.cuda()
    ft_model.load_state_dict(torch.load("../feature_extraction/checkpoints/ft_model_760.pth"))
    ft_model = ft_model.eval()
    predict(ft_model, dataloader)

if __name__ == '__main__':
    main()
