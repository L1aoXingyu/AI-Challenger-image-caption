__author__ = "sherlock"
import json
import os
import pickle

import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms

PAD_WORD = "<blank>"
UNK_WORD = "<unkown>"
EOS_WORD = "<s>"


class Vocabulary(object):
    def __init__(self, file_word2idx=None, file_idx2word=None):
        with open(file_word2idx, 'rb') as f:
            self.word2idx = pickle.load(f)

        with open(file_idx2word, 'rb') as f:
            self.idx2word = pickle.load(f)

        self.idx = len(self.word2idx)

    @property
    def total_word(self):
        return len(self.word2idx)

    @property
    def n_class(self):
        return len(self.word2idx) - 1

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def word_to_int(self, word):
        if word in self.word2idx:
            return self.word2idx[word]
        else:
            return len(self.word2idx)

    def int_to_word(self, index):
        if index == len(self.word2idx):
            return UNK_WORD
        elif index < len(self.word2idx):
            return self.idx2word[index]
        else:
            raise Exception('Unknown index!')

    def text_to_arr(self, text):
        arr = []
        for word in text:
            arr.append(self.word_to_int(word))
        return arr

    def arr_to_text(self, arr):
        text = []
        for ix in arr:
            text.append(self.int_to_word(ix))
        return ''.join(text)


vocab = Vocabulary("../word2idx.pickle", "../idx2word.pickle")


class MyDataset(data.Dataset):
    def __init__(self, vocab_dict, img_path, json_path, transform):
        self.vocab_dict = vocab_dict
        with open(json_path, 'r') as f:
            self.caption = json.load(f)

        self.img_path = img_path
        self.transform = transform

    def __getitem__(self, index):
        img_path = os.path.join(self.img_path, self.caption[index]['image_id'])
        img = Image.open(img_path)
        img = self.transform(img)

        cap = self.caption[index]['caption']
        new_cap = []
        for each in cap:
            temp = self.vocab_dict.text_to_arr(each) + [
                vocab.word_to_int(EOS_WORD)
            ]
            new_cap.append(temp)
        return img, new_cap

    def __len__(self):
        return len(self.caption)


def img_transform(x):
    x = x.resize((224, 224))
    x = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])(x)
    return x


def collate_fn(batch):
    img, label = zip(*batch)  # label is a list with 5 items in each
    img = torch.stack(img, 0)
    label = [i for each in label for i in each]
    return img, label


def get_loader(img_transform=img_transform,
               batch_size=32,
               shuffle=True,
               collate_fn=collate_fn):
    vocab_dict = Vocabulary('../word2idx.pickle', '../idx2word.pickle')
    dset = MyDataset(
        vocab_dict,
        img_path=
        '../data/ai_challenger_caption_train_20170902/caption_train_images_20170902',
        json_path=
        '../data/ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json',
        transform=img_transform)

    return data.DataLoader(
        dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
