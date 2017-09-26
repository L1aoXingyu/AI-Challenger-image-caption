__author__ = "sherlock"
import json
import os
import pickle

import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms

UNK_WORD = "<unkown>"
EOS_WORD = "<s>"
PAD_WORD = "<blank>"


class Vocabulary(object):
    def __init__(self, file_word2idx=None, file_idx2word=None):
        with open(file_word2idx, 'rb') as f:
            self.word2idx = pickle.load(f)

        with open(file_idx2word, 'rb') as f:
            self.idx2word = pickle.load(f)

        self.idx = len(self.word2idx)

    @property
    def total_word(self):
        return len(self.word2idx) + 1

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


vocab = Vocabulary("./word2idx.pickle", "./idx2word.pickle")


class MyDataset(data.Dataset):
    def __init__(self, img_path, json_path, transform):
        with open(json_path, 'r') as f:
            self.caption = json.load(f)

        self.img_path = img_path
        self.transform = transform

    def __getitem__(self, index):
        img_path = os.path.join(self.img_path, self.caption[index]['image_id'])
        img = Image.open(img_path)
        img = self.transform(img)

        cap = np.random.choice(self.caption[index]['caption'], 1)  # 5 list
        # cap.sort(key=lambda x: len(x), reverse=True)
        # new_cap = []
        # for each in cap:
        #     temp = self.vocab_dict.text_to_arr(each) + [EOS]
        #     new_cap.append(temp)
        cap = vocab.text_to_arr(cap[0]) + [vocab.word2idx[EOS_WORD]]
        return img, cap

    def __len__(self):
        return len(self.caption)


def img_transform(x):
    x = x.resize((310, 310))
    x = transforms.Compose([
        transforms.RandomCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])(x)
    return x


def collate_fn(batch):
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    img, label = zip(*batch)
    # seperate_label = zip(*label)
    # pad_label = []
    # seq_len = []
    # for l in seperate_label:
    #     temp_label = []
    #     temp_len = []
    #     max_len = len(max(l, key=lambda x: len(x)))
    #     for i in l:
    #         temp = [PAD] * max_len
    #         temp[:len(i)] = i
    #         temp_label.append(temp)
    #         temp_len.append(len(i))
    #     temp_label = np.array(temp_label, dtype='int64')
    #     temp_label.reshape((len(l), -1))
    #     temp_label = torch.from_numpy(temp_label)
    #     pad_label.append(temp_label)
    #     seq_len.append(temp_len)

    pad_label = []
    seq_len = []
    max_len = len(label[0])
    for i in label:
        temp = [vocab.word2idx[PAD_WORD]] * max_len
        temp[:len(i)] = i
        pad_label.append(temp)
        seq_len.append(len(i))
    pad_label = np.array(pad_label, dtype='int64')
    # pad_label.reshape((len(label), -1))
    pad_label = torch.from_numpy(pad_label)
    img = torch.stack(img, 0)
    return img, pad_label, seq_len


def get_loader(img_transform=img_transform,
               batch_size=32,
               shuffle=True,
               collate_fn=collate_fn):
    dset = MyDataset(
        img_path=
        './data/ai_challenger_caption_train_20170902/caption_train_images_20170902',
        json_path=
        './data/ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json',
        transform=img_transform)

    return data.DataLoader(
        dset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_fn), vocab.total_word, vocab.n_class
