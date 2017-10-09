import pickle

import h5py
import numpy as np
import torch
from torch.utils import data

UNK_WORD = "<unknown>"
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
            return self.word2idx[UNK_WORD]

    def int_to_word(self, index):
        if index < len(self.word2idx):
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


class feature_dset(data.Dataset):
    def __init__(self, feature=['resnet', 'vgg', 'densenet']):
        self.feature = feature
        with open("./img_name.pickle", "rb") as f:
            self.name = pickle.load(f)

    def __getitem__(self, index):
        idx_feature = []
        for each in self.feature:
            file = each + "_test.hd5f"
            with h5py.File(file, 'r') as f:
                temp_ft = f[each][index]
            idx_feature.append(temp_ft)
        idx_feature = np.concatenate(idx_feature)
        idx_feature = torch.from_numpy(idx_feature)

        return idx_feature, self.name[index]

    def __len__(self):
        return len(self.name)


def collate_fn(batch):
    ft, name = zip(*batch)
    return torch.stack(ft, 0), name


def get_dataloader(feature=['vgg', 'resnet', 'densenet'],
                   batch_size=32,
                   shuffle=False):
    ft_dset = feature_dset(feature)
    return data.DataLoader(
        ft_dset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=4)
