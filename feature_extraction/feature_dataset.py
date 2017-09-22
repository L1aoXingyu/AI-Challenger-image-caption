import pickle

import h5py
import numpy as np
import torch
from torch.utils import data

PAD = 0


class Vocabulary(object):
    def __init__(self, file_word2idx=None, file_idx2word=None):
        with open(file_word2idx, 'rb') as f:
            self.word2idx = pickle.load(f)

        with open(file_idx2word, 'rb') as f:
            self.idx2word = pickle.load(f)

        self.idx = len(self.word2idx)

    def __len__(self):
        return len(self.word2idx) + 1

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


class feature_dset(data.Dataset):
    def __init__(self, feature=['vgg', 'resnet', 'densenet']):
        self.feature = feature
        with open('feature_label.pickle', 'rb') as f:
            self.label = pickle.load(f)

    def __getitem__(self, index):
        idx_feature = []
        for each in self.feature:
            file = each + ".hd5f"
            with h5py.File(file, 'r') as f:
                temp_ft = f[each][index]
            idx_feature.append(temp_ft)
        idx_feature = np.concatenate(idx_feature)
        idx_feature = torch.from_numpy(idx_feature)

        return idx_feature, self.label[index]

    def __len__(self):
        return len(self.label)


def collate_fn(batch):
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    ft, label = zip(*batch)

    # pad the longest label
    pad_label = []
    seq_len = []
    max_len = len(label[0])
    for i in label:
        temp = [PAD] * max_len
        temp[:len(i)] = i
        pad_label.append(temp)
        seq_len.append(len(i))
    pad_label = np.array(pad_label, dtype='int64')
    pad_label = torch.from_numpy(pad_label)
    return torch.stack(ft, 0), pad_label, seq_len


def get_dataloader(feature=['vgg', 'resnet', 'densenet'],
                   batch_size=32,
                   shuffle=True):
    ft_dset = feature_dset(feature)
    vocab = Vocabulary("../word2idx.pickle", "../idx2word.pickle")
    return data.DataLoader(
        ft_dset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_fn), len(vocab)
