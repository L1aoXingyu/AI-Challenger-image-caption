__author__ = "sherlock"
import pickle


class Vocabulary(object):
    def __init__(self, file_word2idx=None, file_idx2word=None):
        with open(file_word2idx, 'rb') as f:
            self.word2idx = pickle.load(f)

        with open(file_idx2word, 'rb') as f:
            self.idx2word = pickle.load(f)

        self.idx = len(self.word2idx)

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
        return np.array(arr)

    def arr_to_text(self, arr):
        text = []
        for ix in arr:
            text.append(self.int_to_word(ix))
        return ''.join(text)
