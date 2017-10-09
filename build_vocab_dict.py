__author__ = 'sherlock'
import json
import pickle

EOS_WORD = "<s>"
PAD_WORD = "<blank>"
UNK_WORD = "<unknown>"


def build_vocab(json_file, threshold=5):
    word2idx = {}
    idx = 0
    idx2word = {}
    with open(json_file) as f:
        cap = json.load(f)
    cap_list = [
        j.replace('\n', '').replace(' ', '') for i in cap for j in i['caption']
    ]  # 得到每个句子
    word_list = [word for se in cap_list for word in se]  # 得到字符字典
    # 过滤掉频率比较低的字符
    word_set = set(word_list)
    word_count = {}
    for w in word_set:
        word_count[w] = 0
    for w in word_list:
        word_count[w] += 1
    vocab_count_list = []
    for i in word_set:
        vocab_count_list.append([i, word_count[i]])
    valid_word = [i[0] for i in vocab_count_list if i[1] > threshold]

    for word in valid_word:
        word2idx[word] = idx
        idx2word[idx] = word
        idx += 1
    word2idx[EOS_WORD] = idx
    idx2word[idx] = EOS_WORD
    idx += 1
    word2idx[UNK_WORD] = idx
    idx2word[idx] = UNK_WORD
    idx += 1
    word2idx[PAD_WORD] = idx
    idx2word[idx] = PAD_WORD

    with open('word2idx.pickle', 'ab+') as f:
        pickle.dump(word2idx, f)

    with open('idx2word.pickle', 'ab+') as f:
        pickle.dump(idx2word, f)

    print('Finish Processing!')


def main():
    build_vocab(
        '/home/node/dhn/Image_caption/image-caption/data/ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json'
    )


if __name__ == '__main__':
    main()
