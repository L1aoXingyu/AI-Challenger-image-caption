__author__ = 'sherlock'

import os
import time

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

from feature_dataset import Vocabulary, get_dataloader
from ft_model import feature_model


def train_epoch(model, dataloader, criterion, optimizer, print_step):
    running_loss = 0.
    total_loss = 0.
    n_total = 0.
    for i, data in enumerate(dataloader):
        ft, seq, length = data  # seq: b x T
        bs = ft.size(0)
        ft = Variable(ft).cuda()
        seq = Variable(seq).cuda()
        # forward
        out = model(ft, seq, length)
        label = pack_padded_sequence(seq.permute(1, 0), length)
        loss = criterion(out, label[0])
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0] / bs
        n_total += bs
        total_loss += loss.data[0]
        if (i + 1) % print_step == 0:
            print('{}/{} avg loss: {:.5f}'.format(
                i + 1, len(dataloader), running_loss / print_step))
            running_loss = 0.
    return total_loss / n_total


def train(epochs, save_point, model, dataloader, criterion, optimizer,
          print_step):
    model.train()
    for e in range(epochs):
        print('[ epoch {} ]'.format(e + 1))
        since = time.time()
        train_loss = train_epoch(model, dataloader, criterion, optimizer,
                                 print_step)
        print('loss: {:.5f}, time: {:.1f} s'.format(train_loss,
                                                    time.time() - since))
        print()
        if (e + 1) % save_point == 0:
            if not os.path.exists('./checkpoints'):
                os.mkdir('./checkpoints')
            torch.save(model.state_dict(),
                       './checkpoints/ft_model_{}.pth'.format(e + 1))


def get_performance(out, label):
    crit = nn.CrossEntropyLoss(size_average=False)

    return crit(out, label)


def main():
    dataloader = get_dataloader(
        feature=["resnet", "vgg", "densenet"], batch_size=32, shuffle=True)

    vocab = Vocabulary("../word2idx.pickle", "../idx2word.pickle")
    ft_model = feature_model(
        in_feature=2048 + 512 + 2208,  # 2208, 512, 2048
        vocab_size=vocab.total_word,
        n_class=vocab.n_class,
        embed_dim=512,
        hidden_size=512,
        num_layers=2)

    ft_model = ft_model.cuda()

    optimizer = optim.Adam(ft_model.parameters())

    train(
        epochs=100,
        save_point=10,
        model=ft_model,
        dataloader=dataloader,
        criterion=get_performance,
        optimizer=optimizer,
        print_step=1)


if __name__ == '__main__':
    main()
