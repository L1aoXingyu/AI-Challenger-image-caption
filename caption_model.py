import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import models

PAD = 0


class encoder(nn.Module):
    """
    define convolution network to extract features from image,
    this is RNN first sequence element
    """

    def __init__(self, embed_dim, model_name):
        super(encoder, self).__init__()
        if model_name == 'resnet':
            self.model = models.resnet152(pretrained=True)
        elif model_name == 'inception':
            self.model = models.inception_v3(pretrained=True)
        elif model_name == 'dense':
            self.model = models.densenet169(pretrained=True)

        self.feature = nn.Sequential(*list(self.model.children())[:-1])
        self.fc = nn.Linear(
            list(self.model.children())[-1].in_features, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim, momentum=0.01)

        self.init_weight()

    def init_weight(self):
        nn.init.normal(self.fc.weight)
        nn.init.normal(self.fc.bias)

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn(x)
        return x


class decoder(nn.Module):
    """
    define seq2seq model to get the text from convolution encoder hidden state
    """

    def __init__(self, total_vocab, embed_dim, hidden_size, num_layers,
                 dropout):
        super(decoder, self).__init__()
        self.num_layers = num_layers
        self.word2vec = nn.Embedding(total_vocab, embed_dim, padding_idx=PAD)
        self.rnn = nn.GRU(
            embed_dim, hidden_size, num_layers=num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, total_vocab)
        self.init_weight()

    def init_weight(self):
        nn.init.uniform(self.word2vec.weight, -0.1, 0.1)
        nn.init.uniform(self.fc.weight, -0.1, 0.1)
        nn.init.constant(self.fc.bias, 0)

    def forward(self, x, feature, lengths):
        """
        x: batch x length
        feature: batch x embed_dim
        """
        embedding = self.word2vec(x)  # b x l x embed
        combine_input = torch.cat((feature.unsqueeze(1), embedding),
                                  1)  # b x (1 + l) x embed
        combine_input = combine_input.permute(1, 0, 2)  # (1 + l) x b x embd
        combine_input = pack_padded_sequence(combine_input, lengths)
        out, _ = self.rnn(combine_input)
        output = self.fc(out[0])
        return output


class CaptionModel(nn.Module):
    """
    define generating caption model, which combines convolution encode network and seq2seq decode network
    """

    def __init__(self,
                 embed_dim,
                 model_name,
                 total_vocab,
                 hidden_size,
                 num_layers,
                 encoder=encoder,
                 decoder=decoder,
                 dropout=0.5):
        super(CaptionModel, self).__init__()
        self.encoder = encoder(embed_dim, model_name)
        self.decoder = decoder(total_vocab, embed_dim, hidden_size, num_layers,
                               dropout)

    def get_train_param(self):
        for param in self.encoder.feature:
            param.requires_grad = False

        return list(self.encoder.parameters()) + list(
            self.decoder.parameters())

    def forward(self, img, seq, lengths):
        feature = self.encoder(img)
        out = self.decoder(seq, feature, lengths)
        return out

    def sample(self, x):
        pass
