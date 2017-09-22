import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

PAD = 0

class feature_model(nn.Module):
    """
    This is feature vector model, use pretrained model to get image features, then use this feature map to generate caption
    """

    def __init__(self,
                 in_feature,
                 vocab_size,
                 embed_dim,
                 hidden_size,
                 num_layers,
                 dropout=0.5):
        super(feature_model, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_feature, embed_dim),
            nn.BatchNorm1d(embed_dim, momentum=0.01))
        self.word2vec = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.rnn = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout)
        self.classifer = nn.Linear(hidden_size, vocab_size)

        self.init_weight()

    def init_weight(self):
        nn.init.normal(self.project[0].weight, 0, 0.02)
        nn.init.constant(self.project[0].bias, 0)
        nn.init.uniform(self.word2vec.weight, -0.1, 0.1)
        nn.init.uniform(self.classifer.weight, -0.1, 0.1)
        nn.init.constant(self.classifer.bias, 0)

    def forward(self, ft, label, seq_len):
        """
        ft: image feature map
        label: batch x length
        seq_len: label length list
        """
        proj_ft = self.project(ft)  # b x embed
        embedding = self.word2vec(label)  # b x le x embed
        combine_input = torch.cat((proj_ft.unsqueeze(1), embedding), 1)
        combine_input = combine_input.permute(1, 0, 2)  # (len + 1) x b x embed
        combine_input = pack_padded_sequence(combine_input, seq_len)
        out, _ = self.rnn(combine_input)
        output = self.classifer(out[0])
        return output