from torch import nn, torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from lib.nn.ops import GaussianNoise


class Embed(nn.Module):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 embeddings=None,
                 noise=.0,
                 dropout=.0,
                 trainable=False):
        super(Embed, self).__init__()

        # define the embedding layer, with the corresponding dimensions
        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=embedding_dim)

        if embeddings is not None:
            print("Initializing Embedding layer with pre-trained weights!")
            self.init_embeddings(embeddings, trainable)

        # the dropout "layer" for the word embeddings
        self.dropout = nn.Dropout(dropout)

        # the gaussian noise "layer" for the word embeddings
        self.noise = GaussianNoise(noise)

    def init_embeddings(self, weights, trainable):
        self.embedding.weight = nn.Parameter(torch.from_numpy(weights),
                                             requires_grad=trainable)

    def forward(self, x):
        """
        This is the heart of the model. This function, defines how the data
        passes through the network.
        Args:
            x: the input data (the sentences)

        Returns: the logits for each class

        """
        embeddings = self.embedding(x)

        if self.noise.stddev > 0:
            embeddings = self.noise(embeddings)

        if self.dropout.p > 0:
            embeddings = self.dropout(embeddings)

        return embeddings


class RNNEncoder(nn.Module):
    def __init__(self, input_size, rnn_size, num_layers,
                 bidirectional, dropout):
        """
        A simple RNN Encoder.

        Args:
            input_size (int): the size of the input features
            rnn_size (int)
            num_layers (int)
            bidirectional (bool)
            dropout (float)

        Returns: outputs, last_outputs
        - **outputs** of shape `(batch, seq_len, hidden_size)`:
          tensor containing the output features `(h_t)`
          from the last layer of the LSTM, for each t.

        - **last_outputs** of shape `(batch, hidden_size)`:
          tensor containing the last output features
          from the last layer of the LSTM, for each t=seq_len.

        """
        super(RNNEncoder, self).__init__()

        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=rnn_size,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           dropout=dropout,
                           batch_first=True)

        # the dropout "layer" for the output of the RNN
        self.drop_rnn = nn.Dropout(dropout)

        # define output feature size
        self.feature_size = rnn_size
        if bidirectional:
            self.feature_size *= 2

    @staticmethod
    def last_by_index(outputs, lengths):
        # Index of the last output for each sequence.
        # print(lengths)
        # torch.Size([4, 4])
        # >>> y = x.view(16)
        # >>> y.size()
        # torch.Size([16])
        idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
                                               outputs.size(2)).unsqueeze(1)    # unsqueeze D1 *1
        # print('idx')
        # print(idx)
        # print(outputs.gather(1, idx).squeeze())
        # torch.gather(dim, index, out=None) → Tensor
        return outputs.gather(1, idx).squeeze()

    @staticmethod
    def split_directions(outputs):
        direction_size = int(outputs.size(-1) / 2)
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    def last_timestep(self, outputs, lengths, bi=False):
        if bi:
            forward, backward = self.split_directions(outputs)
            # tensor([[[-0.0050,  0.0346, -0.0540,  ...,  0.2081,  0.0078, -0.3285],
            #          [ 0.0382,  0.0580, -0.0032,  ...,  0.3200,  0.0821, -0.4469],
            #          [ 0.0598,  0.1161,  0.0144,  ...,  0.4412,  0.0225, -0.4357],
            #          ...,
            #          [ 0.0147,  0.0658,  0.0158,  ...,  0.4789,  0.0061, -0.4225],
            #          [ 0.0247,  0.0505,  0.0365,  ...,  0.3581,  0.0228, -0.2056],
            #          [ 0.0867,  0.1089, -0.2493,  ...,  0.5207,  0.0136, -0.2245]]],
            #        device='cuda:0', grad_fn=<SliceBackward>)
            # torch.Size([1, 16, 250])
            # print(forward)
            # print(forward.size())

            # print(backward)
            # print(backward.size())
            last_forward = self.last_by_index(forward, lengths)
            if len(last_forward.size()) == 1:
                last_forward = last_forward.unsqueeze(0)
            last_backward = backward[:, 0, :]   # 1*1*250
            # print("last forward and backward demension")
            # print(last_forward)
            # print(last_forward.size())
            # print(last_backward)
            # print(last_backward.size())
            return torch.cat((last_forward, last_backward), dim=-1)

        else:
            return self.last_by_index(outputs, lengths)

    def forward(self, embs, lengths):
        # pack the batch
        # print(embs.size()) # torch.Size([1, 85, 310])
        # print(lengths.size())

        packed = pack_padded_sequence(embs, list(lengths.data),
                                      batch_first=True)
        out_packed, _ = self.rnn(packed)
        # unpack output - no need if we are going to use only the last outputs
        outputs, _ = pad_packed_sequence(out_packed, batch_first=True)

        # get the outputs from the last *non-masked* timestep for each sentence
        last_outputs = self.last_timestep(outputs, lengths,
                                          self.rnn.bidirectional)
        # apply dropout to the outputs of the RNN
        last_outputs = self.drop_rnn(last_outputs)

        return outputs, last_outputs
