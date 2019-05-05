from torch import nn, torch

from lib.config import DEVICE
from lib.nn.attention import SelfAttention
from lib.nn.layers import Embed, RNNEncoder


class ModelHelper:
    @staticmethod
    def _sort_by(lengths):
        """
        Sort batch data and labels by length.
        Useful for variable length inputs, for utilizing PackedSequences
        Args:
            lengths (nn.Tensor): tensor containing the lengths for the data

        Returns:
            - sorted lengths Tensor
            - sort (callable) which will sort a given iterable
                according to lengths
            - unsort (callable) which will revert a given iterable to its
                original order

        """
        batch_size = lengths.size(0)

        sorted_lengths, sorted_idx = lengths.sort() # sort(dim=-1, descending=False) -> (Tensor, LongTensor)
        _, original_idx = sorted_idx.sort(0, descending=True)
        # torch.linspace(start, end, steps=100, out=None) → Tensor => long Tensor
        reverse_idx = torch.linspace(batch_size - 1, 0, batch_size).long()

        # reverse_idx Tensor
        reverse_idx = reverse_idx.to(DEVICE)

        sorted_lengths = sorted_lengths[reverse_idx]

        def sort(iterable):
            if len(iterable.shape) > 1:
                return iterable[sorted_idx.data][reverse_idx]
            else:
                return iterable

        def unsort(iterable):
            if len(iterable.shape) > 1:
                return iterable[reverse_idx][original_idx][reverse_idx]
            else:
                return iterable

        return sorted_lengths, sort, unsort


class FeatureExtractor(nn.Module):
    def __init__(self, embeddings=None, num_embeddings=0, **kwargs):
        super(FeatureExtractor, self).__init__()
        embed_dim = kwargs.get("embed_dim", 310)
        embed_finetune = kwargs.get("embed_finetune", False)
        embed_noise = kwargs.get("embed_noise", 0.2)
        embed_dropout = kwargs.get("embed_dropout", 0.1)

        encoder_size = kwargs.get("encoder_size", 250)
        encoder_layers = kwargs.get("encoder_layers", 2)
        encoder_dropout = kwargs.get("encoder_dropout", 0.3)
        bidirectional = kwargs.get("encoder_bidirectional", True)

        attention = kwargs.get("attention", True)
        attention_layers = kwargs.get("attention_layers", 2)
        attention_dropout = kwargs.get("attention_dropout", 0.3)
        attention_activation = kwargs.get("attention_activation", "tanh")
        # self.attention_context = kwargs.get("attention_context", False)

        # define the embedding layer, with the corresponding dimensions
        if embeddings is not None:
            self.embedding = Embed(
                num_embeddings=embeddings.shape[0],
                embedding_dim=embeddings.shape[1],  # standard: embedding dimension
                embeddings=embeddings,  # fixed embedding
                noise=embed_noise,
                dropout=embed_dropout,
                trainable=embed_finetune)
            encoder_input_size = embeddings.shape[1]
        else:
            # trainable embedding
            if num_embeddings == 0:
                raise ValueError("if an embedding matrix is not passed, "
                                 "`num_embeddings` cant be zero.")
            self.embedding = Embed(
                num_embeddings=num_embeddings,
                embedding_dim=embed_dim,
                noise=embed_noise,
                dropout=embed_dropout,
                trainable=True)
            encoder_input_size = embed_dim

        # Encoders
        self.encoder = RNNEncoder(input_size=encoder_input_size,
                                  rnn_size=encoder_size,
                                  num_layers=encoder_layers,
                                  bidirectional=bidirectional,
                                  dropout=encoder_dropout)

        self.feature_size = self.encoder.feature_size

        if attention:
            att_size = self.feature_size
            self.attention = SelfAttention(att_size,
                                           layers=attention_layers,
                                           dropout=attention_dropout,
                                           non_linearity=attention_activation)

    def init_embeddings(self, weights, trainable):
        self.embedding.weight = nn.Parameter(torch.from_numpy(weights),
                                             requires_grad=trainable)

    @staticmethod
    def _mean_pooling(x, lengths):
        sums = torch.sum(x, dim=1)
        # view(-1, 1): reshape 1 column, not clear how many rows
        _lens = lengths.view(-1, 1).expand(sums.size(0), sums.size(1))
        means = sums / _lens.float()
        return means

    def forward(self, x, lengths):
        embeddings = self.embedding(x)

        attentions = None
        outputs, last_output = self.encoder(embeddings, lengths)

        if hasattr(self, 'attention'):
            representations, attentions = self.attention(outputs, lengths)
        else:
            # no attention
            representations = last_output

        return representations, attentions


class ModelWrapper(nn.Module, ModelHelper):
    def __init__(self, embeddings=None, out_size=11, num_embeddings=0,
                 finetune=None, **kwargs):
        """
        Define the layer of the model and perform the initializations
        of the layers (wherever it is necessary)
        Args:
            embeddings (numpy.ndarray): the 2D ndarray with the word vectors

        """
        super(ModelWrapper, self).__init__()

        self.feature_extractor = FeatureExtractor(embeddings,
                                                  num_embeddings,
                                                  **kwargs)

        self.feature_size = self.feature_extractor.feature_size

        self.linear = nn.Linear(in_features=self.feature_size,
                                out_features=out_size)

        # self.activation = nn.Sigmoid()

    def forward(self, x, lengths):
        """
        Defines how the data passes through the network.
        Args:
            x: the input data (the sentences)
            lengths: the lengths of each sentence

        Returns: the logits for each class

        """
        # print(lengths.size()) torch.Size([1])
        # print(lengths.size(0)) 1
        if lengths.size(0) > 1:
            # sort
            lengths, sort, unsort = self._sort_by(lengths)
            x = sort(x)

        representations, attentions = self.feature_extractor(x, lengths)

        if lengths.size(0) > 1:
            # unsort
            representations = unsort(representations)
            if attentions is not None:
                attentions = unsort(attentions)

        logits = self.linear(representations)
        # modules = []
        # modules.append(self.linear)
        # modules.append(self.activation)
        # logits = nn.Sequential(*modules)(representations)

        return logits, attentions