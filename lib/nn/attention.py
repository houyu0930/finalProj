from torch import nn, torch
from torch.autograd import Variable

from lib.config import DEVICE


class SelfAttention(nn.Module):
    def __init__(self, attention_size,
                 layers=1,
                 dropout=.0,
                 non_linearity="tanh"):
        super(SelfAttention, self).__init__()

        # self.batch_first = batch_first

        if non_linearity == "relu":
            activation = nn.ReLU()
        elif non_linearity == "lrelu":
            activation = nn.LeakyReLU()
        else:
            activation = nn.Tanh()

        modules = []
        for i in range(layers - 1):
            modules.append(nn.Linear(attention_size, attention_size))
            modules.append(activation)
            modules.append(nn.Dropout(dropout))

        # last attention layer must output 1
        modules.append(nn.Linear(attention_size, 1))
        modules.append(activation)
        modules.append(nn.Dropout(dropout))

        self.attention = nn.Sequential(*modules)

        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def get_mask(attentions, lengths):
        """
        Construct mask for padded itemsteps, based on lengths
        """
        max_len = max(lengths.data)
        # Detaches the Tensor from the graph that created it, making it a leaf.
        # Views cannot be detached in-place.
        mask = Variable(torch.ones(attentions.size())).detach() # Variable/Tensor
        mask = mask.to(DEVICE)

        for i, l in enumerate(lengths.data):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0 # [1 1 1 0 0 0 0 ...]
        return mask

    def forward(self, inputs, lengths):

        # STEP 1 - perform dot product of the attention vector and each hidden state

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        # for the first batch: e.g. len=2, hidden_size=3
        # tensor([[-1.7301, -1.3598, -0.7204],
        #         [-1.6151,  1.4487,  2.6711]])
        # tensor([[0.1926, 0.2789, 0.5285],
        #         [0.0105, 0.2251, 0.7644]])
        # (A×1×B×C×1×D) then the out tensor will be of shape: (A×B×C×D).
        # After attention (linear + tanh)
        # [ [att1] [att2] ... ] => [att1 att2 ...]
        scores = self.attention(inputs).squeeze()
        scores = self.softmax(scores)

        # Step 2 - Masking

        # construct a mask, based on sentence lengths
        mask = self.get_mask(scores, lengths)
        # apply the mask - zero out masked timesteps
        masked_scores = scores * mask
        # re-normalize the masked scores
        _sums = masked_scores.sum(-1, keepdim=True)  # sums per row
        scores = masked_scores.div(_sums)  # divide by row sum

        # Step 3 - Weighted sum of hidden states, by the attention scores

        # multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))
        # sum the hidden states
        representations = weighted.sum(1).squeeze()

        return representations, scores
