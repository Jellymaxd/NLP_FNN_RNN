
import torch
import torch.nn as nn
import torch.nn.functional as F


class FNNModel(nn.Module):

    def __init__(self, ntoken, ninp, nhid, context, sharing=False, dropinp=0.2, drophid=0.5):
        super(FNNModel, self).__init__()

        # vocab size
        self.ntoken = ntoken
        # context size
        self.contextlength = context
        # embedding dimension
        self.embdim = ninp
        # embedding matrix
        self.embmatrix = nn.Embedding(ntoken, ninp)
        # sharing between embedding and output weight
        self.sharing = sharing
        # dropout
        self.dropinp = nn.Dropout(dropinp)
        self.drophid = nn.Dropout(drophid)
        # Linear function 1 a=Wx+b input->hidden:
        self.lf1 = nn.Linear(ninp * context, nhid)
        # Non-linearity h=tanh(a)
        self.tanh = nn.Tanh()
        # Linear function 2 hidden to output b=Vh+c:
        self.lf2 = nn.Linear(nhid, ntoken)
        if sharing:
            self.lf3 = nn.Linear(ninp * context, ntoken)

    def forward(self, input):
        # concat ngram embeddings
        x_ = self.embmatrix(input).view(-1, self.contextlength * self.embdim)
        x_ = self.dropinp(x_)
        # Linear function 1
        self.max_norm_(self.lf1.weight)
        a = self.lf1(x_)
        a = self.drophid(a)
        # Non-linearity 1
        h = self.tanh(a)

        # Linear function 2
        self.max_norm_(self.lf2.weight)
        b = self.lf2(h)
        b = self.drophid(b)

        # if sharing is on, add mapping between embMatrix and output weight
        if self.sharing:
            self.max_norm_(self.lf3.weight)
            Wx = self.drophid(self.lf3(x_))
            b = Wx + b
        # y=softmax(b)
        return F.log_softmax(b, dim=1)

    def max_norm_(self, w):
        with torch.no_grad():
            norm = w.norm(2, dim=0, keepdim=True).clamp(min=2)
            desired = torch.clamp(norm, max=3)
            w *= (desired / norm)