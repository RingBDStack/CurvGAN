import torch
import math
from torch import optim
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import manifolds
from utils.data_utils import read_embeddings
from utils.distributions import WrappedNormal

class FermiDiracDecoder(nn.Module):
    """Fermi Dirac to compute edge probabilities based on distances."""

    def __init__(self, r, t):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t

    def forward(self, dist):
        probs = 1. / (torch.exp((dist - self.r) / self.t) + 1.0)
        return probs


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class LinearClassifier():
    def __init__(self, config):
        self.manifold = getattr(manifolds, config.manifold)()
        self.c = torch.Tensor([1.]) if config.c == None else torch.tensor([config.c])
        self.linear = HypLinear(self.manifold, config.dims * 2, config.num_classes, self.c, 0.2, True)
        self.opt = torch.optim.Adam(self.linear.parameters(), lr=0.01, weight_decay=0.001)

    def train(self, x_train, y_train):
        self.linear.train()
        self.opt.zero_grad()
        output = self.linear(x_train)
        loss = F.cross_entropy(output, y_train)
        loss.backward()
        self.opt.step()

    def test(self, x_test):
        self.linear.eval()
        y_pred = self.linear(x_test).data.max(1, keepdim=False)[1].tolist()
        return y_pred




class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.device = config.device
        self.manifold = getattr(manifolds, config.manifold)()
        # Curvatures initialization
        self.c = nn.Parameter(torch.Tensor([1.])).to(self.device) if config.c == None else torch.tensor([config.c]).to(self.device)
        self.wrapped_nd = WrappedNormal(torch.tensor([0.]).to(self.device), torch.tensor([1.]).to(self.device), manifolds.PoincareBallWrapped(c=self.c))
        self.mlp = HypLinear(self.manifold, config.dims, config.dims, self.c, config.dropout, use_bias=True) 
        # import pre-train embedding if exists
        self.embedding = nn.Parameter(torch.Tensor(config.num_nodes, config.dims))
        if config.pretrain != None:
            init_emb = read_embeddings(config.pretrain, config.num_nodes, config.dims)
            self.embedding.data = init_emb
        else:
            init.uniform_(self.embedding)

    def forward(self, node_ids):
        x = self.embedding.index_select(0, torch.tensor(node_ids).to(self.device))
        noise = self.wrapped_nd.sample(x.size()).squeeze()
        x = self.manifold.mobius_add(x, noise, c=self.c) 
        x = self.mlp(x)
        xt = F.leaky_relu(self.manifold.logmap0(x, c=self.c), negative_slope=0.1)
        xt = self.manifold.proj_tan0(xt, c=self.c)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c), c=self.c)
        # return x



class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.device = config.device
        self.manifold = getattr(manifolds, config.manifold)()
        self.n_iter = config.n_iter
        self.c = nn.Parameter(torch.Tensor([1.])).to(self.device) if config.c == None else torch.tensor([config.c]).to(self.device)
        self.embedding = nn.Parameter(torch.Tensor(config.num_nodes, config.dims))
        if config.pretrain != None:
            init_emb = read_embeddings(config.pretrain, config.num_nodes, config.dims)
            self.embedding.data = init_emb
        else:
            init.uniform_(self.embedding)
        self.decoder = FermiDiracDecoder(config.dc_r, config.dc_t)

    def forward(self, emb1, emb2):
        emb_in = emb1
        if not torch.is_tensor(emb1):
            emb_in = self.embedding.index_select(0, torch.tensor(emb1).to(self.device))
        emb_out = emb2
        if not torch.is_tensor(emb2):
            emb_out = self.embedding.index_select(0, torch.tensor(emb2).to(self.device))
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        scores = self.decoder(sqdist)
        return scores
