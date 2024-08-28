import torch
import math
import torch.nn.init
import torch.nn.functional as F
from torchvision.models import vgg19
from torch.nn import Parameter
from torch.nn import init
from torch import nn
import torchvision.models as models
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class GraphConvolution(nn.Module):
    def __init__(self, in_features=4096, out_features=1024, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features)).float()
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        if self.bias is not None:
            return support + self.bias
        else:
            return support

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.query_dim = input_dim // (num_heads )
        self.output_dim = output_dim

        self.query_linear = nn.Linear(input_dim, self.query_dim * self.num_heads)
        self.key_linear = nn.Linear(input_dim, self.query_dim * self.num_heads)
        self.value_linear = nn.Linear(input_dim, self.query_dim * self.num_heads)

        self.output_linear = nn.Linear(self.query_dim * self.num_heads, output_dim)

    def forward(self, x):
        batch_size, input_dim = x.size()
        x = x.view(batch_size, 1, input_dim)

        query = self.query_linear(x)
        key = self.key_linear(x)
        value = self.value_linear(x)

        query = query.view(batch_size, self.num_heads, self.query_dim)
        key = key.view(batch_size, self.num_heads, self.query_dim)
        value = value.view(batch_size, self.num_heads, self.query_dim)

        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.query_dim).float())
        attention_weights = torch.softmax(scores, dim=-1)

        attended_values = torch.matmul(attention_weights, value)

        attended_values = attended_values.view(batch_size, -1)

        output = self.output_linear(attended_values)
        return output

class Fusion(nn.Module):
    def __init__(self, out_features=1024):
        super(Fusion, self).__init__()
        self.f_size = out_features
        self.gate0 = nn.Linear(self.f_size, self.f_size)
        self.gate1 = nn.Linear(self.f_size, self.f_size)

        self.fusion0 = nn.Linear(self.f_size, self.f_size)
        self.fusion1 = nn.Linear(self.f_size, self.f_size)

    def forward(self, vec1, vec2):
        features_1 = self.gate0(vec1)
        features_2 = self.gate1(vec2)
        t = torch.sigmoid(self.fusion0(features_1) + self.fusion1(features_2))
        f = t * features_1 + (1 - t) * features_2
        return f

class Imgencode(nn.Module):
    def __init__(self, img_input_dim=4096, img_output_dim=2048):
        super(Imgencode, self).__init__()
        self.imgLayer = GraphConvolution(img_input_dim, img_output_dim)
        self.imgdec = MultiHeadAttention(img_input_dim, img_output_dim)
        self.imgfusiopn = Fusion(img_output_dim)

    def forward(self, img):
        imgH = self.imgLayer(img)
        img_H = self.imgdec(img)
        img_con = self.imgfusiopn(imgH, img_H)
        return img_con

class Textencode(nn.Module):
    def __init__(self, text_input_dim=1024, text_output_dim=2048, minus_one_dim=1024):
        super(Textencode, self).__init__()
        self.textLayer = GraphConvolution(text_input_dim, text_output_dim)
        self.textdec = MultiHeadAttention(text_input_dim, text_output_dim)
        self.textfusiopn = Fusion(text_output_dim)
    def forward(self, text):
        textH = self.textLayer(text)
        text_H = self.textdec(text)
        text_con = self.textfusiopn(textH, text_H)
        return text_con


class Imgencode_single(nn.Module):
    def __init__(self, img_output_dim=2048, minus_one_dim=1024):
        super(Imgencode_single, self).__init__()
        self.imgencode_single = GraphConvolution(img_output_dim, minus_one_dim)

    def forward(self, img):
        imgcom = self.imgencode_single(img)
        return imgcom

class Imgencode_com(nn.Module):
    def __init__(self, img_output_dim=2048, minus_one_dim=1024):
        super(Imgencode_com, self).__init__()
        self.imgencode_com = GraphConvolution(img_output_dim, minus_one_dim)

    def forward(self, img):
        imgcom = self.imgencode_com(img)
        return imgcom

class Textencode_single(nn.Module):
    def __init__(self, text_output_dim=2048, minus_one_dim=1024):
        super(Textencode_single, self).__init__()
        self.textcode_single = GraphConvolution(text_output_dim, minus_one_dim)

    def forward(self, text):
        textcon = self.textcode_single(text)

        return textcon

class Textencode_com(nn.Module):
    def __init__(self, text_output_dim=2048, minus_one_dim=1024):
        super(Textencode_com, self).__init__()
        self.textencode_com = GraphConvolution(text_output_dim, minus_one_dim)

    def forward(self, text):
        textcon = self.textencode_com(text)

        return textcon

class CrossGAT(nn.Module):

    def __init__(self, batch_size, img_input_dim=4096, img_output_dim=2048,
                 text_input_dim=1024, text_output_dim=2048, minus_one_dim=1024, output_dim=10, adj=None):
        super(CrossGAT, self).__init__()
        self.batch_size = batch_size
        self.W_A = nn.Parameter(torch.eye((self.batch_size), requires_grad=True))
        self.W_B = nn.Parameter(torch.eye((self.batch_size), requires_grad=True))
        self.W_C = nn.Parameter(torch.eye((self.batch_size), requires_grad=True))

        self.imgencode = Imgencode(img_input_dim, img_output_dim)
        self.txtencode = Textencode(text_input_dim, text_output_dim)

        self.img_single = Imgencode_single(img_output_dim, minus_one_dim)
        self.img_com = Imgencode_single(img_output_dim, minus_one_dim)
        self.text_single = Textencode_single(text_output_dim, minus_one_dim)
        self.text_com = Textencode_com(text_output_dim, minus_one_dim)
        self.comfusion = Fusion(minus_one_dim)
        self.shareClassifier = nn.Linear(minus_one_dim, output_dim)
        self.adjfusion = Fusion(batch_size)

    def forward(self, img, text):
        self.batch_size = img.shape[0]
        imgfeature = self.imgencode(img)
        txtfeature = self.txtencode(text)


        imgs1 = self.img_single(imgfeature)
        imgc1 = self.img_com(imgfeature)

        texts1 = self.text_single(txtfeature)
        textc1 = self.text_com(txtfeature)

        adj_A = self.W_A
        adj_B = self.W_B
        adj_C = self.W_C

        imgs2 = torch.matmul(adj_A, imgs1)
        texts2 = torch.matmul(adj_B, texts1)
        imgc2 = torch.matmul(adj_C, imgc1)
        textc2 = torch.matmul(adj_C, textc1)

        img_all = self.comfusion(imgs2, imgc2)
        text_all = self.comfusion(texts2, textc2)
        img_predict = self.shareClassifier(img_all)
        text_predict = self.shareClassifier(text_all)

        return imgs1, imgs2, texts1, texts2, imgc1, imgc2, textc1, textc2, img_all, text_all, img_predict, text_predict, adj_A, adj_B, adj_C



class GeneratorV(nn.Module):
    def __init__(self):
        super(GeneratorV, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(4096, 2000, normalize=False),
            *block(2000, 800, normalize=False),
            nn.Linear(800, 1024),
            nn.Tanh()
        )
    def forward(self, z):
        img = self.model(z)
        return img


class DiscriminatorV(nn.Module):
    def __init__(self):
        super(DiscriminatorV, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    def forward(self, img):
        validity = self.model(img)
        return validity


class GeneratorT(nn.Module):
    def __init__(self):
        super(GeneratorT, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            *block(300, 2000, normalize=False),
            *block(2000, 1000, normalize=False),
            *block(1000, 500, normalize=False),
            nn.Linear(500, 1024),
            nn.Tanh()
        )
    def forward(self, z):
        txt = self.model(z)
        return txt


class DiscriminatorT(nn.Module):
    def __init__(self):
        super(DiscriminatorT, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    def forward(self, txt):
        validity = self.model(txt)
        return validity