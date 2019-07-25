import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Chain, initializers


def pixel_shuffler(out_ch, x, r=2):
    b, c, w, h = x.shape
    x = F.reshape(x, (b, r, r, int(out_ch/(r*2)), w, h))
    x = F.transpose(x, (0, 3, 4, 1, 5, 2))
    out_map = F.reshape(x, (b, int(out_ch/(r*2)), w*r, h*r))

    return out_map


class CBR(Chain):
    def __init__(self, in_ch, out_ch, up=False, use_bn=False):
        self.use_bn = use_bn
        self.up = up
        w = initializers.GlorotUniform()
        super(CBR, self).__init__()

        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch * 4, 3, 1, 1, initialW=w)
            self.bn0 = L.BatchNormalization(out_ch)

    def __call__(self, x):
        h = self.c0(x)
        h = pixel_shuffler(256, h)
        if self.use_bn:
            h = self.bn0(h)
        h = F.relu(h)

        return h


class SpatialAttention(Chain):
    def __init__(self, in_ch, out_ch):
        w = initializers.GlorotUniform()
        super(SpatialAttention, self).__init__()

        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch, 1, 1, 0, initialW=w)
            self.c1 = L.Convolution2D(out_ch, out_ch, 1, 1, 0, initialW=w)

    def __call__(self, x):
        h = F.relu(self.c0(x))
        h = F.sigmoid(self.c1(h))

        return h


class ChannelAttention(Chain):
    def __init__(self, in_ch, out_ch, k=16):
        w = initializers.GlorotUniform()
        super(ChannelAttention, self).__init__()

        with self.init_scope():
            self.l0 = L.Linear(in_ch, int(in_ch/16))
            self.l1 = L.Linear(int(in_ch/16), in_ch)

    def __call__(self, x):
        batch, ch, height, width = x.shape
        h = F.reshape(F.average_pooling_2d(x, (height, width)), (batch, ch))
        h = F.relu(self.l0(h))
        h = self.l1(h)
        h = F.transpose(F.broadcast_to(h, (height, width, batch, ch)), (2, 3, 0, 1))

        return h


class ResidualAttentionModule(Chain):
    def __init__(self, in_ch, out_ch):
        w = initializers.GlorotUniform()
        super(ResidualAttentionModule, self).__init__()

        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch, 3, 1, 1, initialW=w)
            self.c1 = L.Convolution2D(out_ch, out_ch, 3, 1, 1, initialW=w)
            self.ca = ChannelAttention(out_ch, out_ch)
            self.sa = SpatialAttention(out_ch, out_ch)

    def __call__(self, x):
        h = F.relu(self.c0(x))
        h = self.c1(h)
        sa = self.sa(h)
        ca = self.ca(h)
        hs = F.sigmoid(sa + ca)

        return h * hs


class Model(Chain):
    def __init__(self, base=64, layer=16):
        w = initializers.GlorotUniform()
        super(Model, self).__init__()

        ram_list = chainer.ChainList()
        for num in range(layer):
            ram_list.add_link(ResidualAttentionModule(base, base))

        with self.init_scope():
            self.c0 = L.Convolution2D(3, base, 3, 1, 1, initialW=w)
            self.ram = ram_list
            self.cmiddle = L.Convolution2D(base, base, 3, 1, 1, initialW=w)
            self.up0 = CBR(base, base, up=True)
            self.up1 = CBR(base, base, up=True)
            self.cout = L.Convolution2D(base, 3, 3, 1, 1, initialW=w)

    def __call__(self, x):
        hinit = F.relu(self.c0(x))
        for index, ram in enumerate(self.ram.children()):
            if index == 0:
                h = ram(hinit)
            else:
                h = ram(h)
        h = self.cmiddle(h)
        h = h + hinit
        h = self.up0(h)
        h = self.up1(h)
        h = self.cout(h)

        return F.tanh(h)


class VGG(Chain):
    def __init__(self, last_only=False):
        super(VGG, self).__init__()
        self.last_only = last_only
        with self.init_scope():
            self.base = L.VGG19Layers()

    def __call__(self, x, last_only=False):
        h1 = self.base(x, layers=["conv1_2"])["conv1_2"]
        h2 = self.base(x, layers=["conv2_2"])["conv2_2"]
        h3 = self.base(x, layers=["conv3_4"])["conv3_4"]
        h4 = self.base(x, layers=["conv4_4"])["conv4_4"]
        h5 = self.base(x, layers=["conv5_4"])["conv5_4"]

        return [h1, h2, h3, h4, h5]