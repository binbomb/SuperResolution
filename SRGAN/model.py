import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain, cuda, Variable
import numpy as np

xp = cuda.cupy

def pixel_shuffler(out_ch, x, r = 2):
    b, c, w, h = x.shape
    x = F.reshape(x, (b, r, r, int(out_ch/(r*2)), w, h))
    x = F.transpose(x, (0,3,4,1,5,2))
    out_map = F.reshape(x, (b, int(out_ch/(r*2)), w*r, h*r))

    return out_map

class CBR(Chain):
    def __init__(self, in_ch, out_ch, bn = True, activation = F.relu, down = None):
        super(CBR,self).__init__()
        self.activation = activation
        self.bn = bn
        self.out_ch = out_ch
        self.down = down
        w = chainer.initializers.Normal(0.02)

        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch, 3,1,1, initialW = w)
            self.bn0 = L.BatchNormalization(int(out_ch/4))

    def __call__(self,x):
        h = self.c0(x)
        h = pixel_shuffler(self.out_ch, h)
            
        if self.bn:
            h = self.bn0(h)
            
        if self.activation is not None:
            h = self.activation(h)

        return h

class CBR_dis(Chain):
    def __init__(self, in_ch, out_ch, bn = True, activation = F.relu, down = None):
        super(CBR_dis,self).__init__()
        self.activation = activation
        self.bn = bn
        self.out_ch = out_ch
        self.down = down
        w = chainer.initializers.Normal(0.02)

        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch, 3,1,1, initialW = w)
            self.cdown = L.Convolution2D(in_ch,out_ch, 3,2,1)
            self.bn0 = L.BatchNormalization(int(out_ch))

    def __call__(self,x):
        if self.down:
            h = self.cdown(x)
        else:
            h = self.c0(x)

        if self.bn:
            h = self.bn0(h)

        if self.activation is not None:
            h = self.activation(h)

        return h

class Gen_ResBlock(Chain):
    def __init__(self, in_ch, hid_ch):
        super(Gen_ResBlock, self).__init__()
        w = chainer.initializers.Normal(0.02)
        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, hid_ch,3,1,1, initialW = w)
            self.c1 = L.Convolution2D(hid_ch, hid_ch,3,1,1, initialW = w)

            self.bn0 = L.BatchNormalization(hid_ch)
            self.bn1 = L.BatchNormalization(hid_ch)
    
    def __call__(self,x):
        h = F.relu(self.bn0(self.c0(x)))
        h = self.bn1(self.c1(h))
        
        return h + x

class Generator(Chain):
    def __init__(self, base = 32):
        super().__init__()
        w = chainer.initializers.Normal(0.02)
        with self.init_scope():
            self.l0 = L.Convolution2D(3,base,9,1,4,initialW = w)
            self.r0 = Gen_ResBlock(base, base)
            self.r1 = Gen_ResBlock(base, base)
            self.r2 = Gen_ResBlock(base, base)
            self.c0 = L.Convolution2D(base,base,3,1,1,initialW = w)
            self.cbr0 = CBR(base, base*4)
            self.cbr1 = CBR(base, base*4)
            self.c1 = L.Convolution2D(base, 3, 9, 1, 4, initialW = w)

            self.bn0 = L.BatchNormalization(base)
            self.bn1 = L.BatchNormalization(base)

    def __call__(self,x):
        h1 = F.relu(self.bn0(self.l0(x)))
        h = self.r0(h1)
        h = self.r1(h)
        h = self.r2(h)
        h = h1 + F.relu(self.bn1(self.c0(h)))
        h = self.cbr0(h)
        h = self.cbr1(h)
        h = self.c1(h)

        return h

class Discriminator(Chain):
    def __init__(self,base = 32):
        super(Discriminator, self).__init__()
        w = chainer.initializers.Normal(0.02)
        with self.init_scope():
            self.c0 = L.Convolution2D(3,base,4,2,1,initialW = w)
            self.cbr0 = CBR_dis(base, base, activation=F.leaky_relu, down = True)
            self.cbr1 = CBR_dis(base, base*2, activation=F.leaky_relu)
            self.cbr2 = CBR_dis(base*2, base*2, activation=F.leaky_relu, down = True)
            self.cbr3 = CBR_dis(base*2, base*4, activation=F.leaky_relu)
            self.cbr4 = CBR_dis(base*4, base*4, activation=F.leaky_relu, down = True)
            self.cbr5 = CBR_dis(base*4, base*8, activation=F.leaky_relu)
            self.cbr6 = CBR_dis(base*8, base*8, activation=F.leaky_relu, down = True)
            self.l0 = L.Linear(None, 512, initialW = w)
            self.l1 = L.Linear(512,1,initialW = w)
    
    def __call__(self, x):
        h = F.leaky_relu(self.c0(x))
        h = self.cbr0(h)
        h = self.cbr1(h)
        h = self.cbr2(h)
        h = self.cbr3(h)
        h = self.cbr4(h)
        h = self.cbr5(h)
        h = self.cbr6(h)
        h = F.leaky_relu(self.l0(h))
        h = self.l1(h)

        return h

class VGG(Chain):
    def __init__(self, last_only = False):
        super(VGG, self).__init__()
        self.last_only = last_only
        with self.init_scope():
            self.base = L.VGG16Layers()

    def __call__(self,x, last_only = False):
        h2 = self.base(x, layers=["conv2_1"])["conv2_1"]
        h3 = self.base(x, layers=["conv3_1"])["conv3_1"]
        h4 = self.base(x, layers=["conv4_1"])["conv4_1"]

        return h2,h3,h4