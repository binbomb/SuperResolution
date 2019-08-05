import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Chain, initializers

xp = cuda.cupy
cuda.get_device(0).use()


class ResBlock_single(Chain):
    def __init__(self, in_ch, out_ch):
        w = initializers.GlorotUniform()
        super(ResBlock_single, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch, 3, 1, 1, initialW=w)

    def __call__(self, x):
        h = F.leaky_relu(self.c0(x))

        return h


class Dense_Block(Chain):
    def __init__(self, in_ch, out_ch):
        w = initializers.GlorotUniform()
        super(Dense_Block, self).__init__()
        with self.init_scope():
            self.res0 = ResBlock_single(in_ch, out_ch)
            self.res1 = ResBlock_single(out_ch, out_ch)
            self.res2 = ResBlock_single(out_ch, out_ch)
            self.res3 = ResBlock_single(out_ch, out_ch)
            self.c0 = L.Convolution2D(out_ch, out_ch, 3, 1, 1, initialW=w)

    def __call__(self, x):
        h1 = self.res0(x)
        h2 = self.res1(h1 + x)
        h3 = self.res2(h2 + h1 + x)
        h4 = self.res3(h3 + h2 + h1 + x)
        h = self.c0(h4 + h2 + h1 + x)

        return h


class RRDB(Chain):
    def __init__(self, in_ch, out_ch):
        super(RRDB, self).__init__()
        with self.init_scope():
            self.dense0 = Dense_Block(in_ch, out_ch)
            self.dense1 = Dense_Block(out_ch, out_ch)
            self.dense2 = Dense_Block(out_ch, out_ch)

    def __call__(self, x, beta=0.2):
        h1 = self.dense0(x)
        x1 = beta*h1 + x
        h2 = self.dense1(x1)
        x2 = beta*h2 + x1
        h3 = self.dense2(x2)
        x3 = beta*h3 + x2

        return x + beta * x3


def pixel_shuffler(out_ch, x, r=2):
    b, c, w, h = x.shape
    x = F.reshape(x, (b, r, r, int(out_ch/(r*2)), w, h))
    x = F.transpose(x, (0, 3, 4, 1, 5, 2))
    out_map = F.reshape(x, (b, int(out_ch/(r*2)), w*r, h*r))

    return out_map


class Generator(Chain):
    def __init__(self, base=64, layers=15):
        super(Generator, self).__init__()
        w = initializers.GlorotUniform()
        rrdb = chainer.ChainList()
        for num in range(layers):
            rrdb.add_link(RRDB(base, base))
        self.base = base

        with self.init_scope():
            self.c0 = L.Convolution2D(3, base, 3, 1, 1, initialW=w)
            self.rrdbs = rrdb
            self.lff1 = L.Convolution2D(base, base, 3, 1, 1, initialW=w)
            self.c1 = L.Convolution2D(base, base*2, 3, 1, 1, initialW=w)
            self.c2 = L.Convolution2D(base*2, base*4, 3, 1, 1, initialW=w)
            self.c3 = L.Convolution2D(base, base*2, 3, 1, 1, initialW=w)
            self.c4 = L.Convolution2D(base*2, base*4, 3, 1, 1, initialW=w)
            self.c5 = L.Convolution2D(base, base, 3, 1, 1, initialW=w)
            self.c6 = L.Convolution2D(base, 3, 3, 1, 1, initialW=w)

            self.bn0 = L.BatchNormalization(base)
            self.bn1 = L.BatchNormalization(base)
            self.bn2 = L.BatchNormalization(base)
            self.bn3 = L.BatchNormalization(base)

    def __call__(self, x):
        h = F.relu(self.bn0(self.c0(x)))
        for index, rrdb in enumerate(self.rrdbs.children()):
            h = rrdb(h)
        h = self.lff1(h)
        h = self.c2(self.c1(h))
        h = pixel_shuffler(self.base*4, h)
        h = F.relu(self.bn1(h))
        h = self.c4(self.c3(h))
        h = pixel_shuffler(self.base*4, h)
        h = F.relu(self.bn2(h))
        h = F.relu(self.bn3(self.c5(h)))
        h = self.c6(h)

        return h


class CBR_dis(Chain):
    def __init__(self, in_ch, out_ch, bn=True, activation=F.leaky_relu, down=None):
        super(CBR_dis, self).__init__()
        self.activation = activation
        self.bn = bn
        self.out_ch = out_ch
        self.down = down
        w = chainer.initializers.GlorotUniform()

        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch, 3, 1, 1, initialW=w)
            self.cdown = L.Convolution2D(in_ch, out_ch, 4, 2, 1)
            self.bn0 = L.BatchNormalization(int(out_ch))

    def __call__(self, x):
        if self.down:
            h = self.cdown(x)
        else:
            h = self.c0(x)

        if self.bn:
            h = self.bn0(h)

        if self.activation is not None:
            h = self.activation(h)

        return h


class Dis_ResBlock(Chain):
    def __init__(self, in_ch, out_ch):
        super(Dis_ResBlock, self).__init__()
        w = chainer.initializers.GlorotUniform()
        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch, 3, 1, 1, initialW=w)
            self.c1 = L.Convolution2D(out_ch, out_ch, 3, 1, 1, initialW=w)

            self.bn0 = L.BatchNormalization(out_ch)
            self.bn1 = L.BatchNormalization(out_ch)
    
    def __call__(self, x):
        h = F.leaky_relu(self.bn0(self.c0(x)))
        h = F.leaky_relu(self.bn1(self.c1(h)))
        h = h + x

        return h


class Discriminator(Chain):
    def __init__(self, base=32):
        super(Discriminator, self).__init__()
        w = chainer.initializers.GlorotUniform()
        with self.init_scope():
            self.c0 = L.Convolution2D(3, base, 4, 2, 1, initialW=w)
            self.r0 = CBR_dis(base, base, activation=F.leaky_relu, down=True)
            self.r1 = Dis_ResBlock(base, base)
            self.r2 = Dis_ResBlock(base, base)
            self.r3 = CBR_dis(base, base*2, activation=F.leaky_relu, down=True)
            self.r4 = Dis_ResBlock(base*2, base*2)
            self.r5 = Dis_ResBlock(base*2, base*2)
            self.r6 = CBR_dis(base*2, base*4, activation=F.leaky_relu, down=True)
            self.r7 = Dis_ResBlock(base*4, base*4)
            self.r8 = Dis_ResBlock(base*4, base*4)
            self.r9 = CBR_dis(base*4, base*8, activation=F.leaky_relu, down=True)
            self.r10 = Dis_ResBlock(base*8, base*8)
            self.r11 = Dis_ResBlock(base*8, base*8)
            self.r12 = CBR_dis(base*8, base*16, activation=F.leaky_relu, down=True)
            self.r13 = Dis_ResBlock(base*16, base*16)
            self.r14 = Dis_ResBlock(base*16, base*16)
            self.r15 = CBR_dis(base*16, base*32, activation=F.leaky_relu, down=True)
            self.l0 = L.Linear(None, 1024, initialW=w)
            self.l1 = L.Linear(1024, 1, initialW=w)
    
    def __call__(self, x):
        h = F.leaky_relu(self.c0(x))
        h = self.r0(h)
        h = self.r1(h)
        h = self.r2(h)
        h = self.r3(h)
        h = self.r4(h)
        h = self.r5(h)
        h = self.r6(h)
        h = self.r7(h)
        h = self.r8(h)
        h = self.r9(h)
        h = self.r10(h)
        h = self.r11(h)
        h = self.r12(h)
        h = self.r13(h)
        h = self.r14(h)
        h = self.r15(h)
        h = F.leaky_relu(self.l0(h))
        h = self.l1(h)

        return h


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