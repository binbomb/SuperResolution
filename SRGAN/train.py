import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Chain, optimizers, initializers, serializers
import os
import argparse
import pylab
from preparing import prepare_dataset
from model import Generator, Discriminator, VGG

xp = cuda.cupy
cuda.get_device(0).use()

def set_optimizer(model, alpha=0.0002, beta=0.5):
    optimizer = optimizers.Adam(alpha = alpha, beta1 = beta)
    optimizer.setup(model)

    return optimizer

def calc_loss(feat1, feat2):
    w,h = feat1.shape[2], feat1.shape[3]
    loss = (1/(w*h)) * F.mean_squared_error(feat1, feat2)

    return loss

parser = argparse.ArgumentParser(description="SRGAN")
parser.add_argument("--epoch", default = 1000, type = int, help ="the number of epochs")
parser.add_argument("--batchsize", default = 16, type =int, help = "batchsize")
parser.add_argument("--lam1", default = 0.001, type = float, help = "the weight of vgg loss")
parser.add_argument("--interval", default = 1, type = int, help = "the interval of snapshot")
parser.add_argument("--testsize", default = 2, type = int, help = "testsize")
parser.add_argument("--Ntrain", default = 18000, type = int, help  = "the numbef of train images")

args = parser.parse_args()
epochs = args.epoch
batchsize = args.batchsize
lambda1 = args.lam1
interval = args.interval
testsize = args.testsize
Ntrain = args.Ntrain

image_path = "/usr/MachineLearning/Dataset/trim/train_trim/"
image_list = os.listdir(image_path)

outdir = "./output_per"
if not os.path.exists(outdir):
    os.mkdir(outdir)

test_box = []
for i in range(testsize):
    rnd = np.random.randint(Ntrain + 1, Ntrain + 100)
    image_name = image_path + image_list[rnd]
    _, sr = prepare_dataset(image_name)
    test_box.append(sr)

x_test = chainer.as_variable(xp.array(test_box).astype(xp.float32))

generator = Generator()
generator.to_gpu()
gen_opt = set_optimizer(generator)

discriminator = Discriminator()
discriminator.to_gpu()
dis_opt = set_optimizer(discriminator)

vgg = VGG()
vgg.to_gpu()
vgg_opt = set_optimizer(vgg)
vgg.base.disable_update()

for epoch in range(epochs):
    sum_gen_loss = 0
    sum_dis_loss = 0
    for batch in range(0, Ntrain, batchsize):
        hr_box = []
        sr_box = []
        for i in range(batchsize):
            rnd = np.random.randint(Ntrain)
            image_name = image_path + image_list[rnd]
            hr, sr = prepare_dataset(image_name)
            hr_box.append(hr)
            sr_box.append(sr)
        x = xp.array(sr_box).astype(xp.float32)
        t = xp.array(hr_box).astype(xp.float32)

        x = chainer.as_variable(cuda.to_gpu(x))
        t = chainer.as_variable(cuda.to_gpu(t))

        x_hr = generator(x)
        y_hr = discriminator(x_hr)
        t_hr = discriminator(t)

        x_hr.unchain_backward()

        loss_dis = lambda1 * (F.sum(F.softplus(-y_hr)) / batchsize + F.sum(F.softplus(t_hr)) / batchsize)

        discriminator.cleargrads()
        loss_dis.backward()
        dis_opt.update()
        loss_dis.unchain_backward()

        x_hr = generator(x)
        y_hr = discriminator(x_hr)

        hr_feat1,hr_feat2,hr_feat3 = vgg(t)
        hr_feat1.unchain_backward()
        hr_feat2.unchain_backward()
        hr_feat3.unchain_backward()
        fake_feat1, fake_feat2, fake_feat3 = vgg(x_hr)

        vgg_loss = calc_loss(hr_feat1, fake_feat1)
        vgg_loss += calc_loss(hr_feat2, fake_feat2)
        vgg_loss += calc_loss(hr_feat3, fake_feat3)
        gen_loss = lambda1 * F.sum(F.softplus(y_hr)) / batchsize
        gen_loss += vgg_loss

        #gen_loss = F.mean_squared_error(x_hr, t)

        generator.cleargrads()
        vgg.cleargrads()
        gen_loss.backward()
        gen_opt.update()
        vgg_opt.update()
        gen_loss.unchain_backward()

        sum_dis_loss += loss_dis
        sum_gen_loss += gen_loss

        if epoch % interval == 0 and batch == 0:
            serializers.save_npz("generator_per.model", generator)
            with chainer.using_config("train", False):
                y = generator(x_test)
            y = y.data.get()
            sr = x_test.data.get()
            for i_ in range(testsize):
                tmp = (np.clip((sr[i_,:,:,:])*127.5 + 127.5, 0, 255)).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(testsize,2,2*i_+1)
                pylab.imshow(tmp)
                pylab.axis('off')
                pylab.savefig('%s/visualize_%d.png'%(outdir, epoch))
                tmp = (np.clip((y[i_,:,:,:])*127.5 + 127.5, 0, 255)).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(testsize,2,2*i_+2)
                pylab.imshow(tmp)
                pylab.axis('off')
                pylab.savefig('%s/visualize_%d.png'%(outdir, epoch))

    print("epoch : {}".format(epoch))
    print("Discriminator loss : {}".format(sum_dis_loss / Ntrain))
    print("Generator loss : {}".format(sum_gen_loss / Ntrain))