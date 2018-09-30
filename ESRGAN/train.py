import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda,Chain,optimizers,serializers
import numpy as np
import os
import argparse
import pylab
from model import Generator

xp=cuda.cupy
cuda.get_device(0).use()

def set_optimizer(mode,alpha,beta):
    optimizer = optimizers.Adam(alpha=alpha,beta1=beta)
    optimizer.setup(model)

    return optimizer

def calc_vgg_loss(feat1, feat2):
    _,_,h,w=feat1.shape

    return F.mean_squared_error(feat1,feat2)/(h*w)

parser=argparse.ArgumentParser(description="ESRGAN")
parser.add_argument("--epoch",default-1000,type=int,help="the number of epochs")
parser.add_argument("--batchsize",default=16,type=int,help="batchsize")
parser.add_argument("--testsize",default=2,type=int,help="testsize")
parser.add_argument("--aw",default=0.005,type=float,help="weight of adversarial loss")
parser.add_argument("--l1",default=0.01,type=float,help="weight of l1 loss")
parser.add_argument("--interval",default=1,type=int,help="interval of snapshot")
args = parser.parse_args()
epochs=args.epoch
batchsize=args.batchsize
testsize=args.testsize
adver_weight=args.aw
l1_weight=args.l1
interval=args.interval

outdir="./output"
if not os.path.exists(outdir):
    os.mkdir(outdir)

generator=Generator()
generator.to_gpu()
gen_opt=set_optimizer(generator)

discriminator=Discriminator()
discriminator.to_gpu()
dis_opt=set_optimizer(discriminator)

vgg=VGG()
vgg.to_gpu()
vgg_opt=set_optimizer(vgg)

for epoch in range(epochs):
    sum_gen_loss=0
    sum_dis_loss=0
    for batch in range(0,Ntrain,batchsize):
        hr_box=[]
        sr_box=[]
        for index in range(batchsize):


        x=chainer.as_variable(xp.array(hr_box).astype(xp.float32))
        t=chainer.as_variable(xp.array(sr_box).astype(xp.float32))
        
        y=generator(x)
        y_dis=discriminator(y)
        t_dis=discriminator(t)
        y.unchain_backward()

        fake_mean=F.broadcast_to(F.mean(y_dis),(batchsize,1))
        real_mean=F.broadcast_to(F.mean(t_dis),(batchsize,1))
        dis_loss=F.mean(F.softplus(-(t_dis-fake_mean)))
        dis_loss+=F.mean(F.softplus(y_dis-real_mean))
        dis_loss /= 2

        discriminator.cleargrads()
        dis_loss.backward()
        dis_opt.udpate()
        dis_loss.unchain_backward()

        y=generator(x)
        y_dis=discriminator(y)
        t_dis=discriminator(t)

        fake_mean=F.broadcast_to(F.mean(y_dis),(batchsize,1))
        real_mean=F.broadcast_to(F.mean(t_dis),(batchsize,1))
        adver_loss=F.mean(F.softplus(-(y_dis-real_mean)))
        adver_loss+=F.mean(F.softplus(t_dis-fake_mean))
        adver_loss /= 2

        fake_feat=vgg(y)
        real_feat=vgg(t)
        vgg_loss=calc_vgg_loss(fake_feat,real_feat)

        l1_loss=F.mean_absolute_eror(y,t)

        gen_loss=vgg_loss+adver_weight*adver_loss+l1_weight*l1_loss

        generator.cleargrads()
        gen_loss.backward()
        gen_opt.update()
        gen_loss.unchain_backward()

        sum_dis_loss+=dis_loss.data.get()
        sum_gen_loss+=gen_loss.data.get()

        if epoch%interval==0 and batch==0:
            serializers.save_npz("generator.model",generator)

    print("epoch:{}".format(epoch))
    print("Discrimintor loss:{}".format(sum_dis_loss))
    print("Generator loss:{}".format(sum_gen_loss))


