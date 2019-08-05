import chainer
import chainer.functions as F
import argparse

from pathlib import Path
from chainer import serializers
from model_esrgan import Generator, Discriminator, VGG
from dataset import DatasetLoader
from utils import set_optimizer
from evaluation import Evaluation


class ESRGANLossFunction:
    def __init__(self):
        pass

    @staticmethod
    def content_loss(y, t):
        return F.mean_absolute_error(y, t)

    @staticmethod
    def perceptual_loss(vgg, y, t):
        y_feat = vgg(y)
        t_feat = vgg(t)
        sum_loss = 0

        for yf, tf in zip(y_feat, t_feat):
            _, ch, h, w = yf.shape
            sum_loss += F.mean_squared_error(yf, tf) / (ch * h * w)

        return sum_loss

    @staticmethod
    def dis_hinge_loss(discrminator, y, t):
        fake = discrminator(y)
        real = discrminator(t)

        return F.mean(F.relu(1. - real)) + F.mean(F.relu(1. + fake))

    @staticmethod
    def gen_hinge_loss(discrminator, y):
        fake = discrminator(y)

        return -F.mean(fake)

    def __str__(self):
        return f"func1: {self.content_loss.__name__}"


def train(epochs, iterations, outdir, path, batchsize, validsize,
          adv_weight, content_weight):
    # Dataset Definition
    dataloader = DatasetLoader(path)
    print(dataloader)
    t_valid, x_valid = dataloader(validsize, mode="valid")

    # Model & Optimizer Definition
    model = Generator()
    model.to_gpu()
    optimizer = set_optimizer(model)
    serializers.load_npz('./outdir_pretrain/model_80.model', model)

    discriminator = Discriminator()
    discriminator.to_gpu()
    dis_opt = set_optimizer(discriminator)

    vgg = VGG()
    vgg.to_gpu()
    vgg_opt = set_optimizer(vgg)
    vgg.base.disable_update()

    # Loss Function Definition
    lossfunc = ESRGANLossFunction()
    print(lossfunc)

    # Evaluation Definition
    evaluator = Evaluation()

    for epoch in range(epochs):
        sum_loss = 0
        for batch in range(0, iterations, batchsize):
            t_train, x_train = dataloader(batchsize, mode="train")

            y_train = model(x_train)
            y_train.unchain_backward()
            loss = adv_weight * lossfunc.dis_hinge_loss(discriminator, y_train, t_train)

            discriminator.cleargrads()
            loss.backward()
            dis_opt.update()
            loss.unchain_backward()

            y_train = model(x_train)
            loss = adv_weight * lossfunc.gen_hinge_loss(discriminator, y_train)
            loss += content_weight * lossfunc.content_loss(y_train, t_train)
            loss += lossfunc.perceptual_loss(vgg, y_train, t_train)

            model.cleargrads()
            vgg.cleargrads()
            loss.backward()
            optimizer.update()
            vgg_opt.update()
            loss.unchain_backward()

            sum_loss += loss.data

            if batch == 0:
                serializers.save_npz(f"{outdir}/model_{epoch}.model", model)

                with chainer.using_config('train', False):
                    y_valid = model(x_valid)
                x = x_valid.data.get()
                y = y_valid.data.get()
                t = t_valid.data.get()

                evaluator(x, y, t, epoch, outdir)

        print(f"epoch: {epoch}")
        print(f"loss: {sum_loss / iterations}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAM")
    parser.add_argument('--e', type=int, default=1000, help="the number of epochs")
    parser.add_argument('--i', type=int, default=10000, help="the number of iterations")
    parser.add_argument('--b', type=int, default=32, help="batch size")
    parser.add_argument('--v', type=int, default=3, help="valid size")
    parser.add_argument('--c', type=float, default=0.01, help="the weight of content loss")
    parser.add_argument('--a', type=float, default=0.005, help="the weight of adversarial loss")

    args = parser.parse_args()

    dataset_path = Path('./Dataset/danbooru-images')
    outdir = Path('./outdir_train')
    outdir.mkdir(exist_ok=True)

    train(args.e, args.i, outdir, dataset_path, args.b, args.v, args.a, args.c)