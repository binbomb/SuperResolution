import chainer
import chainer.functions as F
import argparse

from pathlib import Path
from chainer import serializers
from model import Model, VGG
from model_esrgan import Generator
from dataset import DatasetLoader
from utils import set_optimizer
from evaluation import Evaluation


class RAMLossFunction:
    def __init__(self):
        pass

    @staticmethod
    def content_loss(y, t):
        return F.mean_absolute_error(y, t)

    @staticmethod
    def perceptual_loss(y, t):
        sum_loss = 0

        for f1, f2 in zip(y, t):
            _, c, h, w = f1.shape
            sum_loss += F.mean_squared_error(f1, f2) / (c * h * w)

        return sum_loss

    def __str__(self):
        return f"func1: {self.content_loss.__name__}\nfunc2: {self.perceptual_loss.__name__}"


def train(epochs, iterations, outdir, path, batchsize, validsize, model_type):
    # Dataset Definition
    dataloader = DatasetLoader(path)
    print(dataloader)
    t_valid, x_valid = dataloader(validsize, mode="valid")

    # Model & Optimizer Definition
    if model_type == 'ram':
        model = Model()
    elif model_type == 'gan':
        model = Generator()
    model.to_gpu()
    optimizer = set_optimizer(model)

    vgg = VGG()
    vgg.to_gpu()
    vgg_opt = set_optimizer(vgg)
    vgg.base.disable_update()

    # Loss Function Definition
    lossfunc = RAMLossFunction()
    print(lossfunc)

    # Evaluation Definition
    evaluator = Evaluation()

    for epoch in range(epochs):
        sum_loss = 0
        for batch in range(0, iterations, batchsize):
            t_train, x_train = dataloader(batchsize, mode="train")

            y_train = model(x_train)
            y_feat = vgg(y_train)
            t_feat = vgg(t_train)
            loss = lossfunc.content_loss(y_train, t_train)
            loss += lossfunc.perceptual_loss(y_feat, t_feat)

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
    parser.add_argument('--model', type=bool, default='ram', help="the type of model")

    args = parser.parse_args()

    dataset_path = Path('./danbooru-images')
    outdir = Path('./outdir')
    outdir.mkdir(exist_ok=True)

    train(args.e, args.i, outdir, dataset_path, args.b, args.v, args.model)