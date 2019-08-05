import chainer
import chainer.functions as F
import argparse

from pathlib import Path
from chainer import serializers
from model_esrgan import Generator
from dataset import DatasetLoader
from utils import set_optimizer
from evaluation import Evaluation


class ESRGANPretrainLossFunction:
    def __init__(self):
        pass

    @staticmethod
    def content_loss(y, t):
        return F.mean_absolute_error(y, t)

    def __str__(self):
        return f"func1: {self.content_loss.__name__}"


def train(epochs, iterations, outdir, path, batchsize, validsize):
    # Dataset Definition
    dataloader = DatasetLoader(path)
    print(dataloader)
    t_valid, x_valid = dataloader(validsize, mode="valid")

    # Model & Optimizer Definition
    model = Generator()
    model.to_gpu()
    optimizer = set_optimizer(model)

    # Loss Function Definition
    lossfunc = ESRGANPretrainLossFunction()
    print(lossfunc)

    # Evaluation Definition
    evaluator = Evaluation()

    for epoch in range(epochs):
        sum_loss = 0
        for batch in range(0, iterations, batchsize):
            t_train, x_train = dataloader(batchsize, mode="train")

            y_train = model(x_train)
            loss = lossfunc.content_loss(y_train, t_train)

            model.cleargrads()
            loss.backward()
            optimizer.update()
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
    parser = argparse.ArgumentParser(description="ESRGANPretrain")
    parser.add_argument('--e', type=int, default=1000, help="the number of epochs")
    parser.add_argument('--i', type=int, default=20000, help="the number of iterations")
    parser.add_argument('--b', type=int, default=32, help="batch size")
    parser.add_argument('--v', type=int, default=3, help="valid size")

    args = parser.parse_args()

    dataset_path = Path('./Dataset/danbooru-images')
    outdir = Path('./outdir_pretrain')
    outdir.mkdir(exist_ok=True)

    train(args.e, args.i, outdir, dataset_path, args.b, args.v)