# ESRGAN

## Summary
![net](https://github.com/SerialLain3170/Image-Enhancement/blob/master/ESRGAN/esrgan_net.png)

- SRGANに対してBatchNormalizationを取り払って更に複雑にしたResidual in Residual Dense Block(RRDB)を導入
- DiscriminatorにはRelativistic Discriminatorを導入

## Usage
予め`image_path`に256✕256の画像を格納する。その後
```
$ python train.py
```

## Result
私の環境で生成した画像を以下に示す。
![here](https://github.com/SerialLain3170/Image-Enhancement/blob/master/Image/visualize_27.png)

- 論文中ではRelativistic Discriminatorを導入しているが、色合いが安定しなかったため今回は用いていない。
- RRDBは3層のみ。因みに論文中では23層
- バッチサイズは2
- 最適化手法はAdam(α=0.0002,β1=0.5)
- Adversarial lossの重みは0.005, L1 lossの重みは0.01
- 因みに学習は安定していない(悲哀)、60000 iterationくらいで崩壊し始めるのでearly stoppingが必要
