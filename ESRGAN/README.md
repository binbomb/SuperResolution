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
![here](https://github.com/SerialLain3170/Image-Enhancement/blob/master/Image/comparison.png)

- RRDBは15層。因みに論文中では23層
- バッチサイズは8
- 最適化手法はAdam(α=0.0002,β1=0.5)
- Adversarial lossの重みは0.005, L1 lossの重みは0.01
- 初めにL1 lossで事前学習をしている
- 安定はしない、60000iterationくらいで崩壊する
