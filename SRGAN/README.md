# SRGAN
## Summary

![net](https://github.com/SerialLain3170/Image-Enhancement/blob/master/SRGAN/srgan_net.png)
- GeneratorにSRResNetを導入。Residual BlockとPixel Shufflerを考慮(コード中では3層のResidual block)。
- 損失関数としてはAdversarial lossとContent loss(VGGの各層出力を利用)

## Usage
予め`image_path`に396✕396の画像を格納しておく。その後以下のコマンドを実行。
```
$ python train.py
```

### Result
私の環境で生成した画像を以下に示す
![srgan](https://github.com/SerialLain3170/Image-Enhancement/blob/master/Image/srgan.png)

比較対象としてBilinear補間とBicubic補間のアップサンプリングも載せている。
- バッチサイズは3
- 最適化手法はAdam(α=0.0002, β1=0.5)
- Adversarial lossの重みは0.001
