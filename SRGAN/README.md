# SRGAN
## Summary

![net](https://github.com/SerialLain3170/Image-Enhancement/blob/master/SRGAN/srgan_net.png)
- The authors of paper use SRResNet as generator, SRResNet consist of Residual blocks in hidden layers and Pixel shuffler in upsampling layer.
- Loss functions are adversarial loss and perceptual loss calculated with the output of each layer in VGG.

## Usage
Execute the command line below after containing 396×396 images into `image_path`.
```
$ python train.py
```

### Result
Image generated in my development environment is below.
![srgan](https://github.com/SerialLain3170/Image-Enhancement/blob/master/Image/srgan.png)

There are images obtained by bilinear interpolation and bicubic interpolation to compare with the output of SRGAN.
- Batch size: 3
- Using Adam as optimizer
- The weight of adversarial loss is 0.001
