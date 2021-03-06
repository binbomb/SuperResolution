# ESRGAN

## Summary
![net](https://github.com/SerialLain3170/Image-Enhancement/blob/master/ESRGAN/esrgan_net.png)

- The authors of ESRGAN exclude Batch Normalization from network architecture in SRGAN and use Residual in Residual Dense Block(RRDB) which is more complicate than Resudual blocks in SRGAN.
- They also replace conventional discriminator with Relativistic discriminator.

## Usage
Execute the command line below after containing 256×256 images into `image_path`.
```
$ python train.py
```

## Result
Result generated by my development environment is below.
![here](https://github.com/SerialLain3170/Image-Enhancement/blob/master/Image/comparison.png)

- The number of RRDB blocks is 15 instead of 23 in paper.
- Batch size: 8
- Using Adam as optimizer
- The weight of adversarial loss is 0.005 and the weight of l1 loss is 0.01
- Pre-training with l1 loss
- We have found that this implementation is instable and collapses after 60000 iterations
