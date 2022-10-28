## Exploring Non-local Property for GAN Generated Fake Face Detection

This is a PyTorch implementation of NAFID: A Non-local Attention based Fake Image
Detection Network.

### Requirements
* PyTorch >= 1.7.1
* Numpy >= 1.21.2
* torchvision >= 0.8.2
* advertorch >= 0.2.3

### How to run
We include recommended configurations for reproducing our results for detecting StyleGAN2 face forgery in `config.py`.
After specifying your own home directory, you can simply run `python main.py` to reproduce our results.