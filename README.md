# oh-my-dog

### Abstract

This capstone uses generative adversarial network (GAN) to generate realistic dog images from the training data. 

### Objective

* Complete GAN model

How GAN works?

* generateor receives a random seed as input
* the seed is used to produce output
* discriminator then is used to classify real images (drawn from the training set) from fake images (produced by the generator)
* losses are calculated for each of the models
* gradients are used to update the generator and discriminator

### Data

![](./figures/dogs.png) 

https://www.kaggle.com/c/generative-dog-images



### Exploratory Data Analysis

In total, there are 25,000 dogs and 120 breeds.

![](./figures/dog_breed_distribution.png)  

![](./figures/n02085620_7.jpg)

### Network Architecture

Here is the GAN network I used for this capstone. 

![](./figures/my_gan.png)

### Results

GAN is difficult to train. In interest of time, I break the problem into a simpler problem where I focused on grayscale images. Here training results I obtained: 

![](./figures/dcgan-9.gif)

Here is the best dog images I was able to produce during the limited capstone time. 

![](./figures/image_at_epoch_1140.png)


### Conclusion

* GAN is hard to train as it requires an equilibrimum between two competing neural networks. The common problems are mode collapse and non-convergence. 

* Next, I will be exploring BigGAN based transfer learning to improve my dog images.

## References

* https://tfhub.dev/deepmind/biggan-128/2
* https://arxiv.org/abs/1701.00160


