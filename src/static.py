import tensorflow as tf
import os
#loading data
root = '/home/ubuntu/oh-my-dog/data/'
image_path = root + 'all-dogs/'
breed_path = root + 'Annotation/'
all_images = os.listdir(image_path)
all_breeds = os.listdir(breed_path)
#setting image 
image_width = 64
image_height = 64
image_channels = 3
#data visualization: 25 random indices for dog visualization
rand_idx_viz = [10, 230, 490, 900, 1200, 1500, 2500, 2900, 3400, \
                3900, 4300, 4750, 5200, 5298, 6450, 6689, 7022, 7450, 7900, 8250, \
                8500, 8750, 9000, 11150, 15000]
#weight initialization for the generator network
weight_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.2)
# generative network
noise_dim = 100
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, noise_dim])
#checkpoints
checkpoint_dir = './training-checkpoints-2'