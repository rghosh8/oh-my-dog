import tensorflow as tf

image_width = 64
image_height = 64
image_channels = 3
#weight initialization for the generator network
weight_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.2)