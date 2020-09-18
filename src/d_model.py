import tensorflow as tf
from static import *
from tensorflow.keras import layers

def d_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (4,4), strides=(2,2), padding='same', input_shape=[image_width,image_height,3],\
                            kernel_initializer=weight_init))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    print(model.output_shape)

    model.add(layers.Conv2D(128, (4,4), strides=(2,2), padding='same',\
                            kernel_initializer=weight_init))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    print(model.output_shape)
      
    model.add(layers.Conv2D(256, (4,4), strides=(2,2), padding='same',\
                            kernel_initializer=weight_init))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    print(model.output_shape)

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    print(model.output_shape)

    return model

