import tensorflow as tf  
from static import *

def g_mdoel():
    model = tf.keras.Sequential([
        #layter 1
        tf.keras.layers.Dense(8*8*512, use_bias=False, input_shape=(100,0)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Reshape((8,8,512)),

        #layer 2
        tf.keras.layers.Conv2DTranspose(256, (5,5), strides=(2,2), padding='same', \
                                        use_bias=False,\
                                        kernel_initializer=weight_init),

        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.3),

        #layer 3
        tf.keras.layers.Conv2DTranspose(128, (5,5), strides=(2,2), padding='same',\
                                        use_bias=False,\
                                        kernel_initializer=weight_init),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same',\
                                         use_bias=False,\
                                         kernel_initializer=weight_init),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Dense(3,activation='tanh', use_bias=False,\
                                kernel_initializer=weight_init)
    ])

    return model

if __name__=='__main__':
    g_model()