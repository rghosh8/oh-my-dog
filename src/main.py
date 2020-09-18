import tensorflow as tf

print('=============loading external modules now=========')
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
print(f'tensorflow version: {tf.__version__}')
from tensorflow.keras import layers
import os, glob, math, random, time, datetime, PIL, imageio
from collections import defaultdict
from tqdm import tqdm, tqdm_notebook
from PIL import Image
from IPython import display
import xml.etree.ElementTree as ET 
print('\n')
print('=============loading user-defined modules now=========')
print('\n')
start_=time.time()
start=time.time()
from src.optimizers import *
print(f'\t\toptimizers took {time.time()-start} s in loading')
print('======================')
print('\n')
start=time.time()
from src.losses import *
print(f'\t\tlosses took {time.time()-start} s in loading')
print('======================')
print('\n')
start=time.time()
from src.g_model import *
print(f'\t\tgenerator model took {time.time()-start} s in loading')
print('======================')
print('\n')
start=time.time()
from src.d_model import *
print(f'\t\tdiscriminator model took {time.time()-start} s in loading')
print('======================')
print('\n')
start=time.time()
from src.static import *
print(f'\t\tstatic took {time.time()-start} s in loading')
print('======================')
print('\n')
start=time.time()
from src.onboarding import *
print(f'\t\tonboarding took {time.time()-start} s in loading')
print('======================')
from src.train_step import *
from src.generate_and_save_images import *

print(f'\tall user-defined modules took {time.time()-start_} s in loading')

# @tf.function
def train_step(images):
	noise = tf.random.normal([BATCH_SIZE, noise_dim])

	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    	generated_images = generator(noise, training=True)

      	real_output, fake_output = discriminator(images, training=True), discriminator(generated_images, training=True)
      	gen_loss, disc_loss = generator_loss(fake_output), discriminator_loss(real_output, fake_output)

	gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
	gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

	generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
	discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
	for epoch in range(epochs):
    	start = time.time()

    	for image_batch in dataset:
      		train_step(image_batch)

		# Produce images for the GIF as we go
		display.clear_output(wait=True)
		generate_and_save_images(generator, epoch + 1, seed)

		# Save the model every 15 epochs
		if (epoch + 1) % 15 == 0:
			checkpoint.save(file_prefix = checkpoint_prefix)

    	print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

	# Generate after the final epoch
	display.clear_output(wait=True)
	generate_and_save_images(generator, epochs, seed)

def generate_and_save_images(model, epoch, test_input):
	predictions = model(test_input, training=False)

	fig = plt.figure(figsize=(8,8))

	for i in range(predictions.shape[0]):
		plt.subplot(4, 4, i+1)
		plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5)
		plt.axis('off')

  	plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  	# plt.show()

if __name__=='__main__':
	images_inputs, normalized_image_vectors, breeds_names = onboarding(all_images, all_breeds)
	generator = g_model()
	discriminator = d_model()

	# This method returns a helper function to compute cross entropy loss
	cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
	checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
	checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
									discriminator_optimizer=discriminator_optimizer,
									generator=generator,
									discriminator=discriminator)
	print('completed data onboarding')
	BUFFER_SIZE = 20000
	BATCH_SIZE = 32
	imagesIn = tf.cast(normalized_image_vectors, 'float32')
	ds = tf.data.Dataset.from_tensor_slices(imagesIn).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
	print('starting training')
	start=time.time()
	train_dataset=ds
	EPOCHS = 1200
	train(train_dataset, EPOCHS)
	print('finished training')
	print(f'it tool {time.time()-start} in training')