from static import *
import numpy as np

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  r, c = int(np.sqrt(len(test_input))), int(np.sqrt(len(test_input)))

  fig = plt.figure(figsize=(r,c))

  for i in range(predictions.shape[0]):
      plt.subplot(r, c, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5)
      plt.axis('off')

  plt.savefig('../training_results/image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

