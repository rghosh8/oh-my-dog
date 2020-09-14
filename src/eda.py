import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
import glob
import math
import random
import time
import datetime
from collections import defaultdict
from tqdm import tqdm, tqdm_notebook
import xml.etree.ElementTree as ET 
from static import *
from path import *

# import cv2
plt.style.use('ggplot')

class EDA(object):
    def __init__(self, image_input_dir, image_ann_dir):
        self.image_input_dir = image_input_dir
        self.image_ann_dir = image_ann_dir

    @property
    def dog_breed(self):
        dog_breed_dict = {}
        for annotation in os.listdir(self.image_ann_dir):
            try:
                annotations = annotation.split('-')
                dog_breed_dict[annotations[0]] = annotations[1]
            except:
                pass

        return dog_breed_dict

    @property
    def get_input_image_dict(self):
        image_sample_dict = defaultdict(list)
        for image in os.listdir(self.image_input_dir):
            filename = image.split('.')
            label_code = filename[0].split('_')[0]
            breed_name = self.dog_breed[label_code]
            #print('Code: {}, Breed: {}'.format(label_code, breed_name))
            if image is not None:
                image_sample_dict[breed_name].append(image)
        
        print('Created label dictionary for input images.')
        return image_sample_dict


def main():
    a=(EDA(image_input_dir,image_ann_dir).get_input_image_dict)

def plot_class_distributions(image_sample_dict, title='Distribution of Dog Breeds', save_=False):
    class_lengths = []
    labels = []
    total_images = 0
    
    print('Total amount of dog breeds: ', len(image_sample_dict))
    
    for label, _ in image_sample_dict.items():
        total_images += len(image_sample_dict[label])
        class_lengths.append(len(image_sample_dict[label]))
        labels.append(label)
        
    print('Total amount of input images: ', total_images)
        
    plt.figure(figsize = (10,30))
    plt.barh(range(len(class_lengths)), class_lengths)
    plt.yticks(range(len(labels)), labels)
    plt.title(title)
    plt.ylabel('Dog Breed')
    plt.xlabel('Sample size')
    if save_:
        plt.savefig('dog_breed_distribution.png')
    
    plt.show()
    return total_images

if __name__=='__main__':
    main()
    total_images = plot_class_distributions(EDA(image_input_dir,image_ann_dir).get_input_image_dict,
                                             save_=True)