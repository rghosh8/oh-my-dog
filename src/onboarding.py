import numpy as np 
from static import *
import xml.etree.ElementTree as ET
from PIL import Image

def onboarding(all_images_, all_breeds_):
    '''
        Takes all images and breeds
        makes them ready for modeling
        args:
            all_images_: list of all images
            all_breeds_: list of all breeds
        outputs:
            images_input: return vectorized version of images
            breeds_name:  numpy array of dog breeds
    '''
    
    breeds_names = np.array([], dtype='str')
    images_inputs = np.zeros((len(all_images_), image_height, image_width, image_channels))
    
    idxIn = 0
    for breed in all_breeds_:
        if breed == '.DS_Store': 
            continue

        for image_name in os.listdir(breed_path+breed): 
            try: 
                img = Image.open(image_path+image_name+'.jpg') 
            except: 
                continue 
            # print('g')             
            tree = ET.parse(breed_path+breed+'/'+image_name)
            # print(tree)
            root = tree.getroot() 
            objects = root.findall('object') 
            o = objects[0]
            bndbox = o.find('bndbox') # <bndbox>
            xmin = int(bndbox.find('xmin').text) 
            ymin = int(bndbox.find('ymin').text) 
            xmax = int(bndbox.find('xmax').text) 
            ymax = int(bndbox.find('ymax').text) 
            w = np.min((xmax - xmin, ymax - ymin))
            img2 = img.crop((xmin, ymin, xmin+w, ymin+w))
            img2 = img2.resize((image_height,image_width), Image.ANTIALIAS)
            images_inputs[idxIn,:,:,:] = np.asarray(img2)
            idxIn += 1
            breeds_names = np.append(breeds_names, breed)

    normalized_image_vectors = images_inputs/255.0
            
    return images_inputs, np.asarray(normalized_image_vectors), breeds_names

if __name__=='__main__':
    images_inputs, normalized_image_vectors, breeds_names = onboarding(all_images, all_breeds)
