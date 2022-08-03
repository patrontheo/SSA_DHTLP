import os
import pandas as pd
import numpy as np
from utils.synthetic_generator import generator_dhtlp
from PIL import Image
import json

def create_set(imageDir, outDir, n=1):
    images = []
    for root, _, files in os.walk(imageDir):
        for name in files:
            if name.endswith('.fit'):
                images.append(os.path.join(root,name))

    if not os.path.exists(os.path.join(outDir,'images')):
        os.makedirs(os.path.join(outDir,'images'), exist_ok=True)

    
    train = []
    test = []

    df = pd.read_csv(os.path.join(imageDir, 'labels.csv'), header=None, sep=',')
    df.columns = ['type', 'x', 'y', 'filename', 'width', 'height']

    for counter, image in enumerate(images):
        
        split = np.random.randint(0,4) #Do the split here to have all images from the same original image in the same set

        for i in range(n): #use n times the same image to create n different synthetic streaks
            file = os.path.split(image)[1]
            img_name = os.path.splitext(file)[0] + '_{}'.format(i) + '.png'
            img_name_annot = os.path.splitext(file)[0] + '.png'

            crop, y_true, coord2 = generator_dhtlp(image, n_streaks=1) #load original image and add a synthetic streak
            
            coord1 = df[df.filename == img_name_annot][['x', 'y']].values #get coordinates of the original streak

            p1, p2 = coord1
            q1, q2 = coord2[0] #get coordinates of the added synthetic streak

            pil_image=Image.fromarray(np.uint8(crop*255))
            if os.path.exists(os.path.join(outDir, 'images', img_name)): #check if image already exists
                img_name = os.path.splitext(img_name)[0] + '_{}'.format(counter) + '.png'
            assert not os.path.exists(os.path.join(outDir, 'images', img_name))

            pil_image.save(os.path.join(outDir, 'images',img_name)) #save image

            # To do the split randomly
            # if (split != 0): #75/25 split
            #     train.append({'filename': img_name, 'lines': [[int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])], [q1[0], q1[1], q2[0], q2[1]]], 'height': crop.shape[0], 'width': crop.shape[1]})
            # else:
            #     test.append({'filename': img_name, 'lines': [[int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])], [q1[0], q1[1], q2[0], q2[1]]], 'height': crop.shape[0], 'width': crop.shape[1]})
            
            # To do the split by hand
            if (image.find('train_split') != -1): 
                train.append({'filename': img_name, 'lines': [[int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])], [q1[0], q1[1], q2[0], q2[1]]], 'height': crop.shape[0], 'width': crop.shape[1]})
            elif (image.find('val_split') != -1):
                test.append({'filename': img_name, 'lines': [[int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])], [q1[0], q1[1], q2[0], q2[1]]], 'height': crop.shape[0], 'width': crop.shape[1]})
            else:
                raise ValueError('Image not in train nor val split')

    with open(os.path.join(outDir, "train.json"), "w") as outfile:
        json.dump(train, outfile)
    
    with open(os.path.join(outDir, "valid.json"), "w") as outfile:
        json.dump(test, outfile)



# path to the folder containing the file label.csv for the annotations, and two folders 'train_split' and 'val_split' for the images
imageDir = 'img/split' 

# path to the folder where the dataset will be saved
outDir = "data/raw_dataset/"
create_set(imageDir, outDir, n=1)
