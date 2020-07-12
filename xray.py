import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model=load_model(os.path.join(BASE_DIR, 'ml_model', 'xray_model.h5')

img= image.load_image('sample.png',target_size=(224,224))
img=image.image_to_array(img)
img=np.expand_dims(img,axis=0)
p=model.predict_classes(img)
print(p)