import os
import sys
import glob
import time
import imageio

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.misc import imresize 
from sklearn.model_selection import train_test_split

from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator


RANDOM_SEED=159
pd.set_option('display.max_colwidth', -1)

imagedir = "/home/curtis/datasets/bloodcells/dataset-master/JPEGImages/"
labelsfile = "/home/curtis/datasets/bloodcells/dataset-master/labels.csv"
img_width = 320
img_height = 200

labels = pd.read_csv(labelsfile)
labels = labels.loc[:, ['Image', 'Category']]
labels['Image'] = pd.to_numeric(labels['Image'])
labels = labels.dropna()
labels = labels[~labels.Category.str.contains(",")]

labels = pd.get_dummies(labels, columns=['Category'])
# Category_BASOPHIL  Category_EOSINOPHIL  Category_LYMPHOCYTE  Category_MONOCYTE  Category_NEUTROPHIL
basophil = labels['Category_BASOPHIL'].sum()
eosinophil = labels['Category_EOSINOPHIL'].sum()
lymphocyte = labels['Category_LYMPHOCYTE'].sum()
neutrophil = labels['Category_NEUTROPHIL'].sum()
monocyte = labels['Category_MONOCYTE'].sum()
print(labels.describe())
print("""
Basophil   {bas}
Eosinophil {eos}
Lymphocyte {lym}
Neutrophil {neu}
Monocyte   {mono}
""".format(bas=basophil, eos=eosinophil, lym=lymphocyte, neu=neutrophil,
           mono=monocyte))

labels['filename'] = labels.Image.apply(lambda x: imagedir + "BloodImage_" + str(x).zfill(5) + ".jpg")
# Filter for only image files that actually exist
labels = labels[labels['filename'].apply(lambda x: os.path.isfile(x))]

# Load and resize images
images = []
for x in labels['filename']:
    img = imageio.imread(x)
    img = imresize(img, (img_height,img_width))
    images.append(img)

images = np.array(images)
print(images.shape)

#plt.imshow(images[0])
#plt.show()

# Reduce to only categories
y = labels.loc[:,['Category_BASOPHIL','Category_EOSINOPHIL',
                  'Category_LYMPHOCYTE', 'Category_MONOCYTE',
                  'Category_NEUTROPHIL']].values

X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=0.2, stratify=y)

train_gen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, rotation_range=30)
test_gen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_gen.flow(np.array(X_train), 
                                 y_train, 
                                 batch_size=8) 
validation_generator = test_gen.flow(np.array(X_test), y_test)
test_generator = test_gen.flow(np.array(X_test), y_test)

# Model Building
model = Sequential()
model.add(Conv2D(100, activation='relu', kernel_size=3,
                 input_shape=(img_height, img_width, 3), padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(30, activation='relu', kernel_size=3, padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(30, activation='relu', kernel_size=3, padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(30, activation='relu', kernel_size=3, padding='same'))
model.add(Flatten())
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

model.fit_generator(
    train_generator, 
    validation_data=validation_generator,
    steps_per_epoch=len(X_train),
    validation_steps=len(X_test),
    epochs=10
)
model.fit(X_train, y_train, validation_split=0.2, epochs=10)

print(model.evaluate(X_test, y_test))

