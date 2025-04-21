

# Loading of Libraries
from __future__ import division
import numpy as np
import os
import glob


from random import *
from PIL import Image
from keras.utils  import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib. image as mpimg
# %matplotlib inline


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation, BatchNormalizaton 
from keras.layers.convlutional import Convolution2D, Croppping2D, ZeroPadding2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam, RMsprop
from google.colab import drive


drive.mount('/content/gdrive')

#These are the forms in the dataset for quick access from manipulation of the file names on each coloumn. Let's create a dictionary with form and writer ma
d = {}
from subprocess import check_output
with open('gdrive/My Drive/Forms.txt')as f:
    for line in f:
        key = line.split('')[0]
        writer = line.split('')[1]
        d[key] = writer
print(len(d.keys()))


#ALl file-names list and target-writer names list are created.
tmp = []
target_list = []

path_to_files = os.path.join('gdrive/My Drive/dataFromIAm','*')
for filename in sorted(glob.glob(path_to_files)):
    
    

# print(filename)
    tmp.append(filename)
    image_name = filename.split('/')[-1]
    file, xt = os.path.splitext(image_name)
    parts = file.split('-')
    form = parts[0] + '-' + parts[1]
    for key in d:
        if key == form:
            target_list.append(str(d[form]))
    
    
img_files = np.asarray(tmp)
img_targets = np .asarray(target_list)
print(img_files.shape)
print(img_targets.shape)

#visualize the image data
for filename in img_files[:3]:
    img=mpimg.imread(filename)
    plt.figure(figuresize=(10,10))
    plt.imshow(img , camp='gray')
    
    

# normalisation is done using label encoder. NO, categorical data.
encoder = LaberlEncoder()
encoder.fit(img_targets)
encoded_Y = encoder.transform(img_targets)

print(img_files[:5], img_targets[:5], encoded_Y[:5])


#Spliting of data into training and validation sets for cross validation with 4:1:1 ratio.
train_files, rem_files, train_targets, rem_targets =
train_test_split(
    img_files,encoded_Y, train_sizze=0.66, random_state=52, 
    shuffle=True)
    
validation_files, test_files, validation_targets, test_targets = 
train_test_split(
    rem_files, rem_targets, train_size=0.5, random_state=22,
shuffle=True)

print(train_files.shape, validation_files.shape,
test_files.shape)
print(train_targets.shape, validation_targets.shape,
test_targets.shape)

#Input to the model 

#We take pathces of data, each of size 113x113. A generator function is impleneted for that purpose.

#Generator function for generating random crops from each sentence

# # Now create generators for randomly cropping 113x113 patches from these images 

batch_size = 16
num_clases = 50 

# Start with train generator shared in the class and add images augmentations

def generate_data(samples,traget_files, batch_size=batch_size, factor = 0.1):
    num_samples = len(samples)
    while 1: #Loop forever so the generator never terminates
    for offset in range(0, num_samples, batch_size):
        batch_samples = samples[offset:offset+batch_size]
        batch_targets = target_files
    [offset:offset+batch _size]
    
                images = []
                target = []
                for i in range(len(batch_samples)):
                batch_sample = batch_samples[i]
                batch_targets = batch_targets[i]
                im = Image.open(batch_sample)
                cur width = im.size[0]
                cur_height = im.size[1]
                
                #print(curwidth, cur_height)
                height fac = 114 / cur height
                
                new_width = int(cur_width * height fac)
                size = new_width, 113
                
                imresize = im.resize((size) ,Image.ANTIALIAS) #Resize so height = 113 while keeping aspect ratio      
                now_width = imresize.size[0]
                now_height = imresize.size[1]
                #Generate crops of size 113x113 from this resized image and keep random 10% of crops
                
                
                avail x points = list(range(0, now_width - 113))
    # total x start points are form 0  to with 113
    
                # Pick random x%
                pick_num = int(len(avail x points)* factor)
                
                # Now Pick
                random_startx = sample(avail x points, pick_num)
                
                for start in random_startx:
                    imcrop = imresize.crop((start, 0, start+113,113))
                    
                    images.append(np.asarray(imcrop))
                    targets.append(batch_target)
                    
                # trim image to only see section with road
                X_train = np.array(images)
                Y_train = np.array(targets)
                
                # reshape X_train for feeding in lator
                X_train =  X_train.reshape(X_train.shape[0], 113, 113, 1)
                
                # convert to float and normalize
                X_train = X_train.astype('float32')
                X_train /= 255
                
                # One hot encode y
                Y_train = to to_categorical(Y_train, num_classes)
                yield shuffle(X_train, Y_train)
                
            
# For training and testing, generator function is called with the intent of making train and test generator data.
train_generator = generate_data(train_files, train_targets, batch_size= batch_size, factor = 0.3)
validation_generator = generate_data(validation_files, validation_targets, batch_size=batch_size, factor = 0.3)
test_generator = generate_data(test_files, test_targets, batch_size=batch_size, factor = 0.1) 



# A Keras Model built. Summary Of the model printed below.
def resize image ( image) :
    import tensorflow as tf
    return tf.image.resize(image, [56, 56] )
    
# Function to resize image to 64x64
row, col, ch = 113, 113, 1

model = Sequential ()
model.add(ZeroPadding2D((1, 1), input_shape= (row, col, ch) ) )

# Resize data within the neural network
model.add(Lambda(lambda x: resize image) ) #resize images to allow Eor easy
computation
# model.add (Lambda (lambda x: resize image) )

# CNN model — Building the model suggested in paper

model.add(Convolution2D(filters= 32, kernel_size = (5,5), strides= (2,2), padding='same', name='convl'))  #96
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='pool1'))

model.add(Convolution2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', name='conv2')) #256
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='pool2'))

model.add(Flatten())
model.add(Dropout(O.5))

model.add(Dense(512, name='dense1')) #1024
# model. add (3atchNo:maIization ( ) )
model.add(Activation('relu'))
model.add(Dropout(O.5))

model.add(Dense(256, name='dense2') )  #1024
model.add(Activation('relu') )
model.add(Dropout(O.5))


model.add(Dense(num_classes,name='output') )
model.add(Activation('softmax')) #softmax since output is within 50 classes
print ("Worked  till here")

model.compile(loss='to_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])



print (model.summary())



# Training the Model
nb_epoch = 8
samples_per_epoch = 3268
nb_val_samples = 842

# #save every model using Keras checkpoint
from keras.callbacks import ModelCheckpoint
filepath='checkpoint2/check-{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath= filepath, verbose=1,
save_best_only=False)
callbacks_list = [checkpoint]

# # Model  fit generator
history_object = model.fit_generator(train_generator,
samples_per_epoch= samples_per_epoch,

validation_data =validation_generator,

nb_val_samples=nb_val_samples, nb_epoch=nb_epoch, verbose=1,
callbacks=callbacks_list)

