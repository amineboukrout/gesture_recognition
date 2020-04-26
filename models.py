import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16, VGG19
from keras.layers import Dense, Dropout, Flatten, Input
from keras.models import Model
from keras import models
from keras.optimizers import Adam, RMSprop
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, LSTM
import sys

class Models:
    def __init__(self, model, input_shape=(224, 224, 3), optimizer = 'Adam'):
        with open('labels.txt', 'r') as f:
            self.labels = [line.rstrip('\n') for line in f]

        if model == 'vgg16':
            print('Loading vgg16 model...')
            self.model = self.vgg(vggno=16, input_shape=input_shape)
        elif model == 'vgg19':
            print('Loading vgg19 model...')
            self.model = self.vgg(vggno=19, input_shape=input_shape)
        elif model == '2dcnn':
            print('Loading 3d cnn model...')
            self.model = self.cnn()
        else:
            print('{} is not a valid model!'.format(model))
            sys.exit()

        # compile the model
        optimizer = Adam(learning_rate=0.01,amsgrad=False) if optimizer.lower() == 'adam' else RMSprop(learning_rate=0.01, rho=0.9)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        print('Model compiled!!!')

    def vgg(self, input_shape = (224,224,3), scratch=False, vggno=16):
        # input to model
        # img_input = Input(shape=input_shape, name='img_input')
        # remove last layer of model
        if scratch and vggno == 16:
            model = VGG16(input_shape=input_shape,weights = None, include_top = False)
        elif scratch and vggno == 19:
            model = VGG19(input_shape=input_shape,weights=None, include_top=False)
        elif not scratch and vggno == 16:
            model = VGG16(input_shape=input_shape,weights='imagenet', include_top=False)
        else:
            model = VGG19(input_shape=input_shape, weights='imagenet', include_top=False)


        if scratch:
            for layer in model.layers:
                layer.trainable = True
        else:
            for layer in model.layers:
                layer.trainable = False
        vgg = model
        model = models.Sequential()

        # obtain current output of model w/o fully connected layers
        model.add(vgg)

        # append the fully connected layers to model
        # model.add(Flatten(name='flatten'), input_shape=(None, None, 512))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(10, activation='softmax', name='predictions'))

        # model = Model(input = img_input, output = x)
        print(model.summary())
        return model

    def cnn(self, input_shape = (224, 224, 3)):
        # input layer: 32 IS THE BATCH
        # img_input = Input(shape=input_shape, name='img_input')  # input to model
        model = models.Sequential()
        # block layer 1
        model.add(Conv2D(input_shape=input_shape, filters=64, kernel_size=(3,3), padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(strides=(3,3), padding='same'))
        # block layer 2
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(strides=(3,3), padding='same'))
        # block layer 3
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(strides=(3,3), padding='same'))
        # block layer 4
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(strides=(3,3), padding='same'))
        # block layer 2
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(strides=(3,3), padding='same'))
        # fully connected layer
        model.add(Dense(units=512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(units=512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Flatten(name='flatten'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax', name='predictions'))

        # model = Model(input = img_input, output = x)
        print(model.summary())
        return model