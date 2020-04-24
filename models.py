import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16, VGG19
from keras.layers import Dense, Dropout, Flatten, Input
from keras.models import Model
from data import data
from keras.optimizers import Adam, RMSprop
from keras.layers import Conv3D, MaxPooling3D, BatchNormalization, LSTM
import sys

class Models:
    def __init__(self, model, input_shape=(224, 224, 3), optimizer = 'Adam'):
        self.data = data()
        self.labels = self.data.get_labels('data/train')

        if model == 'vgg16':
            print('Loading vgg16 model...')
            self.model = self.vgg(vggno=16, input_shape=input_shape)
        elif model == 'vgg19':
            print('Loading vgg19 model...')
            self.model = self.vgg(vggno=19, input_shape=input_shape)
        elif model == '3dcnn':
            print('Loading 3d cnn model...')
            self.model = self.cnn3d(input_shape=input_shape)
        elif model == 'r3dcnn':
            print('Loading 3d rcnn model...')
            self.model = self.r3dcnn(input_shape=input_shape)
        else:
            print('{} is not a valid model!'.format(model))
            sys.exit()

        # compile the model
        optimizer = Adam(learning_rate=0.01,amsgrad=False) if optimizer.lower() == 'adam' else RMSprop(learning_rate=0.01, rho=0.9)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        print('Model compiled!!!')

    def vgg(self, input_shape = (224,224,3), scratch=False, vggno=16):
        # remove last layer of model
        if scratch and vggno == 16:
            model = VGG16(weights = None, include_top = False)
        elif scratch and vggno == 19:
            model = VGG19(weights=None, include_top=False)
        elif not scratch and vggno == 16:
            model = VGG16(weights='imagenet', include_top=False)
        else:
            model = VGG19(weights='imagenet', include_top=False)

        # input to model
        img_input = Input(shape=input_shape, name='img_input')

        if scratch:
            for layer in model.layers:
                layer.trainable = True
        else:
            for layer in model.layers:
                layer.trainable = False

        # obtain current output of model w/o fully connected layers
        output = model(img_input)

        # append the fully connected layers to model
        x = Flatten(name='flatten')(output)
        x = Dropout(0.5)(x)
        x = Dense(1, activation='softmax', name='predictions')(x)

        model = Model(input = img_input, output = x)
        return model

    def cnn3d(self, input_shape = (224, 224, 3)):
        # input layer
        img_input = Input(shape=input_shape, name='img_input')  # input to model
        # block layer 1
        x = Conv3D(filters=64, kernel_size=(3,3,3), padding='same')(img_input)
        x = BatchNormalization()(x)
        x = MaxPooling3D(strides=(3,3,3), padding='same')(x)
        # block layer 2
        x = Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling3D(strides=(3, 3, 3), padding='same')(x)
        # block layer 3
        x = Conv3D(filters=256, kernel_size=(3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling3D(strides=(3, 3, 3), padding='same')(x)
        # block layer 4
        x = Conv3D(filters=512, kernel_size=(3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling3D(strides=(3, 3, 3), padding='same')(x)
        # block layer 2
        x = Conv3D(filters=512, kernel_size=(3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling3D(strides=(3, 3, 3), padding='same')(x)
        # fully connected layer
        x = Flatten()(x)
        x = Dense(units=512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(units=512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Flatten(name='flatten')(x)
        x = Dropout(0.5)(x)
        x = Dense(1, activation='softmax', name='predictions')(x)

        model = Model(input = img_input, output = x)
        return model

    def r3dcnn(self, input_shape=(224, 224)):
        # input layer
        img_input = Input(shape=input_shape, name='img_input')  # input to model
        # block layer 1
        x = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same')(img_input)
        x = BatchNormalization()(x)
        x = MaxPooling3D(strides=(3, 3, 3), padding='same')(x)
        # block layer 2
        x = Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling3D(strides=(3, 3, 3), padding='same')(x)
        # block layer 3
        x = Conv3D(filters=256, kernel_size=(3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling3D(strides=(3, 3, 3), padding='same')(x)
        # block layer 4
        x = Conv3D(filters=512, kernel_size=(3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling3D(strides=(3, 3, 3), padding='same')(x)
        # block layer 2
        x = Conv3D(filters=512, kernel_size=(3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling3D(strides=(3, 3, 3), padding='same')(x)
        # fully connected layer
        x = Flatten()(x)
        x = Dense(units=512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(units=512, activation='relu')(x)
        x = Flatten()(x)
        x = LSTM(512, return_sequences=False, dropout=0.5, input_shape=x.shape)(x)
        x = Dropout(0.5)(x)
        x = Flatten(name='flatten')(x)
        x = Dropout(0.5)(x)
        x = Dense(1, activation='softmax', name='predictions')(x)

        model = Model(input=img_input, output=x)
        return model
