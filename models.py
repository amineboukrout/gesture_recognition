import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16, VGG19
from keras.callbacks import Callback
from keras.layers import Dense, Dropout, Flatten, Input
from keras.models import Model, Sequential
from keras import optimizers
from data import data
from keras.layers import Conv3D, MaxPooling3D, BatchNormalization, LSTM

class MetricsCheckpoint(Callback):
    # saves state of model after each epoch
    def __init__(self, path):
        super(MetricsCheckpoint, self).__init__()
        self.path = path
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        np.save(self.path, self.history)

class Models:
    def __init__(self):
        self.data = data('data')
        self.labels = self.data.get_labels('data/00')

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
        x = Dense(len(self.labels), activation='softmax', name='predictions')(x)

        model = Model(input = img_input, output = x)
        return model

    def cnn3d(self, input_shape = (224, 224)):
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
        x = Dense(len(self.labels), activation='softmax', name='predictions')(x)

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
        x = Dense(len(self.labels), activation='softmax', name='predictions')(x)

        model = Model(input=img_input, output=x)
        return model

    def compile_model(self, model):
        pass