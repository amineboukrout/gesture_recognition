import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16, VGG19
from keras.callbacks import Callback
from keras.layers import Dense, Dropout, Flatten, Input
from keras.models import Model
from keras import optimizers
from data import data

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

    def vgg16(self, input_shape = (224,224,3), scratch=False, vggno=16):
        # remove last layer of model
        if scratch and vggno == 16:
            model = VGG16(weights = None, include_top = False)
        elif scratch and vggno == 19
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

    