import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16, VGG19
from keras.callbacks import Callback
from keras.layers import Dense, Dropout, Flatten, Input
from keras.models import Model
from keras import optimizers

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

