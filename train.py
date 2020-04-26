import os
import numpy as np
from keras.callbacks import Callback, TensorBoard, EarlyStopping, CSVLogger
from keras.utils import to_categorical
from models import Models as myModels
from data import data as theData
import sys
from keras.preprocessing.image import ImageDataGenerator

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

def train(model, batch_size, nb_epoch, image_shape, optimizer):
    tb = TensorBoard(log_dir=os.path.join('logs',model))
    early_stopper = EarlyStopping(patience=5)
    csv_logger = CSVLogger(os.path.join('logs',model,model+'-training'+'.log'))
    metricsCheckpoint = MetricsCheckpoint('logs')
    steps_per_epoch = 120

    model = myModels(model, image_shape, optimizer)
    data = theData()
    datagen = ImageDataGenerator()
    train_x, train_y = np.array(data.x_train), np.array(data.y_train)
    test_x, test_y = np.array(data.x_test), np.array(data.y_test)
    train_gen = datagen.flow(train_x,to_categorical(train_y))
    test_gen = datagen.flow(test_x, to_categorical(test_y))

    # model.model.fit(train_x, train_y,
    #                 callbacks=[tb, early_stopper, csv_logger],
    #                 validation_data=(test_x,test_y), validation_steps=60,
    #                 steps_per_epoch=steps_per_epoch, batch_size=batch_size)
    model.model.fit_generator(train_gen, epochs=nb_epoch,
                    callbacks=[tb, early_stopper, csv_logger],
                    validation_data=test_gen, validation_steps=60,
                    steps_per_epoch=steps_per_epoch)

def main():
    class_limit = int(2)
    image_height = int(224)
    image_width = int(224)

    # train parameters
    model = '2dcnn'
    batch_size = 32
    nb_epoch = 100
    optimizer = 'Adam'
    image_shape = (image_height, image_width, 3)

    train(model, batch_size, nb_epoch, image_shape, optimizer)

if __name__ == "__main__":
    main()