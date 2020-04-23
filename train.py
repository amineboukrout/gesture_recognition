import os
import numpy as np
from keras.callbacks import Callback, TensorBoard, EarlyStopping, CSVLogger
from keras.utils import to_categorical
from models import Models as myModels
from data import data as theData

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

    data = theData('data')
    x_train, x_test, y_train, y_test = data.x_train, data.x_test, data.y_train, data.y_test
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)
    model = myModels(model, image_shape, optimizer)

    model.model.fit(x_train, y_train, epochs=nb_epoch,
                    callbacks=[tb, early_stopper, csv_logger],
                    validation_data=(x_test,y_test), validation_steps=120,
                    steps_per_epoch=steps_per_epoch,use_multiprocessing=True)

def main():
    class_limit = int(2)
    image_height = int(224)
    image_width = int(224)

    # train parameters
    model = 'vgg16'
    batch_size = 32
    nb_epoch = 100
    optimizer = 'Adam'
    image_shape = (image_height, image_width, 3)

    train(model, batch_size, nb_epoch, image_shape, optimizer)

if __name__ == "__main__":
    # main()
    new_dataset_name = 'data_'
    for class_folders_num in os.listdir('data'):
        class_folders_dir = os.path.join('data', class_folders_num)
        for class_folders in os.listdir(class_folders_dir):
            if not os.path.isdir(os.path.join(new_dataset_name,class_folders[3:])):
                if not os.path.isdir(new_dataset_name): os.mkdir(new_dataset_name)
                os.mkdir(os.path.join(new_dataset_name,class_folders[3:]))
