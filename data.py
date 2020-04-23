import cv2
from keras.preprocessing import image as image_utils
import numpy as np
import os
from sklearn.model_selection import train_test_split
import shutil
import pandas as pd

# def rename():

class data:
    def __init__(self, data_folder):
        if os.path.isfile('splits.npz'):
            self.load_sets()
        else:
            x, y = self.retrieve_data(data_folder)
            self.save_sets(x, y)

    def img_to_array(self, image):
        image = image_utils.load_img(image)
        image = image.resize((224,224))
        image = image_utils.img_to_array(image)
        #normalize image
        image /= 255.0
        return image

    def get_labels(self, label_dir):
        arr = os.listdir(label_dir)
        labels = []
        for clas in arr:
            label = str(clas).split('_')[1]
            if label not in labels: labels.append(label)
        return labels
    # print(get_labels('data/00'))

    def to_category(self, label):
        classes = self.get_labels('data/00')
        return classes.index(label)

    def retrieve_data(self, data_folder, new_data_folder='theFrames'):
        dataCSV = []
        if not os.path.isdir(new_data_folder): os.mkdir(new_data_folder)
        for group in os.listdir(data_folder):
            group_flds = os.listdir(os.path.join(data_folder,group))
            for group_fld in group_flds:
                clas = group_fld[3:]
                if '_' in clas: clas = str(clas).replace('_',' ')
                group_imgs_dir = os.path.join(data_folder,group,group_fld)
                for img in os.listdir(group_imgs_dir):
                    img_dir = os.path.join(data_folder,group,group_fld,img)
                    dest_dir = os.path.join(new_data_folder,img)
                    shutil.move(img_dir, dest_dir)
                    dataCSV.append([img.split('.')[0], dest_dir, clas])
        df = pd.DataFrame(dataCSV, columns=['image','path','label'])
        df.to_csv('data.csv')

    def make_data_folders(self, csv='data.csv'):
        df = pd.read_csv(csv)
        labels = set(df.label)
        for label in labels:
            if not os.path.isdir('frames'):
                os.mkdir('frames')
            if not os.path.isdir(os.path.join('frames',label)):
                os.mkdir(os.path.join('frames',label))

    def split_data(self, x, y, test_size=0.2):
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(x, y, test_size=test_size, train_size=1-test_size)
        return self.x_train, self.x_test, self.y_train, self.y_test

    def save_sets(self, x, y):
        x_train, x_test, y_train, y_test = self.split_data(x, y)
        np.savez('splits.npz', xtrain=x_train, xtest=x_test, ytrain=y_train, ytest=y_test)
        print('Splits are saved!')

    def load_sets(self, file='splits.npz'):
        arrs = np.load(file)
        self.x_train = arrs['xtrain']
        self.x_test = arrs['xtest']
        self.y_train = arrs['ytrain']
        self.y_test = arrs['ytest']
        return self.x_train, self.x_test, self.y_train, self.y_test

# data('data')