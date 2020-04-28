from random import shuffle

import cv2
from keras.preprocessing import image as image_utils
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from sklearn.model_selection import train_test_split
import shutil
import pandas as pd
import subprocess
import sys
import pathlib

class data:
    def __init__(self):
        # if not os.path.isfile('data.csv'): self.move_data(data_folder='data')
        # if not os.path.isfile('test_df.csv') and not os.path.isfile('train_df.csv'):
        #     self.get_traintest_dfs()
        # self.make_data_folders()
        # self.move_to_train_test_dir('test_df.csv')
        # self.move_to_train_test_dir('train_df.csv')
        # os.removedirs('theFrames')
        # self.train_gen, self.test_gen = self.load_generators()
        self.x_train, self.y_train = self.load_to_memory(['new_path','label'],'train_df.csv')
        self.x_test, self.y_test = self.load_to_memory(['new_path', 'label'], 'test_df.csv')

    def rename_classes_folders(self,base_dirs = ['data/train', 'data/test']):
        if ' ' in os.listdir(base_dirs[0])[0]:
            for base_dir in base_dirs:
                classes = os.listdir(base_dir)
                for clas in classes:
                    path = os.path.join(base_dir, clas)
                    path_new = os.path.join(base_dir, clas[3:])
                    os.rename(path, path_new)

    def img_to_array(self, image):
        image = image_utils.load_img(image)
        image = image.resize((224,224))
        image = image_utils.img_to_array(image)
        #normalize image
        image /= 255.0
        return image

    def get_data_ready(self, csv):
        df = pd.read_csv(csv)
        for row in df.iterrows():
            row = row[1]
            img = image_utils.load_img(row['new_path'])
            img = img.resize((224,224))
            img = image_utils.img_to_array(img)
            img /= 255.0
            img = image_utils.array_to_img(img)
            os.remove(row['new_path'])
            img.save(row['new_path'])

    def get_labels(self, label_col, df):
        labels = list(set(df[label_col]))
        with open('labels.txt', 'w') as f:
            for label in labels:
                f.write(label+'\n')
        return labels
    # print(get_labels('data/00'))

    def to_category(self, label):
        classes = self.get_labels('data/00')
        return classes.index(label)

    def move_data(self, data_folder='data', new_data_folder='theFrames'):
        dataCSV = []
        if not os.path.isdir(new_data_folder): os.mkdir(new_data_folder)
        for group in os.listdir(data_folder):
            group_flds = os.listdir(os.path.join(data_folder,group))
            for group_fld in group_flds:
                clas = group_fld
                if '_' in clas: clas = str(clas).replace('_',' ')
                group_imgs_dir = os.path.join(data_folder,group,group_fld)
                for img in os.listdir(group_imgs_dir):
                    img_dir = os.path.join(data_folder,group,group_fld,img)
                    dest_dir = os.path.join(new_data_folder,img)
                    shutil.move(img_dir, dest_dir)
                    dataCSV.append([img.split('.')[0], dest_dir, clas])
        df = pd.DataFrame(dataCSV, columns=['image','path','label'])
        df.to_csv('data.csv')
        subprocess.call(['rm','-r','data'])
        # subprocess.call(['mv','/data/','/data_org/'])

    def make_data_folders(self, csv='data.csv'):
        df = pd.read_csv(csv)
        df['label'] = df['label'].apply(lambda x: x[3:])
        labels = set(df['label'])
        df.to_csv('data.csv')
        splits = ['train', 'test']
        if  not os.path.isdir('data'): os.mkdir('data')
        for split in splits:
            train_test = os.path.join('data',split)
            if not os.path.isdir(train_test):
                os.mkdir(train_test)
            for label in labels:
                label_dir =os.path.join(train_test, str(label))
                if not os.path.isdir(label_dir):
                    os.mkdir(label_dir)

    def get_traintest_dfs(self, data_csv='data.csv'):
        df = pd.read_csv(data_csv)
        df.sample(frac=1)
        fracs = np.array([0.8, 0.2])
        train_df, test_df = np.array_split(
            df, (fracs[:-1].cumsum() * len(df)).astype(int)
        )

        train_df = pd.DataFrame(train_df)
        train_df.to_csv('train_df.csv')
        test_df = pd.DataFrame(test_df)
        test_df.to_csv('test_df.csv')
        return train_df, test_df

    def move_to_train_test_dir(self,split_csv):
        split = str(split_csv).split('_')[0]

        if split not in ['test', 'train']:
            print('Invalid CSV')
            sys.exit()

        df = pd.read_csv(split_csv)
        folder_split = os.path.join('data', split)

        new_paths = []

        for row in df.iterrows():
            folder_dest = os.path.join(folder_split, str(row[1]['label'])[3:], str(row[1]['image'])+'.png')
            shutil.move(row[1]['path'], folder_dest)
            # print(row[1]['path'], folder_dest)
            new_paths.append(folder_dest)
        # sys.exit()

        df['new_path'] = pd.Series(new_paths, index=df.index)
        df = df[['image','path','new_path','label']]
        df.to_csv(split_csv)

    def load_to_memory(self,columns,csv, preprocess=False):
        if len(columns) == 0:
            raise ValueError('Empty list of columns')
        elif len(columns) == 2:
            path_col = columns[0]
            label_col = columns[1]
        else:
            raise Warning('Please provide two columns')

        df = pd.read_csv(csv)
        df = df[[path_col,label_col]]
        df[label_col] = df[label_col].apply(lambda x: x[3:])
        if preprocess: self.get_data_ready(csv)

        # if os.path.isfile('labels.txt') or self.labels is None:
        with open('labels.txt', 'r') as f:
            self.labels = [line.rstrip('\n') for line in f]
        # else:
        self.labels = self.get_labels(label_col,df)
        # print(self.labels)
        # print(set(df[label_col]))

        x, y = [], []
        for row in df.iterrows():
            img = cv2.imread(row[1][path_col])
            label = self.labels.index(row[1][label_col])
            x.append(np.array(img))
            y.append(int(label))
        assert len(x) == len(y)
        return x, y

    def load_generators(self, train = 'data/train', test = 'data/test'):
        # if os.path.isfile('labels.txt') or self.labels is None:
        #     with open('labels.txt', 'r') as f:
        #         self.labels = [line.rstrip('\n') for line in f]
        # else:
        self.labels = self.get_labels('label',pd.read_csv('train_df.csv'))
        self.get_data_ready('train_df.csv')
        self.get_data_ready('test_df.csv')

        img_gen = ImageDataGenerator()
        train_gen = img_gen.flow_from_directory(train,
                                                classes=self.labels,
                                                class_mode='categorical',
                                                target_size=(224, 224))
        test_gen = img_gen.flow_from_directory(test,
                                                classes=self.labels,
                                                class_mode='categorical',
                                               target_size=(224,224))

        return train_gen, test_gen
