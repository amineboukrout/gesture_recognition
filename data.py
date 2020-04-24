from random import shuffle

import cv2
from keras.preprocessing import image as image_utils
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
        self.move_data(data_folder='data')
        self.get_traintest_dfs()
        self.move_to_train_test_dir('test_df.csv')
        self.move_to_train_test_dir('train_df.csv')

        samples_train = self.load_samples('train_df.csv')
        self.train_gen = self.generator(samples_train)

        samples_test = self.load_samples('test_df.csv')
        self.test_gen = self.generator(samples_test)

    def img_to_array(self, image):
        image = image_utils.load_img(image)
        image = image.resize((224,224))
        image = image_utils.img_to_array(image)
        #normalize image
        image /= 255.0
        return image

    def get_data_ready(self):
        # subprocess.call('wget https://storage.googleapis.com/kaggle-data-sets/39466/61155/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1587989257&Signature=lg6cf5sNrsHEtKtVqEVJHgEDCYqD9DlTyAC9A%2BBDPw9BGE7U5rs9m70l%2FbxCSQSn4MkZ9gsIS3lAg16JXu480vE7dVD4Cq%2B2txSKY05Muo6ZFsGAEtz5bIKtKe10keq1jIrz8ZbihwPRyQ2hJJ1N6g6hfx72%2BxHxEf7v%2Fu7dCzEQ%2B8aYoP6CynPFsyi75BkqpefPAO8GpGX0CorhCwIoKQdfa2c7%2BESzPek2DcRs1dxPLjHAeGkXY1J0mqDHV6at8LM961XiFPWcYjLv7ZMobMA%2F3n69vebLO1uviL1yXUu0HWbohZwlOOvjEgXnlUKn7NUec82U9aJmXfgQMucq1A%3D%3D&response-content-disposition=attachment%3B+filename%3Dleapgestrecog.zip')
        subprocess.call('unzip leapgestrecog.zip')
        subprocess.call('rm -r leapgestrecog/leapgestrecog')
        subprocess.call('mv leapgestrecog/leapGestRecog data')
        subprocess.call('rm -r leapgestrecog')

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

    def make_data_folders(self, csv='data.csv'):
        df = pd.read_csv(csv)
        labels = set(df.label)
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
        self.make_data_folders()

        df = pd.read_csv(split_csv)
        folder_split = os.path.join('data', split)

        new_paths = []

        for row in df.iterrows():
            folder_dest = os.path.join(folder_split, str(row[1]['label']), str(row[1]['image'])+'.png')
            # shutil.move(row[1]['path'], folder_dest)
            new_paths.append(folder_dest)

        df['new_path'] = pd.Series(new_paths, index=df.index)
        df.to_csv(split_csv)

    def load_samples(self,csv_file):
        df = pd.read_csv(csv_file)
        df = df[['new_path','label']]
        file_names = list(df.iloc[:,0])
        labels = list(df.iloc[:,1])
        samples = []
        for sam,lab in zip(file_names,labels): samples.append([sam,lab])
        return samples

    def generator(self, samples, batch_size=32,shuffle_data=True,resize=224):
        """
        Yields the next training batch.
        Suppose `samples` is an array [[image1_filename,label1], [image2_filename,label2],...].
        """
        num_samples = len(samples)
        while True: # Loop forever so the generator never terminates
            shuffle(samples)

            # Get index to start each batch: [0, batch_size, 2*batch_size, ..., max multiple of batch_size <= num_samples]
            for offset in range(0, num_samples, batch_size):
                # Get the samples you'll use in this batch
                batch_samples = samples[offset:offset+batch_size]

                # Initialise X_train and y_train arrays for this batch
                X_train = []
                y_train = []

                # For each example
                for batch_sample in batch_samples:
                    # Load image (X) and label (y)
                    img_name = batch_sample[0]
                    label = batch_sample[1]
                    img =  cv2.imread(os.path.join(img_name))

                    # apply any kind of preprocessing
                    img = cv2.resize(img,(resize,resize))
                    # Add example to arrays
                    X_train.append(img)
                    y_train.append(label)

                # Make sure they're numpy arrays (as opposed to lists)
                X_train = np.array(X_train)
                y_train = np.array(y_train)

                # The generator-y part: yield the next training batch
                yield X_train, y_train


# x, y = next(train_gen)
# print(x.shape)