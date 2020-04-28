# Hand Gesture Recognition
## Introduction
This is a Machine Learning model that adapts two pretrained models to perform an automatic hand gesture recognition. This is a guide on using this tool. However, NOTE this is still under production.
## Data
The data used to train and test the model was obtained from [Kaggle](kaggle.com) making use of the [Hand Gesture Recognition Dataset](https://www.kaggle.com/gti-upm/leapgestrecog). It is a dataset containing infrared images of gestures of hands. Please download the dataset and do the following:
   1. Download the dataset, [leapgetrecog.zip](https://www.kaggle.com/gti-upm/leapgestrecog/download), and place it in your project's directory.
   2. Unzip the dataset, you will see two folders within the main folder, namely leapGestRecog and leapgestrecog. 
   3. Move the leapGestRecog to the main directory, and rename the folder to 'data'. This folder should consist of 10 sub-folders. Each of the subsequent folders will have 10 sub-folders.
## Training
Training does take a long time. The models that are adapted to this recognition task. The requirements, to be installed, are as follows:
   * Python 3
   * OpenCV2
   * Keras
   * Pandas
   * Numpy
A *requirements.txt* will be made available in due course. 
Run the following to train the models:
	`python3 train.py`
This will train the model. This is not the final documentation and will be updated accordingly.
## Running the Classification Model
To be added...
## Results
To be added...
