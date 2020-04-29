import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('logs/vgg19/vgg19-training.log')
print(df.head()['val_accuracy'])
plt.figure()
plt.plot(df['epoch'], df['val_accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Validation accuracy')
plt.title('VGG19 validation accuracy')
plt.savefig('figures/fig_vgg19.png')