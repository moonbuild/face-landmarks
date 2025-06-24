import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


df = pd.read_csv('data/input/model_files/training_new.csv')
df.dropna(inplace=True)
imgs = df['Image'].apply(lambda i : np.fromstring(i, sep=' ')).values
imgs = np.vstack(imgs)/255
imgs = imgs.reshape(-1,96,96,1)

landmarks = df[df.columns[:-1]].values
landmarks = (landmarks-48)/48

X_train, X_test, y_train, y_test = train_test_split(imgs, landmarks, test_size=0.2, random_state=42)
#show multiple images with matplotlib

fig, axes = plt.subplots(2, 2, figsize=(5, 5))
axes = axes.flatten()
for i, idx in enumerate(range(4)):
  ax = axes[i]
  ax.imshow(X_test[idx], cmap='gray')
  ax.scatter(y_test[idx][0::2], y_test[idx][1::2], c='r', s=10)
  ax.axis('off')

plt.show()