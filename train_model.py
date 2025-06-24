import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping

import visualkeras

df = pd.read_csv('data/input/model_files/training_new.csv')

df.dropna(inplace=True)

#PreProcessing
imgs = df['Image'].apply(lambda i : np.fromstring(i, sep=' ')).values
imgs = np.vstack(imgs)/255
imgs = imgs.reshape(-1,96,96,1)

landmarks = df[df.columns[:-1]].values
landmarks = (landmarks-48)/48

##Model Architecture
model = Sequential()

model.add(Conv2D(16, (3,3), padding='same', activation='relu', input_shape=(96,96,1)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(30))
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['accuracy']
)
model.summary()
hist = model.fit(imgs, landmarks, validation_split=0.2, epochs=60,callbacks=[early_stopping], batch_size=64,verbose=1)

model.save('data/output/model_files/30landmarks_3.keras')

try:
    visualkeras.layered_view(model, legend=True, to_file='results/landmarks_arch.png')
except Exception as e:
    print(f"Failed to prepare Architecture visualization\n {e}")


#Side-by-Side plot of loss and accuracy
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(hist.history['loss'], label='Train Loss', color='blue')
plt.plot(hist.history['val_loss'], label='Val Loss', color='orange')
plt.title("Model Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(hist.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(hist.history['val_accuracy'], label='Val Accuracy', color='orange')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)

plt.tight_layout()

#will overwrite existing file
plt.savefig('data/output/images/loss_accuracy_plot.png')