#complete face to landmark detection
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

img = cv2.imread('file')
bgr_idx = {0: (227, 243, 247), 1: (227, 243, 247), 2: (168, 139, 4), 3: (67, 101, 240), 4: (168, 139, 4), 5: (67, 101, 240), 6: (129, 93, 153), 7: (255, 193, 222), 8: (129, 93, 153), 9: (255, 193, 222), 10: (129, 93, 153), 11: (77, 163, 109), 12: (77, 163, 109), 13: (124, 240, 255), 14: (124, 240, 255)}
model = load_model('data/output/model_files/30landmarks_2.keras')
face_cascade = cv2.CascadeClassifier('data/input/model_files/haarcascade_frontalface_default.xml')

original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 5)

all_cordinates=[]

for (x,y, w,h) in faces:
  face = gray[y:y+h, x:x+w]
  face = cv2.resize(face, (96,96))
  np_face = np.array(face)/255
  np_face = np.expand_dims(np.expand_dims(np_face, axis=0), axis=3)

  predictions = model.predict(np_face)
  landmarks = (predictions.squeeze()*48)+48
  all_cordinates.append( [(landmarks[::2]*w/96) + x, (landmarks[1::2]*h/96)+y ])

plt.imshow(original)
for cord in all_cordinates:
  for i in range(15):
    color=bgr_idx[i]
    rgb = (color[2]/255, color[1]/255, color[0]/255)
    plt.scatter(cord[0][i], cord[1][i], color=rgb, s=10)
plt.savefig('output/images/sampletest.png')