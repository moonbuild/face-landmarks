import cv2
import numpy as np
from keras.models import load_model

bgr_idx = {0: (227, 243, 247), 1: (227, 243, 247), 2: (168, 139, 4), 3: (67, 101, 240), 4: (168, 139, 4), 5: (67, 101, 240), 6: (129, 93, 153), 7: (255, 193, 222), 8: (129, 93, 153), 9: (255, 193, 222), 10: (129, 93, 153), 11: (77, 163, 109), 12: (77, 163, 109), 13: (124, 240, 255), 14: (124, 240, 255)}

model = load_model('data/output/model_files/30landmarks_3.keras')
face_cascade = cv2.CascadeClassifier('data/input/model_files/haarcascade_frontalface_default.xml')

cap =cv2.VideoCapture(0)

def detect_landmarks(face, x, y, w, h):
    face = cv2.resize(face, (96,96))
    np_img = np.array(face)/255
    
    inp = np.expand_dims(np.expand_dims(np_img, axis=0), axis=3)

    predictions = model.predict(inp)
    landmarks = (predictions.squeeze()*48)+48
    return (landmarks[::2]*w/96)+x, (landmarks[1::2]*h/96)+y


while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 2, 2)
    
    all_cordinates = []
    
    for (x, y, w, h) in faces:
        x -= 5
        y -= 5
        w += 10
        h += 10
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (96,96))
        np_img = np.array(face)/255
        
        inp = np.expand_dims(np.expand_dims(np_img, axis=0), axis=3)

        predictions = model.predict(inp)
        landmarks = (predictions.squeeze()*48)+48
        all_cordinates.append(((landmarks[::2]*w/96)+x, (landmarks[1::2]*h/96)+y))
    
    for cords in all_cordinates:
        x_lst, y_lst = cords
        for i in range(15):
            bgr_color = bgr_idx[i]
            cv2.circle(frame, (int(x_lst[i]), int(y_lst[i])), radius=2, color=bgr_color, thickness=-1)

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) &0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
