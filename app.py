from flask import Flask, render_template, Response, request
import cv2
import time
import numpy as np
from tensorflow.keras.models import load_model
app = Flask(__name__)

bgr_idx = {0: (227, 243, 247), 1: (227, 243, 247), 2: (168, 139, 4), 3: (67, 101, 240), 4: (168, 139, 4), 5: (67, 101, 240), 6: (129, 93, 153), 7: (255, 193, 222), 8: (129, 93, 153), 9: (255, 193, 222), 10: (129, 93, 153), 11: (77, 163, 109), 12: (77, 163, 109), 13: (124, 240, 255), 14: (124, 240, 255)}
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('data/input/model_files/haarcascade_frontalface_default.xml')
model = load_model('data/output/model_files/30landmarks_2.keras')

stream=True

def frame_farm():
    while stream:
        ret, frame = cap.read()
        if not ret:
            print("Camera not available for unknown reasons")
            break
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        all_cordinates = []

        for (x,y,w,h) in faces:
            x -= 5
            y -= 5
            w += 10
            h += 10
            face = gray[y:y+h, x:x+w]
            try:
                face = cv2.resize(face, (96,96))
            except Exception as e:
                continue
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

        ret, frame = cv2.imencode('.jpg', frame)
        frame = frame.tobytes()
        freezed = frame

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')

@app.route("/control", methods=['POST'])
def control():
    global cap, stream
    action = request.get_json().get('action')
    print(action)
    if action =='pause':
        print('Pausing')
        stream=False
        cap.release()
    elif action == 'resume':
        stream=True
        cap = cv2.VideoCapture(0)
    return '', 204

@app.route('/video_feed')
def video_feed():
    return Response(frame_farm(), mimetype = 'multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')


if __name__ == "__main__":
    app.run()
