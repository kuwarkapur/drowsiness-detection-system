import cv2 
from flask import Flask,Response
haar=cv2.CascadeClassifier('haarcascade_eye.xml')
import tensorflow as tf
model=tf.keras.models.load_model('drowsiness.h5',compile=False)
from pygame import mixer
mixer.init()
sound=mixer.Sound('alarm.wav')
app=Flask(__name__)

@app.route('/')
def index():
    return Response(frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def frames():
    cam=cv2.VideoCapture(0)
    name=True
    dims=224
    b=1
    score=0
    while(name):
        ret, img = cam.read()
        #gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = haar.detectMultiScale(img, 1.3, 5)
        for face in faces:
                x, y, w, h = face
                offset = 5
                face_offset = img[y - offset:y + h + offset, x - offset:x + w + offset]
                face_selection = cv2.resize(face_offset, (500, 500))
                resized_image = cv2.resize(face_selection,(dims,dims),interpolation=cv2.INTER_AREA)
#resized_image=resized_image/resized_image.max()
#resized_image=tf.cast(tf.constant(resized_image),dtype=tf.float32) 
                resized_image=tf.expand_dims(resized_image,axis=0)
                c=model.predict(resized_image)
                b=tf.round(c)
        if b==1:
            print('eyes_open')
            cv2.putText(img,"eyes_open",(100,100),4,1,250,4)
        else:
            print('eyes_closed')
            cv2.putText(img,"eyes_closed",(100,100),4,1,250,4)
            score=score+1
            if score>15:
                try:
                    sound.play()
                except:
                    pass

        for  (x, y, w, h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                
        
        
        ret, buffer = cv2.imencode('.jpg', img)
        img = buffer.tobytes()
        yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

if '__main__'==__name__:
    app.run(debug=True)

