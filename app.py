import cv2 
from flask import Flask,Response
haar=cv2.CascadeClassifier('haarcascade_eye.xml')
lhaar=cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
rhaar=cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
import tensorflow as tf
model=tf.keras.models.load_model('drowsiness.h5',compile=False)
from pygame import mixer
rlbl=[99]
llbl=[99]
mixer.init()
sound=mixer.Sound('alarm.wav')
app=Flask(__name__)

@app.route('/')
def index():
    return Response(frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def frames():
    cam=cv2.VideoCapture(0)
    name=True
    b=0
    score=0
    dims=224
    while(name):
        ret, img = cam.read()
    #gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = haar.detectMultiScale(img, 1.3, 5)
        leye = lhaar.detectMultiScale(img, 1.3, 5)
        reye = rhaar.detectMultiScale(img, 1.3, 5)
        for face in leye:
            x, y, w, h = face
            offset = 5
            face_offset = img[y - offset:y + h + offset, x - offset:x + w + offset]
            face_selection = cv2.resize(face_offset, (500, 500))
            resized_image = cv2.resize(face_selection,(dims,dims),interpolation=cv2.INTER_AREA)
            resized_image=tf.expand_dims(resized_image,axis=0)
            c=model.predict(resized_image)
            b=tf.round(c)
        if b==1:
            llbl[0]=1
        else:
            llbl[0]=0

        for face in reye:
            x, y, w, h = face
            offset = 5
            face_offset = img[y - offset:y + h + offset, x - offset:x + w + offset]
            face_selection = cv2.resize(face_offset, (500, 500))
            resized_image = cv2.resize(face_selection,(dims,dims),interpolation=cv2.INTER_AREA)
            resized_image=tf.expand_dims(resized_image,axis=0)
            c=model.predict(resized_image)
            b=tf.round(c)
        if b==1:
            rlbl[0]=1
           
        else:
            rlbl[0]=0
            pass

        if(rlbl[0]==0 and llbl[0]==0):
            score=score+1
            cv2.putText(img,"closed",(100,100),4,1,250,4,cv2.LINE_AA)
   
        else:
            score=score-1
            cv2.putText(img,"open",(100,100),4,1,250,4,cv2.LINE_AA)
        if(score<0):
            score=0
            cv2.putText(img,'Score:'+str(score),(200,100),4,1,250,4,cv2.LINE_AA)
        if(score>15):
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

