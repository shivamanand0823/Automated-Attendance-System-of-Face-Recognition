import cv2, numpy as np;
import xlwrite;
import time
import sys
from playsound import playsound
start=time.time()
period=8
face_cas = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0);
recognizer = cv2.face.LBPHFaceRecognizer_create();
recognizer.read('trainer/trainer.yml');
flag = 0;
id=0;
filename='filename';
dict = {
            'item1': 1
}
#font = cv2.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 5, 1, 0, 1, 1)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, img = cap.read();
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
    faces = face_cas.detectMultiScale(gray, 1.3, 7);
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),2);
        id,conf=recognizer.predict(roi_gray)
        if(conf<=85):
                if(id==1):
                    id='Shivam Anand'
                    if((str(id)) not in dict):
                        filename=xlwrite.output('attendance','class1',4,id,'yes');
                        dict[str(id)]=str(id);
                elif(id==2):
                    id = 'Satyam Anand'
                    if ((str(id)) not in dict):
                        filename =xlwrite.output('attendance', 'class1', 8, id, 'yes');
                        dict[str(id)] = str(id);
                elif(id==3):
                    id = 'Manshi Arya'
                    if ((str(id)) not in dict):
                        filename =xlwrite.output('attendance', 'class1', 12, id, 'yes');
                        dict[str(id)] = str(id);
                elif(id==4):
                    id = 'Swati Priya'
                    if ((str(id)) not in dict):
                        filename =xlwrite.output('attendance', 'class1', 16, id, 'yes');
                        dict[str(id)] = str(id);
                elif(id==5):
                    id = 'Madhurjyabora Sir'
                    if ((str(id)) not in dict):
                        filename =xlwrite.output('attendance', 'class1', 20, id, 'yes');
                        dict[str(id)] = str(id);

                else:
                    id = 'Unknown'
                    print('Unknown')
                    break;
        cv2.putText(img,str(id)+" "+str(conf),(x,y-10),font,0.55,(120,255,120),1)
        #cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h),font,(0,0,255));
    cv2.imshow('frame',img);
    if time.time()>start+period:
        break;
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break;

cap.release();
cv2.destroyAllWindows();
