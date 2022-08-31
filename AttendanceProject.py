import cv2
import face_recognition
import numpy as np
from datetime import datetime


import os
path = 'ImagesAttendance'
images = []
classNames = []
mylist = os.listdir(path)
print(mylist)

# extracting names

for cl in mylist:
    currImg = cv2.imread(f'{path}/{cl}')
    images.append(currImg)
    classNames.append(os.path.splitext(cl)[0])

print(classNames)

# now we can do encoding to all images in the folder.

def findEncoding(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncoding(images)
print("No of encoded Images: ",len(encodeListKnown))
print('encoding complete')


# Taking the images from the webcam
cap = cv2.VideoCapture(0)

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList =  f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dstring}')


while True:
    sucess, img = cap.read()
    #resizing the input images to speed up the process
    imgS = cv2.resize(img, (0,0),None,0.25,0.25) # scaled to 1/4th
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    FaceCurrFrame = face_recognition.face_locations(imgS)
    encodesCurrFrame = face_recognition.face_encodings(imgS,FaceCurrFrame)

    for encodeFace,faceloc in zip(encodesCurrFrame,FaceCurrFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceloc
            y1, x2, y2, x1 = 4*y1,4*x2,4*y2,4*x1
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-250),(x2,y1),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y1-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name) # marking the attendance and time when the face is detected.

    cv2.imshow('webcam',img)
    #cv2.rectangle(img, (FaceCurrFrame[3], FaceCurrFrame[0]), (FaceCurrFrame[1], FaceCurrFrame[2]), (255, 0, 255), 2)
    cv2.waitKey(1)

