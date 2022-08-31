print("Hello World")

import face_recognition
import cv2
import numpy


#first we require to convert GBR to RGB

imgElon = face_recognition.load_image_file('elonmusktrain.png')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
#


imgTest = face_recognition.load_image_file("elonmusktest.png")
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)


# step 2
# finding the faces in images and encode them

faceLoc = face_recognition.face_locations(imgElon)[0]
encodElon = face_recognition.face_encodings((imgElon))[0]
#print(faceLoc) # --> it will give the four value of the rectangle in a face

cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)


faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

# we will compare the encoding of the faces of the both the images, using linear SVM at the backend. it they are similar
# we can say both images are similar, or not similar.

results = face_recognition.compare_faces(encodeTest,[encodElon])  # this function compare the encoding of the two face encoding and return either true or false
face_distance = face_recognition.face_distance([encodElon],encodeTest)
print(results,face_distance)  # in this case it returns true, since both the images are of elon musk.
cv2.putText(imgTest,f'{results}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,250),2)    

cv2.imshow('Musk Image Trian',imgElon)
cv2.imshow('Musk test Image',imgTest)
cv2.waitKey(0)
