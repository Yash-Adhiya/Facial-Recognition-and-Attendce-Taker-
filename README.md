# Facial-Recognition-and-Attendance-Taker-

Objective: Deetect the face and recognized the name associated with it by comparing facial encoding distance between the webcam image and the all images stroed in directry. 

### Installing and Importing the Required Libraries ### 

```python 
!pip install facial_recognition
!pip install opencv-python
import facial_recognition 
import cv2
```

### steps to perform the face detection ###

1. First we need to convert BGR to RGB format.
2. Then, we need to extract a face from an Image
3. Encoding of a face 
4. comparision of Images using Linear SVM model (runs on backend of the imported package)

```python
#covert BGR to RGB
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
```
```python
#encoding of a face
encode = face_recognition.face_encodings(img)
```

