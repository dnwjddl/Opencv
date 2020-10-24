#별론데?

import cv2
import cvlib as cv
import numpy as np

image_path = 'test.jpg'
im = cv2.imread(image_path)

#detect faces
faces, confidences = cv.detect_face(im)

for face in faces:
    (startX, startY) = face[0],face[1]
    (endX, endY) = face[2], face[3]
    #draw rectangle over face
    cv2.rectangle(im, (startX, startY),(endX, endY), (0,255,0),2)
    face_crop = np.copy(im[startY:endY, startX:endX])

    #gender detection
    (label, confidences) = cv.detect_gender(face_crop)

    print(confidences)
    print(label)

    idx = np.argmax(confidences)
    label = label[idx]

    label = "{}:{:2f}%".format(label, confidences[idx] *100)

    Y = startY -10 if startY -10 >10 else startY + 10

    cv2.putText(im, label,(startX, Y), cv2. FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
    #박스 위에 남자인지 여자인지 라벨과 확률 씀

cv2.imwrite('resultGender.jpg',im)
