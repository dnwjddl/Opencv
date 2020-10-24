import cv2

img = cv2.imread('01.jpg')

print(img)
print(img.shape) #(높이, 너비, 채널)

cv2.rectangle(img, pt1=(259, 89), pt2=(380, 348), color=(255,0,0), thickness =2)

#그림에서 왼쪽 맨 위가 (0,0)임  >(x) 와 v(y) 방향으로 커짐
#pt1=왼쪽 위
#pt2=오른쪽 아래
#color=BGR

cv2.circle(img, center = (320, 220), radius=100, color=(0,0,255), thickness=3)
cropped_img = img[89:348, 259:380]
img_resized = cv2.resize(img, (512, 256)) #가로 512, 세로256

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('img', img) #이미치 창에 띄우기
cv2.imshow('crop', cropped_img)

cv2.imshow('result', img_resized)
cv2.imshow('rgb', img_rgb)
cv2.imshow('gray', img_gray)

cv2.waitKey(0) #아무 키 누를때까지 지워지지 않게 하기