import cv2

img = cv2.imread('01.jpg')
overlay_img = cv2.imread('dices.png', cv2.IMREAD_UNCHANGED)
#png는 투명도를 같이 load 를 할라고 함, cv2.IMREAD_UNCHANGED 를 해줘야 투명도가 유지한 상태로 load
#dices 는 png 파일로 배경이 아예 없어야됨(배경: 투명)

overlay_img = cv2.resize(overlay_img, dsize=(150,150))

#반투명, 완전 투명한건 a channel 이 있음 ( 4개의 channel 존재)
print(overlay_img.shape) #alpha channel은 투명하면 0. 불투명하면 255의 색을 가짐

#합성하기 (0아니면1로, 투명성 체크)
overlay_alpha = overlay_img[:,:,3:]/255.0 #0이 배경, 1이 주사위 #주사위
background_alpha = 1.0-overlay_alpha #1이 배경, 0이 주사위 #배경

#x1, y1은 처음 시작하는 좌표
x1 = 100
y1 = 100
#resize 해줬으니까 끝나는 좌표
x2 = x1 + 150
y2 = y1 + 150

#합성하기 (0에서 색들을 합치면 하나 사진의 색만 보이게 합성 가능)
img[y1:y2, x1:x2] = overlay_alpha * overlay_img[:, :, :3] + background_alpha * img[y1:y2, x1:x2]

cv2.imshow('img', img)
cv2.waitKey(0)