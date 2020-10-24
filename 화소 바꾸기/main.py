import cv2
import numpy as np

#모델 load 하기
net = cv2.dnn.readNetFromTorch('models/instance_norm/udnie.t7')
#opencv에 dnn 사용. 모델을 torch로 부터 읽는다.
#torch로 만들어진 모델 가지고 오기

img = cv2.imread('imgs/04.jpg')

## 전처리
h, w, c = img.shape
img= cv2.resize(img, dsize = (500, int(h/ w*500))) #비율유지하면서 작게 만들기

MEAN_VALUE = [103.939, 116.779, 123.680] #연구원이 사용했던 전처리 방법; 성능좋아짐
blob = cv2.dnn.blobFromImage(img, mean = MEAN_VALUE) #이미지 전처리
# Mean_value(빼주는 연산)
# 딥러닝 모델에 넣기 위한 차원 변형: 4차원 차원변형

#print(img.shape) #(325, 500, 3) #(높이, 너비, 채널)
#print(blob.shape) #(1,3,325,500) #(1, 채널, 높이, 너비) -> 1은 배치사이즈 : 1장씩 학습시킨다는 뜻

#결론 추론하기(Inference)
net.setInput(blob)
output = net.forward() #forward: 추론함
#output은 컴퓨터가 이해할 수 있는 값 -> 우리가 확인할 수 있게 수정해야 됨 :후처리

## 후처리
output1 = output.squeeze().transpose((1,2,0))
#차원을 줄여줌: squeeze 차원 변형을 거꾸로: transpose
#squeeze: (1, 채널, 높이, 너비) -> (채널, 높이, 너비)
#transpose: (채널, 높이, 너비) -> (너비, 높이, 채널)
print(output.shape) #(1,3,328,500)
print(output1.shape) #(328, 500,3)

output1 += MEAN_VALUE
#뺴줬던거 다시 더해주기
output1 = np.clip(output1, 0,255)
output1 = output1.astype('uint8') #이미지의 자료형을 일반 이미지에서 쓰이는 정수형(uint8)로 바꿔주기




cv2.imshow('img', img)
cv2.imshow('result', output1)
cv2.waitKey(0)
