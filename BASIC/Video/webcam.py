import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if ret == False:
        break

    frame = frame[183:465, 721:878]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('result', frame)

    if cv2.waitKey(100) == ord('q'):
        break