import cv2
import detector

if not __name__ == '__main__':
    exit()
loop = True
sign = detector.Sign()
capture = cv2.VideoCapture(0)

while loop:
    rate, frame = capture.read()
    if rate is not True:    
        continue
    result = sign.ai_detecting(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        loop = False
cv2.destroyAllWindows()
capture.release()