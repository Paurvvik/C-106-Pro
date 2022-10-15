import cv2

img=cv2.VideoCapture("C:/Users/User/Downloads/C-106 Pro/walking.avi")



body_classifier = cv2.CascadeClassifier("haarcascade_fullbody.xml")

while True:
    ret,frame=img.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = body_classifier.detectMultiScale(gray)
    print(faces)

    for (x,y,w,h) in faces :
     cv2.rectangle(frame,(x,y), (x+w,y+h) , (0,0,255),3)

    cv2.imshow('image',frame)
    cv2.waitKey(0)

img.release()
cv2.destroyAllWindows()