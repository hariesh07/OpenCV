import cv2


def SampleFace():
    cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        f = cascade_face.detectMultiScale(g, 1.3, 4)

        for (x, y, w, h) in f:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 4)

        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def EyeDetect():
    cascade_face = cv2.CascadeClassifier('haarcascade_eye.xml')
    cap = cv2.VideoCapture(0)

    while True:
        ret, img = cap.read()
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        f = cascade_face.detectMultiScale(g, 1.3, 4)

        for (x, y, w, h) in f:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 4)

        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xff

        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


print("Welcome to Face and Eye detector tool!!")
while True:
    print("1.Face\t2.Eye")
    a = int(input("Enter a number: "))
    if a == 1:
        SampleFace()
    elif a == 2:
        EyeDetect()
    else:
        print("Enter a valid option!!")
        break
