from camera import CameraThread
from detector import HaarCascadeDetector
from threading import Event
import cv2

width = 320
height = 240


if __name__ == "__main__":
    print("Modes of Haar Cascade Detection")
    print("Type 1 for Full Body Detection")
    print("Type 2 for Upper Body Detection")
    print("Type 3 for Frontal Face Detection")
    mode = int(input("Enter Mode: "))

    if mode == 1:
        weights = "weights/haarcascade_fullbody.xml"
    elif mode == 2:
        weights = "weights/haarcascade_upperbody.xml"
    elif mode == 3:
        weights = "weights/haarcascade_frontalface.xml"
    else:
        print("Choose a correct mode")
        exit()

    detector = HaarCascadeDetector(weights)

    kill_event = Event()
    cam = CameraThread(kill_event, height = height, width = width)
    cam.start()

    threshold = 8
    rate = 2
    mid_x = width // 2

    while True:
        frame = cam.read()
        try:
            x, y, w, h = detector.detect(frame)
            frame = cv2.rectangle(frame, (x, y),(x+w, y+h), (0, 255, 0), 2)
        except TypeError:
            pass

        cv2.imshow('webcam', frame)
        if cv2.waitKey(1) == 27:
            break

    kill_event.set()
    cv2.destroyAllWindows()
