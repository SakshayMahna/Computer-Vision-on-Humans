from camera import CameraThread
from detector import HaarCascadeDetector, FacialLandmarkLBF, DelaunayTriangulation
from threading import Event
import cv2
import numpy as np

width = 640
height = 480


if __name__ == "__main__":
    weights = "weights/haarcascade_frontalface.xml"
    detector = HaarCascadeDetector(weights)

    face_weights = "weights/lbfmodel.yaml"
    face_landmarks = FacialLandmarkLBF(face_weights)

    triangulation = DelaunayTriangulation(height, width)

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
            # frame = cv2.rectangle(frame, (x, y),(x+w, y+h), (0, 255, 0), 2)

            landmark = face_landmarks.detect(frame, np.array([(x, y, w, h)]))
            points = []
            for x, y in landmark[0]:
                frame = cv2.circle(frame, (int(x), int(y)), 1, (255, 255, 255), 1)
                points.append((x, y))

            triangulation_points = triangulation.get_delaunay(points)
            for pt1, pt2, pt3 in triangulation_points:
                frame = cv2.line(frame, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA, 0)
                frame = cv2.line(frame, pt2, pt3, (0, 255, 0), 1, cv2.LINE_AA, 0)
                frame = cv2.line(frame, pt3, pt1, (0, 255, 0), 1, cv2.LINE_AA, 0)

        except TypeError:
            pass

        cv2.imshow('webcam', frame)
        if cv2.waitKey(1) == 27:
            break

    kill_event.set()
    cv2.destroyAllWindows()
