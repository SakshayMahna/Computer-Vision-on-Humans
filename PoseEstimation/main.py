from camera import CameraThread
from detector import PoseEstimator
from threading import Event
import cv2

width = 640
height = 480


if __name__ == "__main__":
    print("Modes of Pose Detection")
    print("Type 1 for Full Body Detection")
    print("Type 2 for Hand Detection")
    mode = int(input("Enter Mode: "))

    if mode == 1:
        proto_file = "models/pose/pose_deploy_linevec_faster_4_stages.prototxt"
        weights_file = "models/pose/pose_iter_160000.caffemodel"
        maps = 15
    elif mode == 2:
        proto_file = "models/hand/pose_deploy.prototxt"
        weights_file = "models/hand/pose_iter_102000.caffemodel"
        maps = 21
    else:
        print("Choose a correct mode")
        exit()

    detector = PoseEstimator(proto_file, weights_file, width, height)

    kill_event = Event()
    cam = CameraThread(kill_event, height = height, width = width)
    cam.start()

    threshold = 8
    rate = 2
    mid_x = width // 2

    while True:
        frame = cam.read()
        try:
            points = detector.detect(frame, maps)
            for i, (x, y) in enumerate(points):
                frame = cv2.circle(frame, (int(x), int(y)), 10, (0, 255, 255), 
                                   thickness = -1, lineType = cv2.FILLED)
                # frame = cv2.putText(frame, "{}".format(i), (int(x), int(y)), 
                #                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3,
                #                     lineType = cv2.LINE_AA)
                
            if mode == 1:
                for pair in detector.pose_pairs:
                    if points[pair[0]] and points[pair[1]]:
                        frame = cv2.line(frame, points[pair[0]], points[pair[1]], (0, 255, 0), 3)
            elif mode == 2:
                for pair in detector.hand_pairs:
                    if points[pair[0]] and points[pair[1]]:
                        frame = cv2.line(frame, points[pair[0]], points[pair[1]], (0, 255, 0), 3)
        except TypeError:
            pass

        cv2.imshow('webcam', frame)
        if cv2.waitKey(1) == 27:
            break

    kill_event.set()
    cv2.destroyAllWindows()
