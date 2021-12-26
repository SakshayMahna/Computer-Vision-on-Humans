import cv2

class HaarCascadeDetector:
    def __init__(self, weights):
        self.detector = cv2.CascadeClassifier(weights)

    def detect(self, frame):
        # Reading the image as gray scale image
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Search the co-ordinates of the image
        upperbody = self.detector.detectMultiScale(gray_img, scaleFactor = 1.05, minNeighbors=5)
        # Sorting the faces on basis of area
        upperbody = sorted(upperbody, key=lambda x:x[3]*x[2])

        # Get the coordinates and return
        if len(upperbody) > 0:
            x,y,w,h = upperbody[-1]
            return x, y, w, h

        return None