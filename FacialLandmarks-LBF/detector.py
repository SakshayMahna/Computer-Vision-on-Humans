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

class FacialLandmarkLBF:
    def __init__(self, weights):
        self.detector = cv2.face.createFacemarkLBF()
        self.detector.loadModel(weights)

    def detect(self, frame, face):
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, landmarks = self.detector.fit(gray_img, face)

        return landmarks[-1]

class DelaunayTriangulation:
    def __init__(self, height, width):
        self.rect = (0, 0, width, height)

    def get_delaunay(self, points):
        subdiv = cv2.Subdiv2D(self.rect)
        for p in points:
            subdiv.insert(p)

        triangle_list = subdiv.getTriangleList()
        triangle_points = []
        for t in triangle_list:
            pt1 = (int(t[0]), int(t[1]))
            pt2 = (int(t[2]), int(t[3]))
            pt3 = (int(t[4]), int(t[5]))

            pts = (pt1, pt2, pt3)
            triangle_points.append(pts)

        return triangle_points

            