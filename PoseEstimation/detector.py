import cv2

class PoseEstimator:
    def __init__(self, model_file, weights_file, width, height):
        self.net = cv2.dnn.readNetFromCaffe(model_file, weights_file)
        self.width = width
        self.height = height

        self.pose_pairs = [
            (0, 1),
            (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
            (1, 14),
            (14, 8), (8, 9), (9, 10), (14, 11), (11, 12), (12, 13)
        ]
        
        self.hand_pairs = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]

    def detect(self, frame, maps, threshold = 0.5):
        # Reading the image for input to network
        input_image = cv2.dnn.blobFromImage(
            frame, 1.0 / 255, (368, 368), (0, 0, 0), swapRB = False, crop = False
        )
        self.net.setInput(input_image)

        # Calculate predictions
        output = self.net.forward()

        # Return predictions
        height_o = output.shape[2]
        width_o = output.shape[3]
        points = []

        for i in range(maps):
            # Confidence map of corresponding body part
            confidence_map = output[0, i, :, :]

            # Find global maxima of the confidence map
            min_value, probability, min_location, point = cv2.minMaxLoc(confidence_map)

            # Scale the point to fit on original image
            x = (self.width * point[0]) / width_o
            y = (self.height * point[1]) / height_o

            if probability > threshold:
                points.append((int(x), int(y)))
            else:
                points.append(None)

        return points