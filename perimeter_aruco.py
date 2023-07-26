import cv2
import numpy as np
class Marker:
    def __init__(self, id, corners):
        self.id = id
        self.corners = corners
def detectArucoFace(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initialize the detector parameters using default values
    # Define the dictionary
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)

    # Detect the markers
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, dictionary)

    for i, corner in enumerate(corners):
        # corners[i] is a list of four points (which are each a list of two coordinates)
        # so we reshape it into a 4x2 numpy array for easier manipulation
        reshaped_corners = np.reshape(corner, (4, 2))

        perimeter = 0
        for j in range(4):
            perimeter += np.linalg.norm(reshaped_corners[j] - reshaped_corners[(j + 1) % 4])
        cX = int(np.mean(reshaped_corners[:, 0]))
        cY = int(np.mean(reshaped_corners[:, 1]))

        # Draw perimeter value near the marker
        cv2.putText(img, str(round(perimeter, 2)), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print(f"Perimeter of marker with id {ids[i][0]}: {perimeter}")
    return img
