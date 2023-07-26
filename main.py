import cv2
import numpy as np
import requests
import mediapipe as mp
import projectFunc as pf
import cv2.aruco as aruco
import pandas as pd
import pytesseract
import ArucoScale
import detect_text as dt
import perimeter_aruco as pa
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path


def get_frame(url):
    # Using video capture with URL
    cap = cv2.VideoCapture(url)
    ret, frame = cap.read()
    if not ret:
        print("Error getting frame")
        return None
    return frame


def perspectiveCorrection(src, imgPoints, refPoints):
    imgPoint2f = np.float32(imgPoints).reshape(-1, 1, 2)
    refPoint2f = np.float32(refPoints).reshape(-1, 1, 2)

    homography, _ = cv2.findHomography(imgPoint2f, refPoint2f)
    dst = cv2.warpPerspective(src, homography, (src.shape[1], src.shape[0]))

    return dst

class Marker:
    def __init__(self, id, corners):
        self.id = id
        self.corners = corners
def bboxFace(image):
    """
    modified on Dec 10, 2022 for pesentation
    :param image:
    :return:
    """
    # mp_face_detection = mp.solutions.face_mesh
    mp_face_detection = mp.solutions.face_detection#original
    mp_drawing = mp.solutions.drawing_utils
    # with mp_face_detection.FaceD(
    with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=.5) as face_detection:
        result = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not result.detections:
            # print("NO Face Detected in 'bboxFace'")
            xmin2 = None
            ymin2 = None
            width = None
            height = None
        else:
            imHeight,imWidth,ch=image.shape
            bbox = result.detections[0].location_data.relative_bounding_box
            xmin = int(bbox.xmin*imWidth)
            ymin = int(bbox.ymin*imHeight)
            width = bbox.width
            height = bbox.height
            height=int(height*imHeight)
            width=int(width*imWidth)
            left_bdr=xmin
            top_bdr=ymin
            right_bdr=imWidth-xmin-width
            bottom_bdr=imHeight-ymin-height
            maxbdr=min(left_bdr,right_bdr,top_bdr,bottom_bdr)
            if maxbdr> width/2:
                bdr=width/2
            elif maxbdr>width/3:
                bdr=width/3
            else:bdr=0
            xmin2=int((xmin-bdr))
            ymin2=int((ymin-bdr))
            width = int(width+2*bdr)
            height = int(height+2*bdr)
        return [xmin2,ymin2,width,height]
def getPixelHeightInMM(pixelRow,img):
    """
    modified on 17 Dec2022, returns Height in MM
    :param img: image with 0 to 9 aruco markers
    :param markerSize:
    :param totalMarkers:
    :return: height in MM
    """
    # pf.showImage(pf.resize(img,.4))
    markerSize=4
    totalMarkers=1000
    draw=False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.getPredefinedDictionary(key)
    # arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(gray, arucoDict)
    # print('len ids-{}'.format(len(ids)))
    # for i in range(10):
    #     if [i] not in ids:print(i)

    # print('ids- \n {}'.format(ids))
    markedRowList = []  # list of rows passing through the aruco corners
    if ids is None:return 0
    if len(ids)!=10:return 0
    for x in bboxs:
        markedRowList.append(int((x[0, 2, 1] + x[0, 3, 1]) / 2))
        markedRowList.append(int((x[0, 0, 1] + x[0, 1, 1]) / 2))
    markedRowList.sort()
    heightInMM = [x for x in reversed(range(0, 1000, 50))]
    df = pd.DataFrame({'pixelRows': markedRowList,
                       'heightInMM': heightInMM})

    row_height_list=df.values.tolist()
    xmin = row_height_list[0][0]
    xmax = row_height_list[-1][0]
    if pixelRow < xmin:
        height = 1001
    elif pixelRow >= xmax:
        height = 0

    else:
        for i, row_height in enumerate(row_height_list):
            # print(row_height)
            if pixelRow <= row_height[0]:
                x2 = row_height[0]
                y2 = row_height[1]
                x1 = row_height_list[i - 1][0]
                y1 = row_height_list[i - 1][1]
                print(x1, y1, x2, y2)
                height = y1 + ((y2 - y1) / (x2 - x1)) * (pixelRow - x1)
                break
    # for height
    height+=900
    return int(height)
def mainHSV(img):
    # img=cv2.imread(inputPath.imgPersp)
    # getArData(img)
    # img=cv2.imread(inputPath.imgLiqCrop)
    corners=ArucoScale.getArucoCorners(img)
    if not isinstance(corners,np.ndarray): return 0,img
    xpoints=[x[0] for x in corners]
    ypoints=[x[1] for x in corners]
    arWidth=1.5*(-corners[0,0]+corners[1,0])
    crop_x2 = int(max(xpoints) + arWidth)
    crop_y2 = int(max(ypoints) + arWidth)
    crop_x1 = int(min(xpoints) - arWidth)
    crop_y1 = int(min(ypoints) - arWidth)
    pf.drawCircles(img,crop_x1,crop_y1,color=1)
    pf.drawCircles(img,crop_x2,crop_y2,color=1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # hsv(312, 98 %, 94 %)
    lower_bound = np.array([140,50, 10])
    upper_bound = np.array([350, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    kernel = np.ones((7, 7), np.uint8)
    # Remove unnecessary noise from mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Segment only the detected region
    segmented_img = cv2.bitwise_and(img, img, mask=mask)
    contours, hierarchy = cv2.findContours(mask.copy(),
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    y=0
    txt="initailization"
    for cnt in contours:
        # print(cv2.isContourConvex(cnt) )
        x, y, w, h = cv2.boundingRect(cnt)
        if h>w*10 and x>crop_x1 and x <crop_x2 and y>crop_y1 and y<crop_y2:
            # cv2.rectangle(segmented_img,(x-10,y),(x+w,y+h),(0,255,0),2)
            # for i in range(50):
            #     arHeight=ArucoScale.getPixelHeightInMM(y,img)
            #     if arHeight!=0: break

            cv2.rectangle(img,(x-10,y),(x+w,y+h),(0,255,0),2)
            arHeight=ArucoScale.getPixelHeightInMM(y,img)
            if arHeight!=0:
                txt="{}[{}MM]".format(y,arHeight)

            cv2.putText(img, txt, (200, y-50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
            img = cv2.line(img, (200, y), (crop_x1, y), (0, 0, 255), 5)
            # pf.writeData(fieldName='liqHeight',dataValue=y,source_module="liquidLevel")


    # Showing the output
    # cv2.imshow("Output", output)
    # cv2.imwrite(inputPath.imgLiqLev,segmented_img)
    # cv2.imwrite(inputPath.imgLiqCrop,img)
    # # cv2.imwrite(inputPath.imgLiqLev,segmented_img)
    # # segmented_img=pf.resize(segmented_img,.2)
    # # cv2.imshow("segmentedHSV", segmented_img)
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()
    return img
def detect_numbers(frame):
    """
    This function takes an image frame, converts it to grayscale and uses pytesseract to extract digits from the frame.
    :param frame: input image
    :return: extracted digits from the frame
    """

    # Convert the image to grayscale (this can improve OCR results)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    # # # Convert grayscale image to black and white
    # # Threshold the image
    # _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    #
    # # Find contours in the image
    # contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Use pytesseract to extract text from the image
    # for contour in contours:
    #     # Get the bounding rectangle
    #     x, y, w, h = cv2.boundingRect(contour)
    #
    #     # Extract the region of interest
    #     roi = binary[y:y + h, x:x + w]
    #
    #     # Use pytesseract to extract text
    #     # We add a new configuration option: '-c tessedit_char_whitelist=0123456789'
    #     # This tells Tesseract to only recognize digits
    #     text = pytesseract.image_to_string(roi, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
    #
    #     # Print the extracted text
    #     print(text)
    text = dt.ocr_image(frame)
    print(text)

def heightInPixel(imageWithPerspectiveCorrection):
    annotated_image = imageWithPerspectiveCorrection.copy()
    print("heightInPixel called")
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    heightInMM=0
    # height = []
    with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        image = imageWithPerspectiveCorrection
        h, w, ch = image.shape
        xmin, ymin, width, height = bboxFace(image)  # ensure bboxFace function is defined
        if ((xmin is not None) and (ymin is not None)):
            print('Face detected, Image cropped')
            crop_face = image[ymin:ymin + height, xmin:xmin + width]
            crop_face2 = crop_face.copy()
            results = face_mesh.process(cv2.cvtColor(crop_face, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                annotated_image = image.copy()
                landmarks = results.multi_face_landmarks[0].landmark
                # define the coordinates here...
                x175 = int(landmarks[175].x * width + xmin)
                y175 = int(landmarks[175].y * height + ymin)
                x152 = int(landmarks[152].x * width + xmin)
                y152 = int(landmarks[152].y * height + ymin)
                x33 = int(landmarks[33].x * width + xmin)
                y33 = int(landmarks[33].y * height + ymin)
                x133 = int(landmarks[133].x * width + xmin)
                y133 = int(landmarks[133].y * height + ymin)
                x362 = int(landmarks[362].x * width + xmin)
                y362 = int(landmarks[362].y * height + ymin)
                x263 = int(landmarks[263].x * width + xmin)
                y263 = int(landmarks[263].y * height + ymin)
                # ----------------------------------
                y127 = int(landmarks[127].y * height + ymin)
                y356 = int(landmarks[356].y * height + ymin)
                y162 = int(landmarks[162].y * height + ymin)
                y389 = int(landmarks[389].y * height + ymin)
                x389 = int(landmarks[389].x * width + xmin)
                x162 = int(landmarks[162].x * width + xmin)
                x356 = int(landmarks[356].x * width + xmin)
                x127 = int(landmarks[127].x * width + xmin)
                # ----------------------------------
                yEye = int((y127 + y356 + y162 + y389) / 4)
                # yEye = int((y33 + y133 + y362 + y263) / 4)
                yChin = y175
                yVertex = 2 * yEye - yChin
                xVertex = x175
                print("heightInPixel(height pixel={}) data written to 'dataFile.csv'".format(yVertex))
                # define circles here...
                cv2.circle(annotated_image, (x175, y175), 5, (0, 255, 0), 1)
                cv2.circle(annotated_image, (x152, y152), 5, (0, 255, 0), 1)
                cv2.circle(annotated_image, (x33, y33), 5, (0, 255, 0), 1)
                cv2.circle(annotated_image, (x133, y133), 5, (0, 255, 0), 1)
                cv2.circle(annotated_image, (x362, y362), 5, (0, 255, 0), 1)
                cv2.circle(annotated_image, (x263, y263), 5, (0, 255, 0), 1)
                # ------------
                cv2.circle(annotated_image, (x127, y127), 5, (0, 255, 0), 1)
                cv2.circle(annotated_image, (x162, y162), 5, (0, 255, 0), 1)
                cv2.circle(annotated_image, (x356, y356), 5, (0, 255, 0), 1)
                cv2.circle(annotated_image, (x389, y389), 5, (0, 255, 0), 1)

                # ---------
                cv2.circle(annotated_image, (xVertex, yVertex), 2, (0, 0, 255), 2)
                # pf.showImage(annotated_image)
                annotated_image = cv2.line(annotated_image, (0, yVertex),
                                           (w, yVertex), (0, 0, 255), 5)
                # if(len(height)==10):
                #     height.sort()
                #     height_final=0
                #     for i in range(2,8):
                #         height_final
                heightInMM = getPixelHeightInMM(yVertex, imageWithPerspectiveCorrection)  # ensure getPixelHeightInMM function is defined
                cv2.putText(annotated_image, 'vertex({})[{} mm]'.format(yVertex, heightInMM), (20, yVertex), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                print("face cropped but face mesh not detected")
                yVertex = 0
                annotated_image = imageWithPerspectiveCorrection
        else:
            print("face not found")
            yVertex = 0
            annotated_image = imageWithPerspectiveCorrection
    return annotated_image

def detectArucoFace(mat):
    height, width, _ = mat.shape
    xaxis=width/2
    print("Entered detectArucoFace")

    if mat.size == 0:
        print("Frame is empty")
        return None

    gray = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100))

    markers = [Marker(ids[i][0], corners[i]) for i in range(len(corners))]

    markers.sort(key=lambda marker: marker.corners[0][0][1])  # Sort markers by y-coordinate of top-left corner

    sortedCorners = [marker.corners for marker in markers]
    sortedIds = [marker.id for marker in markers]

    imgPoints = [corner for sublist in sortedCorners for point in sublist for corner in point]

    refPoints = []

    for id in sortedIds:
        arucosize = int(height / 21)

        if id == 9:
            refPoints += [[xaxis, arucosize], [xaxis + arucosize, arucosize],
                          [xaxis + arucosize, arucosize + arucosize],
                          [xaxis, arucosize + arucosize]]  # top-left, top-right, bottom-right, bottom-left
        if id == 8:
            refPoints += [[xaxis, arucosize + arucosize * 2], [xaxis + arucosize, arucosize + arucosize * 2],
                          [xaxis + arucosize, arucosize + arucosize * 3],
                          [xaxis, arucosize + arucosize * 3]]  # top-left, top-right, bottom-right, bottom-left
        if id == 7:
            refPoints += [[xaxis, arucosize + arucosize * 4], [xaxis + arucosize, arucosize + arucosize * 4],
                          [xaxis + arucosize, arucosize + arucosize * 5],
                          [xaxis, arucosize + arucosize * 5]]  # top-left, top-right, bottom-right, bottom-left
        if id == 6:
            refPoints += [[xaxis, arucosize + arucosize * 6], [xaxis + arucosize, arucosize + arucosize * 6],
                          [xaxis + arucosize, arucosize + arucosize * 7],
                          [xaxis, arucosize + arucosize * 7]]  # top-left, top-right, bottom-right, bottom-left
        if id == 5:
            refPoints += [[xaxis, arucosize + arucosize * 8], [xaxis + arucosize, arucosize + arucosize * 8],
                          [xaxis + arucosize, arucosize + arucosize * 9],
                          [xaxis, arucosize + arucosize * 9]]  # top-left, top-right, bottom-right, bottom-left
        if id == 4:
            refPoints += [[xaxis, arucosize + arucosize * 10], [xaxis + arucosize, arucosize + arucosize * 10],
                          [xaxis + arucosize, arucosize + arucosize * 11],
                          [xaxis, arucosize + arucosize * 11]]  # top-left, top-right, bottom-right, bottom-left
        if id == 3:
            refPoints += [[xaxis, arucosize + arucosize * 12], [xaxis + arucosize, arucosize + arucosize * 12],
                          [xaxis + arucosize, arucosize + arucosize * 13],
                          [xaxis, arucosize + arucosize * 13]]  # top-left, top-right, bottom-right, bottom-left
        if id == 2:
            refPoints += [[xaxis, arucosize + arucosize * 14], [xaxis + arucosize, arucosize + arucosize * 14],
                          [xaxis + arucosize, arucosize + arucosize * 15],
                          [xaxis, arucosize + arucosize * 15]]  # top-left, top-right, bottom-right, bottom-left
        if id == 1:
            refPoints += [[xaxis, arucosize + arucosize * 16], [xaxis + arucosize, arucosize + arucosize * 16],
                          [xaxis + arucosize, arucosize + arucosize * 17],
                          [xaxis, arucosize + arucosize * 17]]  # top-left, top-right, bottom-right, bottom-left
        if id == 0:
            refPoints += [[xaxis, arucosize + arucosize * 18], [xaxis + arucosize, arucosize + arucosize * 18],
                          [xaxis + arucosize, arucosize + arucosize * 19],
                          [xaxis, arucosize + arucosize * 19]]  # top-left, top-right, bottom-right, bottom-left

    print(len(imgPoints))
    print(len(refPoints))
    if len(imgPoints) == len(refPoints) and len(imgPoints) > 8:
        mat = perspectiveCorrection(mat, imgPoints, refPoints)  # You need to implement this function
        mat = heightInPixel(mat)
        mat = mainHSV(mat)
    # Free up memory
    corners = ()

    return mat


def main():
    url = 'http://192.168.29.148:8080/video'  # replace with your url
    while True:
        frame = get_frame(url)
        if frame is None:
            continue
        #
        # # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        # #
        # # for (x, y, w, h) in faces:
        # #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        frame = detectArucoFace(frame)

        # detect_numbers(frame)
        cv2.imshow('IP Webcam - Face Detection', frame)
        # image = "C:/Users/varun/OneDrive/Desktop/led.jpg"




        # Then you can directly call the function ocr_image() on the read image
        # recognized_text = dt.ocr_image(image)
        #
        # print(recognized_text)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
def get_frame_USB(cap):
    # Using video capture with URL
    ret, frame = cap.read()
    if not ret:
        print("Error getting frame")
        return None
    height, width, _ = frame.shape
    print("Resolution: {} x {}".format(width, height))
    return frame

def mainUsbWebCam():
    cap = cv2.VideoCapture(0)  # 0 is generally the default webcam. Change it accordingly if you have multiple webcams


    while True:
        frame = get_frame_USB(cap)
        if frame is None:
            continue

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        #
        # for (x, y, w, h) in faces:
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # frame = detectArucoFace(frame)
        frame=pa.detectArucoFace(frame)
        cv2.imshow('Webcam - Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    mainUsbWebCam()
    # main()
