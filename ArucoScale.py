"""
https://machinelearningknowledge.ai
/augmented-reality-using-aruco-marker-detection-with-python-opencv/
"""
from tabulate import tabulate
import cv2
import projectFunc as pf
import cv2.aruco as aruco
import numpy as np
import os
import pandas as pd

def perspectiveCorrection(height,width,refPoints,imgCam):
    """
    modified Dec 10,2022
    issue: detecting more ponints that expected from cam frame
    :param imgRef: reference image from inkscape drwing
    :param imgCam: camera frame
    :return: image with perspective correction or None when no match.
    """
    impersp=imgCam
    imHeight,imWidth,ch=imgCam.shape
    imgPoints=getArucoCorners(imgCam)
    # print(type(imgPoints))
    if isinstance(imgPoints, np.ndarray):
        if len(refPoints)==len(imgPoints):
            # logger.debug("no of image points ({}) no of ref points({})".format(len(imgPoints), len(refPoints)))
            h, mask = cv2.findHomography(imgPoints, refPoints, cv2.RANSAC, 5.0)
            impersp = cv2.warpPerspective(imgCam, h, (width, height))
        else:
            y=int(height*9/10)
            cv2.putText(imgCam,"Err:PC", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            impersp=imgCam
    else:
        y = int(height * 9 / 10)
        cv2.putText(imgCam, "Err:PC", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        impersp = imgCam

    # return imgCam
    return impersp
def getPixelHeightInMM(pixelRow,img):
    """
    modified on 17 Dec2022, returns Height in MM
    :param img: image with 0 to 9 aruco markers
    :param markerSize:
    :param totalMarkers:
    :return: height in MM
    """
    # logger.debug("function called")
    # pf.showImage(pf.resize(img,.4))
    markerSize=4
    totalMarkers=1000
    draw=False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.getPredefinedDictionary(key)
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
        # logger.info("pixel row above max height")
    elif pixelRow >= xmax:
        height = 0
        # logger.info("pixel row below min height")

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
    return int(height)

def getArRowsDF(img, markerSize = 4, totalMarkers=1000):
    """
    modified on 7 Dec2022, returns horizontal rows of
    arucocorners
    :param img: image with 0 to 9 aruco markers
    :param markerSize:
    :param totalMarkers:
    :return: dataframe
    """
    # logger.debug('function called')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(gray, arucoDict, parameters = arucoParam)
    if ids is None:return 0
    id_list = [x[0] for x in ids]
    # logger.debug('No of markers (in getArRowsDF) detected ({})\n marker ids{}'.format(len(id_list),id_list))
    print("len ids({})".format(len(ids)))
    if len(ids)==10:
        # logger.debug("ALL MARKERS DETECTED")
        markedRowList=[]     #list of rows passing through the aruco corners
        for x in bboxs:
            markedRowList.append(int((x[0,2,1]+x[0,3,1])/2))
            markedRowList.append(int((x[0,0,1]+x[0,1,1])/2))
        markedRowList.sort()
        heightInMM=[x for x in reversed(range(0,1000,50))]
        df=pd.DataFrame({'pixelRows':markedRowList,
                         'heightInMM':heightInMM })
        # df.to_csv(inputPath.csvHeight_PixelRows,index=False)
    else:
        # logger.warning("ALL MARKERS NOT DETECTED")
        df=None

    return df

def getHeightInMicrons(pixelRow,row_height_list):
    """
    created on 08 December 2022
    :param pixelRow: row value of the pixel whose height is to be measured
    :param row_height_list: [[row height], [row height]...]
    :return: real world height in microns
    """
    # logger.debug("function called")
    xmin=row_height_list[0][0]
    xmax=row_height_list[-1][0]
    if pixelRow<xmin:
        height=1001
        # logger.info("pixel row above max height")
    elif pixelRow>=xmax:
        height=0
        # logger.info("pixel row below min height")

    else:
        for i,row_height in enumerate(row_height_list):
            # print(row_height)
            if pixelRow<=row_height[0]:
                x2=row_height[0]
                y2=row_height[1]
                x1=row_height_list[i-1][0]
                y1=row_height_list[i-1][1]
                print(x1,y1,x2,y2)
                height= y1 + ((y2-y1)/(x2-x1))*(pixelRow-x1)
                break
    # for height
    return int(height*1000)
def getHeightInMM(y,img):
    df_row=getArRowsDF(img)
    # logger.debug('pxl-height({}),img.shape({})'.format(y,img.shape))

    if df_row is not None:
        height=getHeightInMicrons(y,df_row.values.tolist())
    else:return 0
    return int(height/1000)
def get40ArucoCorners(img, markerSize = 4, totalMarkers=1000, draw=True):
    # logger.debug('function called')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(gray, arucoDict, parameters = arucoParam)
    bboxs1=np.concatenate(bboxs,axis=1)
    ccoord=np.copy(bboxs1)
    return ccoord[0]
def getArucoCorners(img, markerSize = 4, totalMarkers=1000, draw=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    bboxs, ids, rejected = aruco.detectMarkers(gray, arucoDict)
    # logger.debug('marker ids {}'.format(ids))
    # print(ids)
    ccoord =[0]
    if ids is not None:
        # if len(ids!=10):
            # logger.debug("marker detection error no of markers={}".format(len(ids)))
        # if draw:
        #     aruco.drawDetectedMarkers(img, bboxs)
        bboxs1=np.concatenate(bboxs,axis=1)
        ccoord=np.copy(bboxs1)
    return ccoord[0]
def imMarkers(img, markerSize = 4, totalMarkers=1000, draw=True):
    # logger.debug('function called')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.getPredefinedDictionary(key)
    bboxs, ids, rejected = aruco.detectMarkers(gray, arucoDict, )
    # logger.debug('marker ids {}'.format(ids))
    # if len(ids!=10):
        # logger.error("marker detection error no of markers={}".format(len(ids)))
    # if draw:
    #     aruco.drawDetectedMarkers(img, bboxs)
    bboxs1=np.concatenate(bboxs,axis=1)
    ccoord=np.copy(bboxs1)
    return ccoord[0],img

def camLauncher():
    width = 1280
    height = 720
    url = "http:192.168.0.29:8080/video"
    cam = cv2.VideoCapture(url, cv2.CAP_DSHOW)
    # cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cam.set(cv2.CAP_PROP_FPS, 15)
    # cam.set(cv2.CAP_PROP_FPS, 30)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    return cam

def main():
    """
    working
    :return:
    """
    # cam = cv2.VideoCapture(url)
    frame_counter=0
    # cam=camLauncher()
    # url = "http:192.168.0.29:8080/video"
    url = "http:192.168.0.29:8080/video"
    cam = cv2.VideoCapture(url)
    # cam = camLauncher()
    while 1:
        ret, img = cam.read()
        if not ret:
            print("no frame")
            continue
        print('pixel({}), Height[{}]'.format(200,getPixelHeightInMM(200,img)))

        cv2.imshow('img', img)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
           break
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    # testipcam()