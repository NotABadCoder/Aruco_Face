# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import numpy as np
import pandas as pd

import glob, math
def writeData(fieldName,dataValue,
              source_module,fileName="dataFile.csv"):
    """
    :param fieldName: name of value passed to csv file
    :param dataValue: Value of height
    :param source_module: Name of module
    :param fileName: csv file name from inputPath
    :return: None
    """
    data=pd.read_csv(fileName)
    fieldList=data['field'].to_list()
    if fieldName in fieldList:
        dataRow=data[data['field']==fieldName].index
        print("Value for {} already exists".format(fieldName))
        print("{} ={}".format(fieldName,data.iloc[dataRow[0],1]))
        print("replaceing {} value  {} with {}".format(fieldName,data.iloc[dataRow[0],1],dataValue))
        data.iloc[dataRow[0],1]=dataValue
    else:
         df=pd.DataFrame({'field name':[fieldName],'data value':[dataValue],
                         'source module':[source_module]})
         df.to_csv(fileName,mode="a",index=False,header=False)
    df=pd.read_csv(fileName)
    print("csv data")
    df2table(df)
def drawCircles(img,x,y,r=5,thickness=3,color=0):
    if color==0:ccolor=(255,255,255)
    if color==1:ccolor=(255,0,0)
    if color==2:ccolor=(0,255,0)
    if color==3:ccolor=(0,0,255)
    x=int(x)
    y=int(y)
    img=cv2.circle(img, (x,y), r, ccolor,5)
    return img
from tabulate import tabulate
def df2table(dataframe):
    # displaying the DataFrame
    print(tabulate(dataframe, headers='keys', tablefmt='pretty'))
def color2grey(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
def showImage(img):
    winname='test'
    cv2.namedWindow(winname)
    cv2.moveWindow(winname,800,10)
    cv2.imshow(winname, img)
    cv2.waitKey(0)
# Press the green button in the gutter to run the script.
def grey2binary(imgrey):
    ret, thresh1 = cv2.threshold(imgrey, 80, 255, cv2.THRESH_BINARY)
    return thresh1
    # print_hi('DST BMI 2021')
def resize(img,ratio):
    return cv2.resize(img, (0, 0), fx=ratio, fy=ratio)
def lineImg(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray=grey2binary(gray)
    edges = cv2.Canny(gray, 80, 120)
    lines = cv2.HoughLinesP(edges, 1, 3.14 / 2, 2, None, 30, 1);
    # print(lines)
    for line in lines:
        pt1 = (line[0][0], line[0][1])
        pt2 = (line[0][2], line[0][3])
        cv2.line(img, pt1, pt2, (0, 0, 255), 2)
    return img
def getLines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray=grey2binary(gray)
    edges = cv2.Canny(gray, 80, 120)
    lines = cv2.HoughLinesP(edges, 1, 3.14 / 2, 2, None, 30, 1);
    return lines

def camCalib():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 7, 3), np.float32)
    print("objp")
    print(objp)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
    print("objp2")
    print(objp)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    images = glob.glob('*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
            cv2.destroyAllWindows()

def preProcess(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, im = cv2.threshold(im, 50, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    im = cv2.erode(im, kernel, iterations=1)
    im = cv2.dilate(im, kernel, iterations=1)
    im = cv2.dilate(im, kernel, iterations=1)
    im = cv2.erode(im, kernel, iterations=1)
    # im = imutils.rotate_bound(im, 45)
    # im = cv2.erode(im, kernel, iterations=1)
    # im = cv2.dilate(im, kernel, iterations=1)
    # im = cv2.dilate(im, kernel, iterations=1)
    # im = cv2.erode(im, kernel, iterations=1)
    # im = imutils.rotate_bound(im, -45)
    # showImage(im)
    return im

def calculateGenCamMatrix_coefficients(img):
    #get height and width of image
    h,w=img.shape[:2]
    focal_length = 1 * w
    matrix_coefficients = np.array([[focal_length, 0, h / 2],
                           [0, focal_length, w / 2],
                           [0, 0, 1]])
    distortion_coefficients = np.zeros((4, 1), dtype=np.float64)
    return matrix_coefficients,distortion_coefficients

def getCornerImage(im):
    gray=preProcess(im)
    corners = cv2.goodFeaturesToTrack(gray, 300, 0.01, 10)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.001)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    corners2int = np.int0(corners2)
    corList=corners2int.tolist()
    temp=[]
    for cor in corList:
        # print(cor)
        temp.append(cor[0])
    cordList=temp
    # print(cordList)
    cordSorted=sorted(cordList, key=lambda k: [k[1], k[0]])
    print(cordSorted)
    band=10
    rowList=[]
    while len(cordSorted)>0:
        temprow = []
        for y,x in cordSorted:
            upperBand=cordSorted[0][1]+band
            if x<upperBand:temprow.append([y,x])
        cordSortedtemp=[i for i in cordSorted if i not in temprow]
        cordSorted=cordSortedtemp
        row = sorted(temprow,key=lambda k:[k[0]])
        rowList.append(row)
        print(row)
    # for corner in corners2:
        # print(corner.ravel())
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 0, 0)
    font_size=1
    # Line thickness of 2 px
    thickness = 0
    for row in rowList:
        count = 0
        for x,y in row:
            # x, y = i
            print('x{0}={2} ; y{0}={1}'.format(count,x,y))
            count+=1
            cv2.circle(im, (x, y), 3, (0,0,255),1)
            # cv2.putText(im, str(count), (x,y), font,
            #             font_size, color, thickness, cv2.LINE_AA,False)
    return im,corners2
def plotCorners(im,cornerList):
    for row in cornerList:
        count = 0
        for x,y in row:
            # x, y = i
            # print('x{0}={2} ; y{0}={1}'.format(count,x,y))
            count+=1
            cv2.circle(im, (x, y), 3, (0,0,255),1)
            # cv2.putText(im, str(count), (x,y), font,
            #             font_size, color, thickness, cv2.LINE_AA,False)
    return im

def getDist(point1,point2):
    x1=point1[0]
    x2=point2[0]
    y1=point1[1]
    y2=point2[1]
    x=x2-x1
    y=y2-y1
    return math.sqrt(x*x+y*y)
def getCornerImageFinal(im):
    gray = preProcess(im)
    corners = cv2.goodFeaturesToTrack(gray, 400, 0.01, 10)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.001)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    corners2int = np.int0(corners2)
    corList = corners2int.tolist()
    temp = []
    for cor in corList:
        temp.append(cor[0])
    cordList = temp
    cordSorted = sorted(cordList, key=lambda k: [k[1], k[0]])
    band = 10
    rowList = []
    while len(cordSorted) > 0:
        temprow = []
        for y, x in cordSorted:
            upperBand = cordSorted[0][1] + band
            if x < upperBand: temprow.append([y, x])
        cordSortedtemp = [i for i in cordSorted if i not in temprow]
        cordSorted = cordSortedtemp
        row = sorted(temprow, key=lambda k: [k[0]])
        rowList.append(row)
    '''clean duplicate points'''
    tempRowList=[]
    for row in rowList:
        for coord in row:
            if row.count(coord)>1:
                row.remove(coord)
    for row in rowList:
        for coord in row[1:-1]:
            point=coord
            p_index=row.index(point)
            point1=row[p_index-1]
            point2=row[p_index+1]
            if (getDist(point,point1)<5
                or getDist(point,point2))<5:
                row.remove(point)
    return im, rowList

def getxColmn(rowList,col):
    colList=[]
    band=8
    ymin=col[0]-band
    ymax=col[0]+band
    for row in rowList:
        for y,x in row:
            if y>ymin and y<ymax:
                colList.append([y,x])
    return colList
def testPointInList(point,col_list):
    colFound=False
    # cv2.waitKey(0)
    for col in col_list:
        if point in col:
            # print('3-True')
            colFound= True

    return colFound

def getGrid(rowList):
    colmnList=[]
    for row in rowList:
        for y,x in row:
            # print(colmnList)
            if testPointInList([y,x],colmnList):
                continue
            else:
                colmnList.append(getxColmn(rowList, [y, x]))
    return colmnList


def main():
    im = cv2.imread('chessbHarshit1.jpg')
    gray=color2grey(im)
    imcorners, rowList = getCornerImageFinal(im)
    newRowList=[]
    for row in rowList[5:-7]:
        newRowList.append(row[1:-2])
    print('no of rows {} no of columns {}'.format(
        len(newRowList),len(newRowList[0])))
    for row in newRowList:
        print(row)
    obj_grid=[]
    for x in range(0,100,10):
        temp=[]
        for y in range(0,100,10):
            temp.append([y,x,0])
        obj_grid.append(temp)
    objpoints=np.array(obj_grid, dtype=np.float32)
    imgpoints=np.array(newRowList, dtype=np.float32)
    print(objpoints.dtype)
    print(imgpoints.dtype)
    for row in obj_grid:
        print(row)
    # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    #     objpoints, imgpoints, gray.shape[::-1], None, None)

    # imcorners = plotCorners(im,newRowList)
    # imcorners=resize(imcorners,2)
    # showImage(imcorners)
def testCamCalibration():
    im = cv2.imread('chessbHarshit1.jpg')
    gray=color2grey(im)
    imcorners, rowList = getCornerImageFinal(im)
    newRowList=[]
    for row in rowList[5:-7]:
        newRowList.append(row[1:-2])
        print(row[1:-2])
    print('Image points no of rows {} no of columns {}'.format(
        len(newRowList),len(newRowList[0])))
    imgpoints = np.array(newRowList, dtype=np.float32)
    print(type(imgpoints))
    print(imgpoints)
    obj_grid=[]
    for x in range(0,100,10):
        temp=[]
        for y in range(0,100,10):
            temp.append([y,x,0])
        obj_grid.append(temp)
    objpoints = np.array(obj_grid, dtype=np.float32)
    print(objpoints)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    # imcorners = plotCorners(im,newRowList)
    # imcorners=resize(imcorners,2)
    # showImage(imcorners)
if __name__ == '__main__':
    # main()
    testCamCalibration()