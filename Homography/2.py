from contextlib import suppress
import sys
from turtle import shape
import numpy as np
import cv2 as cv
import mouse_click_example
import math

if __name__ == '__main__':
    img = cv.imread(sys.argv[1])
    points1 = mouse_click_example.mouth_click(sys.argv[1])
    
    if len(points1) < 4:
        print('pls select four point from Left to Right, top to buttom')
        sys.exit(1)

    Rows = img.shape[0] # rows / height / 1478
    Cols = img.shape[1] # cols / width / 1108

    pointLeftUp = [0, 0]
    pointRightUp = [Cols-1, 0]
    pointLeftDown = [0, Rows-1]
    pointRightDown = [Cols-1, Rows-1]
    points2 = np.array([pointLeftUp, pointRightUp, pointLeftDown, pointRightDown])

    # select input point
    selectPointNum = 4
    arMatrixA = np.empty((selectPointNum*2, 9))

    for num in range(0, selectPointNum):
        pointU = points1[num][0]
        pointV = points1[num][1]
        pointUPrime = points2[num][0]
        pointVPrime = points2[num][1]
        arMatrixA[num*2] = [0, 0, 0, -1*pointU, -1*pointV, -1, pointVPrime*pointU, pointVPrime*pointV, pointVPrime]
        arMatrixA[(num*2)+1] = [pointU, pointV, 1, 0, 0, 0, -1*pointUPrime*pointU, -1*pointUPrime*pointV, -1*pointUPrime]

    # svd
    u, s, vT = np.linalg.svd(arMatrixA, full_matrices=True)
    # select the last column and reshape
    arMatrixH = vT[len(vT)-1]
    arMatrixH = arMatrixH.reshape(3, 3)
    arMatrixHInv = np.linalg.inv(arMatrixH)

    # build map
    arPoint=np.empty((3, 1))
    arCorNew2Ori=np.empty((Rows, Cols, 2))
    for row in range(0, Rows): #1478
        for column in range(0, Cols): #1108 
            arPoint = arMatrixHInv.dot(np.array([[column], [row], [1]]))
            arPoint = arPoint/arPoint[2]
            oriCol = arPoint[0][0]
            oriRow = arPoint[1][0]

            if (oriCol<0) or (oriRow<0) or (oriCol>Cols-1) or (oriRow>Rows-1):
                continue
            
            arCorNew2Ori[row, column] = np.array([oriCol, oriRow])

    # output
    outputimage=np.empty((Rows, Cols, 3))
    for row in range(0, Rows): #1478
        for col in range(0, Cols): #1108
            # get old coordinate
            colCor = arCorNew2Ori[row, col, 0]
            rowCor = arCorNew2Ori[row, col, 1]

            colDown = int(math.floor(colCor))
            colUp = int(math.ceil(colCor))
            rowDown = int(math.floor(rowCor))
            rowUp = int(math.ceil(rowCor))

            colDis = colCor - colDown
            rowDis = rowCor - rowDown
            
            # leftdown rightdown leftup rightup
            outputimage[row, col] = np.array((1-colDis)*(1-rowDis)*img[rowDown, colDown] + colDis*(1-rowDis)*img[rowDown, colUp] + rowDis*(1-colDis)*img[rowUp, colDown] + colDis*rowDis*img[rowUp, colUp])

    cv.imwrite("images/2-1.jpg", outputimage)