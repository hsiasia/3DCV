import sys
from turtle import shape
import numpy as np
import cv2 as cv
from skimage import io, transform          
from shapely.geometry import MultiPoint  
import math

def get_sift_correspondences(img1, img2, rate, filtFlag):
    '''
    Input:
        img1: numpy array of the first image
        img2: numpy array of the second image

    Return:
        points1: numpy array [N, 2], N is the number of correspondences
        points2: numpy array [N, 2], N is the number of correspondences
    '''
    #sift = cv.xfeatures2d.SIFT_create()# opencv-python and opencv-contrib-python version == 3.4.2.16 or enable nonfree
    sift = cv.SIFT_create()             # opencv-python==4.5.1.48
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matcher = cv.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        # [1-0 1-1]: 0.13 / [1-0 1-2]: 0.65
        if m.distance < float(rate) * n.distance:
            if filtFlag:
                if int(kp1[m.queryIdx].pt[0]) != 843 and int(kp1[m.queryIdx].pt[0]) != 730:
                    good_matches.append(m)
            else:
                good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)
    print('all match point: ', len(good_matches))
    points1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
    points2 = np.array([kp2[m.trainIdx].pt for m in good_matches])
    
    img_draw_match = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow('match', img_draw_match)
    cv.waitKey(0)
    return points1, points2


if __name__ == '__main__':
    # python 1-2.py images/1-0.png images/1-1.png groundtruth_correspondences/correspondence_01.npy 0.6 4/8/20
    # python 1-2.py images/1-0.png images/1-2.png groundtruth_correspondences/correspondence_02.npy 0.65 4/8/20
    img1 = cv.imread(sys.argv[1])
    img2 = cv.imread(sys.argv[2])
    filtFlag = False
    mapFeaturePoint2Rate = {'4': 0.09,
                            '8': 0.1,
                            '20': 0.1275,
                            'other': 0.6}
    if sys.argv[2] == 'images/1-2.png':
        filtFlag = True
        mapFeaturePoint2Rate = {'4': 0.54,
                                '8': 0.57,
                                '20': 0.651,
                                'other': 0.7}

    FeaturePoint = sys.argv[4]
    if sys.argv[4] not in ['4', '8', '20']:
        FeaturePoint = 'other'
    rate = mapFeaturePoint2Rate[FeaturePoint]
    gt_correspondences = np.load(sys.argv[3])

    Ps = gt_correspondences[0]
    Pt = gt_correspondences[1]
    # [[ 5.16559729e+02  8.00938840e+02  1.00000000e+00]
    # [ 7.35600119e+02  4.63709569e+02  1.00000000e+00]
    # ...[ui, vi, 1]]
    Ps = np.c_[Ps, np.ones(shape=(100, 1))]
    Pt = np.c_[Pt, np.ones(shape=(100, 1))]

    points1, points2 = get_sift_correspondences(img1, img2, rate, filtFlag)
    pointSize = len(points1)-1

    # select 4, 8, 20 point
    selectPointNum =int(sys.argv[4])
    arMatrixA = np.empty(shape=(selectPointNum*2, 9))

    arPoint = np.zeros(shape=(selectPointNum, 2))
    arPointPrime = np.zeros(shape=(selectPointNum, 2))
    for num in range(0, selectPointNum):
        # idxPoint = random.randrange(pointSize)
        arPoint[num] = points1[num]
        arPointPrime[num] = points2[num]

    # do normalize find new origin
    points = MultiPoint(arPoint)
    pointNewOrigin = np.array([points.centroid.x, points.centroid.y])
    points = MultiPoint(arPointPrime)
    pointPrimeNewOrigin = np.array([points.centroid.x, points.centroid.y])

    # do normalize min length
    arPointNorm = np.zeros(shape=(selectPointNum, 2))
    arPointPrimeNorm = np.zeros(shape=(selectPointNum, 2))
    for num in range(0, selectPointNum):
        arPointNorm[num] = arPoint[num] - pointNewOrigin
        arPointPrimeNorm[num] = arPointPrime[num] - pointPrimeNewOrigin

    # distance on Point
    arNorm = np.linalg.norm((arPoint - pointNewOrigin), axis=1, keepdims=True)
    # distance sum
    arSum = np.sum(arNorm, axis=0)
    # sum/n*sq(2)
    arNum = arSum/(selectPointNum*math.sqrt(2))
    # sum be sq(2)'s new point
    arPointAfterEcu = arPointNorm / arNum

    TPrime1 = transform.estimate_transform('similarity', arPoint, arPointAfterEcu).params
    T1 = transform.estimate_transform('similarity', arPointAfterEcu, arPoint).params

    arPoint = np.c_[arPoint, np.ones(shape=(selectPointNum, 1))]
    arPoint = np.transpose(arPoint)
    # TPrime1.dot(arPoint) should be arPointAfterEcu

    # distance on Prime Point
    arNorm = np.linalg.norm((arPointPrime - pointPrimeNewOrigin), axis=1, keepdims=True)
    arSum = np.sum(arNorm, axis=0)
    arNum = arSum/(selectPointNum*math.sqrt(2))
    arPointAfterEcuPrime = arPointPrimeNorm / arNum

    TPrime2 = transform.estimate_transform('similarity', arPointPrime, arPointAfterEcuPrime).params
    T2 = transform.estimate_transform('similarity', arPointAfterEcuPrime, arPointPrime).params

    arPointPrime = np.c_[arPointPrime, np.ones(shape=(selectPointNum, 1))]
    arPointPrime = np.transpose(arPointPrime)
    # TPrime2.dot(arPointPrime) should be arPointAfterEcuPrime

    for num in range(0, selectPointNum):
        pointU = arPointAfterEcu[num][0]
        pointV = arPointAfterEcu[num][1]
        pointUPrime = arPointAfterEcuPrime[num][0]
        pointVPrime = arPointAfterEcuPrime[num][1]
        arMatrixA[num*2] = [0, 0, 0, -1*pointU, -1*pointV, -1, pointVPrime*pointU, pointVPrime*pointV, pointVPrime]
        arMatrixA[(num*2)+1] = [pointU, pointV, 1, 0, 0, 0, -1*pointUPrime*pointU, -1*pointUPrime*pointV, -1*pointUPrime]

    # svd
    u, s, vT = np.linalg.svd(arMatrixA, full_matrices=True)
    # select the last column and reshape
    arMatrixH = vT[len(vT)-1]
    arMatrixH = arMatrixH.reshape(3, 3)
    # H prime back to H
    arMatrixH = (np.linalg.inv(TPrime2).dot(arMatrixH)).dot(TPrime1)

    # Ps transpose form 100*3 to 3*100 to do mutiply
    # [[ ui1  ui2  ui3  ...  ui100]
    # [ vi1  vi2  vi3  ...  vi100]
    # [ 1, 1, 1 ... 1]]
    Ps = np.transpose(Ps)
    # caluate the Ps after homograpy
    arMatrixH = arMatrixH.dot(Ps)

    # arMatrixH transpose form 3*100 to 100*3 (the last column is param)
    arMatrixH = np.transpose(arMatrixH)
    for coordinate in arMatrixH:
        coordinate[0] = coordinate[0]/coordinate[2]
        coordinate[1] = coordinate[1]/coordinate[2]
        coordinate[2] = coordinate[2]/coordinate[2]

    # caluate the L2 norm and do average
    arError = np.linalg.norm((Pt - arMatrixH), axis=1, keepdims=True)
    errorL2Mean = np.sum(arError)/len(arError)
    print('error: ', errorL2Mean)