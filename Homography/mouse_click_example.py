import sys
import numpy as np
import cv2 as cv

WINDOW_NAME = 'window'


def on_mouse(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        param[0].append([x, y])  


def mouth_click(path):
    img = cv.imread(path)

    points_add= []
    cv.namedWindow(WINDOW_NAME)
    cv.setMouseCallback(WINDOW_NAME, on_mouse, [points_add])
    while True:
        img_ = img.copy()
        for i, p in enumerate(points_add):
            # draw points on img_
            cv.circle(img_, tuple(p), 2, (0, 255, 0), -1)
        cv.imshow(WINDOW_NAME, img_)

        key = cv.waitKey(20) % 0xFF
        if key == 27: break # exist when pressing ESC

    cv.destroyAllWindows()
    
    # print('{} Points added'.format(len(points_add)))
    return points_add