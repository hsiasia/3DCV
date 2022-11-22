import pandas as pd
import numpy as np
import cv2 as cv
from display import draw_cube

def main():
    images_df = pd.read_pickle("data/images.pkl")
    image_id = images_df["IMAGE_ID"].to_list()
    imgName = []

    for i in range(163, len(image_id)):
        idx = image_id[i]
        fname = ((images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values)[0]
        imgName.append(fname)
    imgName = sorted(imgName, key=lambda name: int(name[name.find('g')+1 : name.find('.')]))
    
    resultR = np.load("result/Rotation.npy")
    resultT = np.load("result/Translation.npy")
    img = []
    R = []
    T = []
    for i in range(len(imgName)):
        idxImg = ((images_df.loc[images_df["NAME"] == imgName[i]])["IMAGE_ID"].values)[0]
        idxRT = idxImg-164
        
        fname = imgName[i]
        rimg = cv.imread("data/frames/" + fname, cv.IMREAD_COLOR)

        img.append(rimg)
        R.append(resultR[idxRT])
        T.append(resultT[idxRT])
    
    # create 3D cube
    cube_vertice = np.load("cube_vertices.npy")

    out = cv.VideoWriter("AR.avi", cv.VideoWriter_fourcc(*'MJPG'), 20, (1080, 1920))
    for i in range(len(img)):
        img[i] = draw_cube(img[i], R[i], T[i], cube_vertice)
        out.write(img[i])

    cv.destroyAllWindows()
    out.release()


if __name__ == '__main__':
    main()