import pandas as pd
import numpy as np
import cv2 as cv
from tool import P3P, ransac, imageUndistortion

# load data
print("loading...")
images_df = pd.read_pickle("data/images.pkl")
train_df = pd.read_pickle("data/train.pkl")
points3D_df = pd.read_pickle("data/points3D.pkl")
point_desc_df = pd.read_pickle("data/point_desc.pkl")
print("loading done")

# camera property
cameraMatrix = np.array([[1868.27, 0, 540], [0, 1869.18, 960], [0, 0, 1]])
distCoeffs = np.array([0.0847023, -0.192929, -0.000201144, -0.000725352])

class Match():
    def __init__(self):
        # Process model descriptors
        desc_df = self.average_desc(train_df, points3D_df)
        self.kp_model = np.array(desc_df["XYZ"].to_list()) #3D point
        self.desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32) #3D des

    def average_desc(self, train_df, points3D_df):
        train_df = train_df[["POINT_ID", "XYZ", "RGB", "DESCRIPTORS"]]
        desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
        desc = desc.apply(lambda x: list(np.mean(x, axis=0)))
        desc = desc.reset_index()
        desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
        return desc

    def find_match(self, idx):
        # Load query keypoints and descriptors
        points = point_desc_df.loc[point_desc_df["IMAGE_ID"] == idx]
        kp_query = np.array(points["XY"].to_list()) #2D point
        desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32) #2D des

        bf = cv.BFMatcher()
        matches = bf.knnMatch(desc_query, self.desc_model, k=2)

        gmatches = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                gmatches.append(m)

        point2D = np.empty((0, 2))
        point3D = np.empty((0, 3))

        for mat in gmatches:
            query_idx = mat.queryIdx
            model_idx = mat.trainIdx
            point2D = np.vstack((point2D, kp_query[query_idx]))
            point3D = np.vstack((point3D, self.kp_model[model_idx]))

        return point3D, point2D

def main():
    image_id = images_df["IMAGE_ID"].to_list()
    img_size = [1920, 1080]

    # create match
    match = Match()

    # create solver
    pnpSolver = P3P(cameraMatrix, distCoeffs)

    # store all the R and t
    resultR = []
    resultT = []
    gtR = []
    gtT = []
    
    for i in range(163, len(image_id)):
        # process all image
        idx = image_id[i]

        # read image
        points3D, points2D = match.find_match(idx)
        points3D = points3D.T
        points2D = points2D.T

        points2D = imageUndistortion(points2D, distCoeffs, img_size)

        print("\n[{}/{}] {}, select {} points".format(i+1, len(image_id),((images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values)[0], points2D.shape[-1]))

        # solve PnP
        R, T = ransac(pnpSolver, points3D, points2D)

        # Get camera pose groudtruth
        ground_truth = images_df.loc[images_df["IMAGE_ID"] == idx]
        rotq_gt = ground_truth[["QX", "QY", "QZ", "QW"]].values
        tvec_gt = ground_truth[["TX", "TY", "TZ"]].values

        resultR.append(R)
        resultT.append(T)
        gtR.append(np.squeeze(rotq_gt))
        gtT.append(np.squeeze(tvec_gt))

    resultR = np.array(resultR)
    resultT = np.array(resultT)
    gtR = np.array(gtR)
    gtT = np.array(gtT)

    np.save("result/Rotation.npy", resultR)
    np.save("result/Translation.npy", resultT)
    np.save("result/gtRotation.npy", gtR)
    np.save("result/gtTranslation.npy", gtT)


if __name__ == '__main__':
    main()