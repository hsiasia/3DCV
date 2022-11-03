from scipy.spatial.transform import Rotation
import pandas as pd
import numpy as np
import open3d as o3d
from tool import diff
from display import load_point_cloud, get_transform_mat, createCameraPosition

def main():
    # load data
    print("loading...")
    points3D_df = pd.read_pickle("data/points3D.pkl")
    print("loading done")

    # camera property
    cameraMatrix = np.array([[1868.27, 0, 540], [0, 1869.18, 960], [0, 0, 1]])

    # store all the R and t
    resultR = []
    resultT = []
    gtR = []
    gtT = []
    resultR = np.load("result/Rotation.npy")
    resultT = np.load("result/Translation.npy")
    gtR = np.load("result/gtRotation.npy")
    gtT = np.load("result/gtTranslation.npy")
    
    diffR = []
    diffT = []
    for i in range(0, resultR.shape[0]):
        d_r, d_t = diff(resultR[i], resultT[i], gtR[i], gtT[i])
        diffR.append(d_r)
        diffT.append(d_t)
        # break

    diffR=np.array(diffR)
    diffT=np.array(diffT)
    errR=np.median(diffR)
    errT=np.median(diffT)

    print("Rotation Error:{}, Translation Error:{}".format(errR, errT))

    # Display
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    
    pcd = load_point_cloud(points3D_df)
    vis.add_geometry(pcd)
    
    for i in range(0, resultR.shape[0], 10):
        R = Rotation.from_quat(resultR[i]).as_matrix()
        T = resultT[i].reshape(3, 1)

        line_set = createCameraPosition(cameraMatrix, R, T)
        vis.add_geometry(line_set)

    # just set a proper initial camera view
    vc = vis.get_view_control()
    vc_cam = vc.convert_to_pinhole_camera_parameters()
    initial_cam = get_transform_mat(np.array([7.227, -16.950, -14.868]), np.array([-0.351, 1.036, 5.132]), 1)
    initial_cam = np.concatenate([initial_cam, np.zeros([1, 4])], 0)
    initial_cam[-1, -1] = 1.
    setattr(vc_cam, 'extrinsic', initial_cam)
    vc.convert_from_pinhole_camera_parameters(vc_cam)
    
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    main()