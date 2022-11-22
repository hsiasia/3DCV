import open3d as o3d
import numpy as np
import cv2 as cv
import sys, os, argparse, glob
import multiprocessing as mp
import statistics

class Frame():
    def __init__(self, R, T, keypoints, descriptors, keyMatch = None, desMatch = None):
        """
        Parameters:
            R: realtive rotation matrix in WCS
            T: realtive translation matrix in WCS
            key: keypoints
            des: descriptors
            keyMatch: matching points of previous frame to current
            desMatch: matching descriptors of previous frame to current
        """
        # pose
        self.R = R
        self.T = T
        # ORB keypoint
        self.key = keypoints
        self.des = descriptors
        # for triangulation
        self.keyMatch = keyMatch
        self.desMatch = desMatch

class SimpleVO:
    def __init__(self, args):
        camera_params = np.load(args.camera_parameters, allow_pickle=True)[()]
        self.K = camera_params['K']
        self.dist = camera_params['dist']
        
        self.img_paths = sorted(list(glob.glob(os.path.join(args.input, '*.png'))))

        # self.feature_extractor = cv.ORB_create()
        # self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        self.pre_frame: Frame = None
        self.cur_frame: Frame = None
        self.pos_frame: Frame = None

        self.R_until_now = []
        self.T_until_now = []

    def run(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        queue = mp.Queue()
        p = mp.Process(target=self.process_frames, args=(queue, ))
        p.start()
        
        keep_running = True
        while keep_running:
            try:
                R, t = queue.get(block=False)
                if R is not None:
                    #TODO:
                    # insert new camera pose here using vis.add_geometry()
                    line_set = self.createCameraPosition(R, t)
                    vis.add_geometry(line_set)
                    # pass
            except: pass
            
            keep_running = keep_running and vis.poll_events()
        vis.destroy_window()
        p.join()

    def process_frames(self, queue):
        ORB = cv.ORB_create()

        # img0
        preImg = cv.imread(self.img_paths[0])
        pre_keypoints, pre_descriptors = ORB.detectAndCompute(preImg, None)
        self.pre_frame = Frame(np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64), pre_keypoints, pre_descriptors)

        # img1
        curImg = cv.imread(self.img_paths[1])
        cur_keypoints, cur_descriptors = ORB.detectAndCompute(curImg, None)
        R_relative, T_relative, keyMatch, desMatch = self.cal_matches_E_pose(pre_keypoints, pre_descriptors, cur_keypoints, cur_descriptors)
        self.cur_frame = Frame(R_relative, -1*T_relative, cur_keypoints, cur_descriptors, keyMatch, desMatch)

        self.R_until_now = R_relative
        self.T_until_now = -1*T_relative

        for frame_path in self.img_paths[2:]:
            #TODO: compute camera pose here

            # img
            posImg = cv.imread(frame_path)
            pos_keypoints, pos_descriptors = ORB.detectAndCompute(posImg, None)
            R_relative, T_relative, keyMatch, desMatch = self.cal_matches_E_pose(self.cur_frame.key, self.cur_frame.des, pos_keypoints, pos_descriptors)
            self.pos_frame = Frame(R_relative, -1*T_relative, pos_keypoints, pos_descriptors, keyMatch, desMatch)

            # triangulation
            scale = self.cal_scale(self.pre_frame, self.cur_frame, self.pos_frame)
            if scale > 2:
                scale = 2

            # R, T in WCS
            R = self.R_until_now @ R_relative
            T = scale * self.R_until_now @ (-1*T_relative) + self.T_until_now

            # update
            self.R_until_now = R
            self.T_until_now = T
            self.pre_frame = self.cur_frame
            self.cur_frame = self.pos_frame

            queue.put((R, T))
            
            # show image
            img_show = cv.drawKeypoints(posImg, pos_keypoints, None, color = (0, 255, 0))
            cv.imshow('frame', img_show)
            if cv.waitKey(30) == 27: break
    
    def cal_matches_E_pose(self, cur_keypoints, cur_descriptors, pos_keypoints, pos_descriptors):
        macher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = macher.match(cur_descriptors, pos_descriptors)

        cur_points = np.empty((0, 2))
        pos_points = np.empty((0, 2))
        temp_descriptors = np.empty((0, pos_descriptors.shape[1])) # [500, 32]
        for m in matches:
            cur_points = np.vstack((cur_points, cur_keypoints[m.queryIdx].pt))
            pos_points = np.vstack((pos_points, pos_keypoints[m.trainIdx].pt))
            temp_descriptors = np.vstack((temp_descriptors, pos_descriptors[m.trainIdx]))

        # normalize
        cur_points = cv.undistortPoints(cur_points, self.K, self.dist, None, self.K)
        pos_points = cv.undistortPoints(pos_points, self.K, self.dist, None, self.K)

        # find essential matrix
        E, _ = cv.findEssentialMat(cur_points, pos_points, self.K)
        # decompose into R, t (pos frame coordinate system to current)
        val, R_relative, T_relative, inliner = cv.recoverPose(E, cur_points, pos_points, self.K)

        # [N, 1] -> [N] -> [X, 1] -> [X]
        inliner_idx = np.squeeze(np.argwhere(np.squeeze(inliner))) # [N, 1]
        
        keyMatch = pos_points[inliner_idx, :]
        desMatch = temp_descriptors[inliner_idx, :].astype("uint8")

        return R_relative, T_relative, keyMatch, desMatch

    # ####
    def cal_scale(self, pre_frame: Frame, cur_frame: Frame, pos_frame: Frame):
        pre_proj = self.K @ np.array([[1, 0, 0, 0],
                                      [0, 1, 0, 0],
                                      [0, 0, 1, 0]], dtype=np.float32)
        # concate right [3, 3] [3, 1] -> [3, 4]
        cur_proj = self.K @ np.hstack((cur_frame.R, cur_frame.T))
        pos_proj = self.K @ np.hstack(((cur_frame.R @ pos_frame.R), (cur_frame.R @ pos_frame.T + cur_frame.T)))
        
        # find match
        macher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches_cur_pre = macher.match(cur_frame.desMatch, pre_frame.des)
        matches_cur_pos = macher.match(cur_frame.desMatch, pos_frame.desMatch)
        queryIdx_cur_pre = [m.queryIdx for m in matches_cur_pre]
        queryIdx_cur_pos = [m.queryIdx for m in matches_cur_pos]

        # [X]
        queryIdx_all, ind_cur_pre, ind_cur_pos = np.intersect1d(queryIdx_cur_pre, queryIdx_cur_pos, return_indices=True)

        pre_points = np.empty((0, 2))
        cur_points = np.empty((0, 2))
        pos_points = np.empty((0, 2))
        for i in range(len(queryIdx_all)):
            pre_points = np.vstack((pre_points, pre_frame.key[ matches_cur_pre[ind_cur_pre[i]].trainIdx ].pt))
            cur_points = np.vstack((cur_points, cur_frame.keyMatch[queryIdx_all[i]]))
            pos_points = np.vstack((pos_points, pos_frame.keyMatch[ matches_cur_pos[ind_cur_pos[i]].trainIdx ]))

        # [X, 2] -> [2, X]
        cur_points = cur_points.T
        pre_points = pre_points.T
        pos_points = pos_points.T

        # Triangulation
        N = cur_points.shape[1]
        scales = []
        for _ in range(50):
            # [2]
            idx = np.random.randint(N, size=2)
            # [2, 2]
            pre_2point = pre_points[:, idx]
            cur_2point = cur_points[:, idx]
            pos_2point = pos_points[:, idx]

            # k-1, k (pre_cur)
            structure1 = cv.triangulatePoints(pre_proj, cur_proj, pre_2point, cur_2point) # [4, 2]
            # homogenous
            structure1_homo = (structure1[:3, :] / structure1[3, :].reshape(1, -1)) # [3, 2] / [2] -> [1, 2] 
            # X - X'
            distance1 = np.linalg.norm(structure1_homo[:, 0] - structure1_homo[:, 1])

            # k, k+1 (cur_pos)
            structure2 = cv.triangulatePoints(cur_proj, pos_proj, cur_2point, pos_2point)
            structure2_homo = (structure2[:3, :] / structure2[3, :].reshape(1, -1))
            distance2 = np.linalg.norm(structure2_homo[:, 0] - structure2_homo[:, 1])

            scales.append((distance2 / distance1))
            median_scale = statistics.median(scales)
        
        return median_scale

    def createCameraPosition(self, R, T):
        arCameraCorner = np.array([[0, 0, 1], [0, 360, 1], [640, 360, 1], [640, 0, 1]]).T
        v = np.linalg.pinv(self.K) @ arCameraCorner
        arCameraCorner3d = np.linalg.pinv(R) @ v + T
        arCameraCorner3d = np.concatenate((arCameraCorner3d, T), axis=1).T

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(arCameraCorner3d)
        line_set.lines = o3d.utility.Vector2iVector([[0, 1], [1, 2], [2, 3], [0, 3], [0, 4], [1, 4], [2, 4], [3, 4]])

        color = [0, 0, 1]
        colors = np.tile(color, (8, 1))
        line_set.colors = o3d.utility.Vector3dVector(colors)

        return line_set

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='frames/', help='directory of sequential frames')
    parser.add_argument('--camera_parameters', default='camera_parameters.npy', help='npy file of camera parameters')
    args = parser.parse_args()

    vo = SimpleVO(args)
    vo.run()
