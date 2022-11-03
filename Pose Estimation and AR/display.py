
import cv2 as cv
import numpy as np
import pandas as pd
import open3d as o3d
from scipy.spatial.transform import Rotation

def load_point_cloud(points3D_df: pd.DataFrame):
    xyz = np.vstack(points3D_df['XYZ'])
    rgb = np.vstack(points3D_df['RGB'])/255

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    return pcd


def get_transform_mat(rotation, translation, scale):
    r_mat = Rotation.from_euler('xyz', rotation, degrees=True).as_matrix()
    scale_mat = np.eye(3) * scale
    transform_mat = np.concatenate([scale_mat @ r_mat, translation.reshape(3, 1)], axis=1)
    return transform_mat


def createCameraPosition(cameraMatrix, R, T):
    arCameraCorner = np.array([[0, 0, 1], [0, 1920, 1], [1080, 1920, 1], [1080, 0, 1]]).T
    # PCS -> CCS (v = K^-1 u)
    v = np.linalg.pinv(cameraMatrix) @ arCameraCorner
    # CCS -> WCS
    arCameraCorner3d = np.linalg.pinv(R) @ v + T
    # add center
    arCameraCorner3d = np.concatenate((arCameraCorner3d, T), axis=1).T

    # create o3d object lineset
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(arCameraCorner3d)
    line_set.lines = o3d.utility.Vector2iVector([[0, 1], [1, 2], [2, 3], [0, 3], [0, 4], [1, 4], [2, 4], [3, 4]])
    
    color = [1, 0, 0]
    colors = np.tile(color, (8, 1))
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


class Point_class():
    def __init__(self, position, color):
        self.position = position
        self.color = color


def generate_points(cube_vertice):
    point_list = []
    # top
    top_surface = list(cube_vertice[:4])
    top_x = (top_surface[1]-top_surface[0])/9
    top_y = (top_surface[2]-top_surface[0])/9
    for i in range(8):
        point_row_pose = top_surface[0] + (i+1)*top_x
        for j in range(10):
            point_pose = point_row_pose + j*top_y
            point_list.append(Point_class(point_pose, (255, 0, 0)))

    # front
    front_surface = list(cube_vertice[[0, 1, 4, 5]])
    front_x = (front_surface[1]-front_surface[0])/9
    front_y = (front_surface[2]-front_surface[0])/9
    for i in range(8):
        point_row_pose = front_surface[0] + (i+1)*front_x
        for j in range(8):
            point_pose = point_row_pose + (j+1)*front_y
            point_list.append(Point_class(point_pose, (0, 255, 0)))

    # back
    back_surface = list(cube_vertice[[2, 3, 6, 7]])
    back_x = (back_surface[1]-back_surface[0])/9
    back_y = (back_surface[2]-back_surface[0])/9
    for i in range(8):
        point_row_pose = back_surface[0] + (i+1)*back_x
        for j in range(8):
            point_pose = point_row_pose + (j+1)*back_y
            point_list.append(Point_class(point_pose, (255, 0, 255)))

    # botton
    botton_surface = list(cube_vertice[[4, 5, 6, 7]])
    botton_x = (botton_surface[1]-botton_surface[0])/9
    botton_y = (botton_surface[2]-botton_surface[0])/9
    for i in range(8):
        point_row_pose = botton_surface[0] + (i+1)*botton_x
        for j in range(10):
            point_pose = point_row_pose + j*botton_y
            point_list.append(Point_class(point_pose, (0, 0, 255)))

    # right
    right_surface = list(cube_vertice[[1, 3, 5, 7]])
    right_x = (right_surface[1]-right_surface[0])/9
    right_y = (right_surface[2]-right_surface[0])/9
    for i in range(10):
        point_row_pose = right_surface[0] + i*right_x
        for j in range(10):
            point_pose = point_row_pose + j*right_y
            point_list.append(Point_class(point_pose, (255, 255, 0)))

    # left
    left_surface = list(cube_vertice[[0, 2, 4, 6]])
    left_x = (left_surface[1]-left_surface[0])/9
    left_y = (left_surface[2]-left_surface[0])/9
    for i in range(10):
        point_row_pose = left_surface[0] + i*left_x
        for j in range(10):
            point_pose = point_row_pose + j*left_y
            point_list.append(Point_class(point_pose, (0, 255, 255)))

    return point_list


def draw_cube(img, R, T, cube_vertice):
    cameraMatrix = np.array([[1868.27, 0, 540], [0, 1869.18, 960], [0, 0, 1]])
    R = Rotation.from_quat(R).as_matrix()

    points = generate_points(cube_vertice)

    for i in range(len(points)):
        pixel = (cameraMatrix @ (R @ (points[i].position - T).T)).T
        pixel /= pixel[2]

        if((pixel < 0).any()):
            continue
        img = cv.circle(img, (int(pixel[0]), int(pixel[1])), radius=5, color=points[i].color, thickness=-1)

    return img