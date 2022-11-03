import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation

class P3P():
    def __init__(self, cameraMatrix, distCoeffs) -> None:
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs

    def __call__(self, point3D, point2D):
        '''
        Parameters:
            point3D [3, 4]: scene point in WCS (world coordinate system)
            point2D [2, 4]: image point in PCS (pixel coordinate system)
        Return:
            R: rotation matrix
            T: translation matrix
        '''
        # compute angles and distances of 3D points
        # PCS -> CCS (v = K^-1 u)
        v = np.linalg.pinv(self.cameraMatrix) @ np.concatenate((point2D, np.ones((1, point2D.shape[1]))))    
        v = v / np.linalg.norm(v, axis=0)

        # [3] (ab/|a||b|)
        C_ab = v[:, 0] @ v[:, 1]
        C_ac = v[:, 0] @ v[:, 2]
        C_bc = v[:, 1] @ v[:, 2]

        # distance(point3D[:, 0], point3D[:, 1])
        R_ab = norm(point3D[:, 0]-point3D[:, 1])
        R_ac = norm(point3D[:, 0]-point3D[:, 2])
        R_bc = norm(point3D[:, 1]-point3D[:, 2])
        
        # compute distances abc, x, y
        K_1 = (R_bc/R_ac)**2
        K_2 = (R_bc/R_ab)**2

        G_4 = (K_1*K_2 - K_1 - K_2)**2 - 4*K_1*K_2*(C_bc**2)
        G_3 = 4*(K_1*K_2 - K_1 - K_2)*K_2*(1 - K_1)*C_ab \
            + 4*K_1*C_bc*((K_1*K_2 - K_1 + K_2) * C_ac + 2*K_2*C_ab*C_bc)
        G_2 = (2*K_2*(1-K_1)*C_ab)**2 \
            + 2*(K_1*K_2 - K_1 - K_2)*(K_1*K_2 + K_1 - K_2) \
            + 4*K_1*((K_1 - K_2)*(C_bc**2) + K_1*(1-K_2)*(C_ac**2) - 2*(1+K_1)*K_2*C_ab*C_ac*C_bc)
        G_1 = 4*(K_1*K_2 + K_1 - K_2)*K_2*(1-K_1)*C_ab \
            + 4*K_1*((K_1*K_2 - K_1 + K_2)*C_ac*C_bc + 2*K_1*K_2*C_ab*(C_ac**2))
        G_0 = (K_1*K_2 + K_1 - K_2)**2 \
            - 4*(K_1**2)*K_2*(C_ac**2)

        # x
        roots = np.roots([G_4, G_3, G_2, G_1, G_0])
        x = np.array([np.real(r) for r in roots if np.isreal(r)])
        # y
        m, p, q = (1 - K_1), 2*(K_1*C_ac - x*C_bc), (x**2 - K_1)
        m_, p_, q_ = 1, 2*(-x*C_bc), (x**2)*(1-K_2) + 2*x*K_2*C_ab - K_2
        y = -1*(m_*q - m*q_)/(p*m_ - p_*m)
        # radius
        a = np.sqrt((R_ab**2)/(1+(x**2)-2*x*C_ab))
        b = x*a
        c = y*a
        
        # calulate camera center T
        camera_center = []
        for i in range(len(a)):
            T1, T2 = trilateration(point3D[:, 0], point3D[:, 1], point3D[:, 2], a[i], b[i], c[i])
            # [n, 1, 3] [x, y, z]
            camera_center.append(T1)
            camera_center.append(T2)

        # compute lambda, R
        # [camera_center*2, 3] [[R], [T], [lamda]]
        solutions = []
        for T in camera_center:
            # [3, 1] [[x], [y], [z]]
            T = T.reshape((3, 1))
            for sign in [1, -1]:
                # [1, 3] [norm p1, norm p2, norm p3]
                lamda = sign * norm((point3D[:, :3] - T), axis=0)
                # [3, 1, 3] R
                R = (lamda * v[:, :3]) @ np.linalg.pinv(point3D[:, :3] - T)
                if np.linalg.det(R) > 0 and (lamda>0).all():
                    solutions.append([R, T, lamda])

        # identify correct solution through 4th point
        bestR = solutions[0][0]
        bestT = solutions[0][1]
        min_error = np.Inf # infinity
        for R, T, lamda in solutions:
            # KR(P-T)
            proj2D = self.cameraMatrix @ R @ (point3D[:, 3].reshape(3, 1) - T) # [3, 1] [[x], [y], [?]]
            # homography
            proj2D /= proj2D[-1] # [3, 1] [[x], [y], [1]]
            # proj 2D(PCS) - original 2D(PCS)
            error = norm(proj2D[:2, :] - point2D[:, 3].reshape(2, 1))
            if error < min_error:
                bestR = R
                bestT = T
                min_error = error

        return bestR, bestT


def imageUndistortion(points, distCoeffs, size):
    '''
    Image undistortion, by Brown-Conrady model.
    Parameters:
        points [2, n]: image point in PCS
        distCoeffs [4,]: distortion parameters
        size [M, N]: image size
    Return:
        undistortedPoints [2, n]: image point in PCS
    '''
    # normalize
    points /= np.array([size[1], size[0]]).reshape((2, 1))

    center = np.array([0.5, 0.5]).reshape((2, 1))
    r = norm((points - center), axis=0)

    xc, yc = center[0], center[1]
    xd, yd = points[0], points[1]
    k1, k2, p1, p2 = distCoeffs[0], distCoeffs[1], distCoeffs[2], distCoeffs[3]

    xu = xd + (xd - xc)*(k1*(r**2) + k2*(r**4)) + (p1*(r**2 + 2*((xd - xc)**2)) + 2*p2*(xd - xc)*(yd - yc))
    yu = yd + (yd - yc)*(k1*(r**2) + k2*(r**4)) + (p2*(r**2 + 2*((yd - yc)**2)) + 2*p1*(xd - xc)*(yd - yc))

    undistortedPoints = np.vstack((xu, yu))*np.array([size[1], size[0]]).reshape((2, 1))
    return undistortedPoints


# ?
def trilateration(A, B, C, rA, rB, rC):
    '''
    Compute common points of three shperes
    Parameters:
        A, B, C [3,]: 3D coordinate of shperes center
        rA, rB, rC [1,]: radius
    Return:
        T1, T2: two common points
    '''
    AB = B-A
    AC = C-A
    l_AB = norm(AB)
    l_AC = norm(AC)
    l_BC = norm(C-B)

    x2 = (l_AB**2 + l_AC**2 - l_BC**2) / (2 * l_AB)
    y2 = np.sqrt(l_AC**2 - x2**2)

    x3 = (l_AB**2 + rA**2 - rB**2) / (2*l_AB)
    y3 = (l_AC**2 + rA**2 - rC**2) / 2
    y3 = (y3 - x2*x3) / y2
    
    z3 = np.sqrt(rA**2 - x3**2 - y3**2)
    
    n = np.cross(AB, AC)
    n = n / norm(n)
    AB = AB / l_AB
    b = np.cross(n, AB)

    T1 = A + x3*AB + y3*b + z3*n
    T2 = A + x3*AB + y3*b - z3*n

    return T1, T2

# v1 = P2 - P1
# v2 = P3 - P1
# i_v1 = v1 / norm(v1)
# i_v2 = v2 / norm(v2)

# # unit vector
# i_x = v1 / norm(v1)
# i_z = (np.cross(i_v1, i_v2)) / norm(np.cross(i_v1, i_v2))
# i_y = np.cross(i_x, i_z)

# c1 = np.array([0, 0, 0])
# c2 = np.array([(i_x @ v1), 0, 0])
# c3 = np.array([(i_x @ v2), (i_y @ v2), 0])

# proj_x = ((r1**2) - (r2**2) + (c2[0]**2)) / (2*c2[0])
# temp = (c3[0]**2) + (c3[1]**2)
# proj_y = ((r1**2) - (r3**2) + temp - (2*c3[0]*proj_x)) / (2*c3[1])
# proj_z = np.sqrt(r1**2 - proj_x**2 - proj_y**2)

# direction_1 = proj_x * i_x + proj_y * i_y + proj_z * i_z
# direction_2 = proj_x * i_x + proj_y * i_y - proj_z * i_z

# T1 = P1 + direction_1
# T2 = P1 + direction_2

# return T1, T2


def ransac(pnpSolver, point3D, point2D, s=3, e=0.5, p=0.99, d=10):
    """
    RANSAC algorithm (same pic and do much time P3P)
    Parameters:
        pnpSolver: pnp algorithm to get R and T
        point3D [3, n]: scene point in WCS
        point2D [2, n]: image point in PCS
        s: number of points in a sample
        e: probabiliaty that a point is an outlier
        p: desired probability
        d: distance threshold ( np.sqrt(5.99 * (self.s**2)) )
    """
    # Ransac parameter
    N = np.log((1 - p)) / np.log(1 - np.power((1 - e), s))  # number of sample = 34
    standOutlier = round(point2D.shape[1]*0.01)

    bestR = None
    bestT = None
    minOutliers = np.Inf
    for i in range(round(N)):
        # sample
        idx = np.random.randint(point2D.shape[1], size=4)
        # four points for P3P
        sample3D = point3D[:, idx]
        sample2D = point2D[:, idx]
        try:
            R, T = pnpSolver(sample3D, sample2D)
            projection = pnpSolver.cameraMatrix @ (R @ (point3D - T)) # [3, n] [[x], [y], [?]]
            projection /= projection[-1, :].reshape((1, -1)) # [n,] -> [1, n]

            errors = norm(projection[:2, :] - point2D, axis=0)
            outliers = len(errors[np.where(errors > d)])
            if outliers < minOutliers:
                print("round:{}, number of outlier: {}".format(i, outliers))
                bestR = R
                bestT = T
                minOutliers = outliers
                if outliers < standOutlier:
                    print("Inlier already larger than 99 percent, stop ransac")
                    break
        except:
            print("WARN: something error")
            
    bestR = Rotation.from_matrix(bestR).as_quat()
    bestT = bestT.reshape(-1)

    return bestR, bestT


def diff(rotq, tvec, rotq_gt, tvec_gt):
    d_t = norm(tvec_gt - tvec)

    R = Rotation.from_quat(rotq_gt).as_matrix() @ np.linalg.pinv(Rotation.from_quat(rotq).as_matrix())
    d_r = Rotation.from_matrix(R).as_rotvec()

    return d_r, d_t