import numpy as np
import os
import cv2
import math

def get_intrinsic_params(img_id):
    """
    read and obtain camera parameters.txt file
    :return: tuple of intrinsic parameters
    """
    img_type = img_id.strip().split("_")[2]
    intrinsic_param_file = f'../../Rotterdam/rotterdam_intrinsic.txt'

    # resolution
    pixel_size = 0.003251891892

    width = 12210
    height = 16354

    intrinsic_param = []
    with open(intrinsic_param_file) as file:
        for line in file:
            parts = line.strip().split(",")
            if parts[0] == img_type:
                # fx and fy
                parts[1] = (float(parts[1]) / pixel_size)
                parts[2] = (float(parts[2]) / pixel_size)

                # cx and cy
                parts[3] = (float(parts[3]) / pixel_size) + width / 2
                parts[4] = (float(parts[4]) / pixel_size) + height / 2

                return parts


def get_extrinsic_params(img_id):
    """
    read and obtain camera parameters.txt file
    :return: tuple of intrinsic parameters
    """
    extrinsic_param_file = f'../../Rotterdam/rotatiematrix.txt'

    extrinsic_param = []
    with open(extrinsic_param_file) as file:
        for line in file:
            parts = line.strip().split(", ")
            if parts[0] == img_id:
                for i in range(1, len(parts)):
                    extrinsic_param.append(float(parts[i]))
                return extrinsic_param


def get_camera_pos(img_id):
    """
    read and obtain camera parameters.txt file
    :return: tuple of intrinsic parameters
    """
    camera_pos_file = f'../../Rotterdam/external.txt'
    camera_pos = []
    with open(camera_pos_file) as file:
        for line in file:
            parts = line.strip().split(",")
            if parts[0] == img_id:
                for i in range(3):
                    camera_pos.append(float(parts[i+1]))
                return camera_pos

def K_matrix(intrinsic_param):

    # Define intrinsic camera parameters
    fx = float(intrinsic_param[1])  # focal length in x direction
    fy = float(intrinsic_param[2])  # focal length in y direction
    cx = float(intrinsic_param[3])  # image center x coordinate
    cy = float(intrinsic_param[4])  # image center y coordinate

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    return K

def R_matrix(rotatiematrix):

    R = np.array([[rotatiematrix[0], rotatiematrix[1], rotatiematrix[2]],
                   [rotatiematrix[3], rotatiematrix[4], rotatiematrix[5]],
                    [rotatiematrix[6], rotatiematrix[7], rotatiematrix[8]]])
    return R


def t_matrix(coord_3d, camera_pos):

    t = np.array([coord_3d[0] - float(camera_pos[0]),
                  coord_3d[1] - float(camera_pos[1]),
                  coord_3d[2] - float(camera_pos[2])])
    return t


# def projection_matrix(coord_3d, extrinsic_param):
#     """
#     compute projection matrix
#     :param intrinsic_param: np.array([type(direction), fx,fy,cx,cy,a0,a1,a2])
#     :param extrinsic_param: np.array([ID    E    N    H    O    P    K])
#     :return: projection matrix of single image
#     """
#     coord3d = np.array([coord_3d[0], coord_3d[1], coord_3d[2]])
#
#     K = K_matrix(img_id)
#
#     # rotation matrix
#     R = R_matrix(img_id)
#
#     # 3dpoint - camera center
#     t = t_matrix(coord_3d, img_id)
#
#     # Create projection matrix
#     t1 = R @ coord3d
#     t2 = t1 + t
#     P = K @ t2
#
#     return P


def perspective_projection(img_id, coord_3d):
    # Define the 3D point in world coordinates

    intrinsic_param = get_intrinsic_params(img_id)
    rotatiematrix = get_extrinsic_params(img_id)
    camera_pos = get_camera_pos(img_id)
    t1 = np.array([camera_pos[0],camera_pos[1],camera_pos[2]])
    K = K_matrix(intrinsic_param)

    # rotation matrix
    R = R_matrix(rotatiematrix)

    # 3dpoint - camera center
    t = t_matrix(coord_3d, camera_pos)

    # # Create projection matrix
    # t1 = R @ coord_3d
    # t2 = t1 + t
    # P = K @ t2
    #
    # P = P[:2]/P[2]

    # Compute the 2D homogeneous image coordinates
    X = np.array([coord_3d[0], coord_3d[1], coord_3d[2], 1]).reshape(4,1)
    M = K @ np.hstack((R, t1.reshape(3, 1)))
    x_homogeneous = M @ X
    x = x_homogeneous[:2] / x_homogeneous[2]

    width = 10640
    height = 14414

    x[1] = height - x[1]

    return x


def plot_2d(img_path):
    # img_path = '/Users/katherine/Desktop/402_0031_00130748.tif'

    img = cv2.imread(img_path)
    color = (0, 0, 255)  # Red color
    radius = 50  # Circle radius
    thickness = -1  # Fill circle
    center = (10000,13000)
    cv2.circle(img, center, radius, color, thickness)

    # Show the image with the projected point
    cv2.imshow('Projected Point', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':

    img_path = f'../../Rotterdam/23_07130_b_rgb.jpg'
    coord_3d = np.array([92418, 437710.24,1])
    x,y = perspective_projection('23_07130_b_rgb.tif', coord_3d)
    print("the 2D coordinate is {}, {}".format(x,y))
    plot_2d(img_path)

    print("end!")
