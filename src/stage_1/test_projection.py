import numpy as np
import os
import cv2
import math

#  this is for testing original camera parameters

def get_intrinsic_params(img_id):
    """
    read and obtain camera parameters.txt file
    :return: tuple of intrinsic parameters
    """
    img_type = img_id.strip().split("_")[0]

    intrinsic_param_file = f'../../metadata/intrinsic_parameters.txt'

    # resolution
    pixel_size = 0.00376

    intrinsic_param = []
    with open(intrinsic_param_file) as file:
        for line in file:
            parts = line.strip().split(",")
            if parts[0] == img_type:
                # fx and fy
                parts[1] = (float(parts[1]) / pixel_size)
                parts[2] = (float(parts[2]) / pixel_size)

                # cx and cy
                parts[3] = (float(parts[3]) / pixel_size)
                parts[4] = (float(parts[4]) / pixel_size)

                for item in parts:
                    intrinsic_param.append(float(item))
                return intrinsic_param


def get_extrinsic_params(img_id):
    """
    read and obtain camera parameters.txt file
    :return: tuple of intrinsic parameters
    """
    extrinsic_param_file = f'../../metadata/test.txt'

    extrinsic_param = []
    with open(extrinsic_param_file) as file:
        for line in file:
            parts = line.strip().split("    ")
            if parts[0] == img_id:
                for item in parts:
                    extrinsic_param.append(float(item))
                return extrinsic_param


def K_matrix(intrinsic_param):

    height = 10640
    width = 14192
    # Define intrinsic camera parameters
    fx = intrinsic_param[1]  # focal length in x direction
    fy = intrinsic_param[2]  # focal length in y direction
    cx = intrinsic_param[3] + width / 2 # image center x coordinate
    cy = intrinsic_param[4] + height / 2 # image center y coordinate

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    return K

def R_matrix(extrinsic_param):

    # Rotation angles in degrees
    omega = extrinsic_param[4]
    phi = extrinsic_param[5]
    kappa = extrinsic_param[6]

    omega_rad = np.deg2rad(omega)
    phi_rad = np.deg2rad(phi)
    kappa_rad = np.deg2rad(kappa)

    # Create rotation matrices for each axis
    # Rx
    R_omega = np.array([[1, 0, 0],
                   [0, np.cos(omega_rad), np.sin(omega_rad)],
                   [0, -np.sin(omega_rad), np.cos(omega_rad)]])

    R_phi = np.array([[np.cos(phi_rad), 0, -np.sin(phi_rad)],
                   [0, 1, 0],
                   [np.sin(phi_rad), 0, np.cos(phi_rad)]])

    R_kappa = np.array([[np.cos(kappa_rad), np.sin(kappa_rad), 0],
                   [-np.sin(kappa_rad), np.cos(kappa_rad), 0],
                   [0, 0, 1]])

    # Compute the composite rotation matrix
    R = np.transpose(R_omega) @ np.transpose(R_phi) @ np.transpose(R_kappa)
    print(np.transpose(R))
    return R

def get_offset():
    """
    offset of 3D coordinates
    :return: offset in x,y,z dimension
    """
    offset_file = f'../../import_Almere/offset.xyz'

    with open(offset_file) as file:
        for line in file:
            parts = line.strip().split(" ")
            offset_x = parts[0]
            offset_y = parts[1]
            offset_z = parts[2]

    return float(offset_x), float(offset_y) , float(offset_z)


def t_matrix(extrinsic_param, offset_x, offset_y, offset_z):

    t = np.array([float(extrinsic_param[1])-offset_x,
                  float(extrinsic_param[2]-offset_y),
                  float(extrinsic_param[3])-offset_z])
    return t


def projection_matrix(coord_3d, img_id):
    """
    compute projection matrix
    :param intrinsic_param: np.array([type(direction), fx,fy,cx,cy,a0,a1,a2])
    :param extrinsic_param: np.array([ID    E    N    H    O    P    K])
    :return: projection matrix of single image
    """
    intrinsic_param = get_intrinsic_params(img_id)
    extrinsic_param = get_extrinsic_params(img_id)

    offset_x, offset_y, offset_z = get_offset()


    P = np.array([coord_3d[0]-offset_x, coord_3d[1]-offset_y, coord_3d[2]-offset_z,1])

    K = K_matrix(intrinsic_param)

    # rotation matrix
    R = R_matrix(extrinsic_param)

    # 3dpoint - camera center
    t = t_matrix(extrinsic_param, offset_x, offset_y, offset_z)

    t = t.reshape(3, 1)
    t1 = -1. * R @ t
    Rt = np.hstack((R, t1.reshape(3, 1)))
    M = K @ Rt

    # camera model m = K[R | -Rt]X

    P_proj = M @ P.reshape(4, 1)

    x, y = P_proj[0]/P_proj[2], P_proj[1]/P_proj[2]

    return x, y


def correct_radial_distortion(x, y, intrinsic_param):
    # Compute the squared radial distance from the principal point
    r_squared = x ** 2 + y ** 2

    k1 = intrinsic_param[5]
    k2 = intrinsic_param[6]
    k3 = intrinsic_param[7]

    # Compute the radial distortion correction factor
    r_correction = 1 + k1 * r_squared + k2 * r_squared ** 2 + k3 * r_squared ** 3

    # Correct the distorted image coordinates
    x_corrected = x * r_correction
    y_corrected = y * r_correction

    return x_corrected, y_corrected


def plot_2d(center):
    img_path = '/Users/katherine/Desktop/402_0031_00130748.tif'
    color = (0, 0, 255)  # Red color
    radius = 5  # Circle radius
    thickness = -1  # Fill circle

    cv2.circle(img_path, center, radius, color, thickness)

    # Show the image with the projected point
    cv2.imshow('Projected Point', img_path)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':

    img_path = f'/Users/katherine/Desktop/405_0035_00124455.tif'
    coord_3d = np.array([143304.26, 487494.176, 0.1])
    x,y = projection_matrix(coord_3d, '405_0035_00124455')
    print("the result 2D coordinate is ({}, {})".format(14192-x,y))
    print("the ideal 2D coordinate is (3877, 726)")
    # plot_2d(coord_3d)

    # invertedY_x = (14192-x,y)
    #
    # GT_x, GT_y = [3877, 726]
    # # percentage error
    # print(f"Percentage error in x: {abs(14192-x - GT_x) / GT_x * 100}")
    # print(f"Percentage error in y: {abs(y - GT_y) / GT_y * 100}")
    # print(f"Percentage error in inverted y: {abs(invertedY_x[1] - GT_y) / GT_y * 100}")  # bottom left corner is (0,0)

    print("end!")
