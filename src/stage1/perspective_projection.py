import numpy as np
import cv2
from PIL import Image

def calculate_Rt():

    t = np.array([143866.848, 487542.401, 440.729])

    omega = np.radians(-0.23827)
    phi = np.radians(45.082413)
    kappa = np.radians(90.241991)

    sin_w = np.sin(omega)
    sin_phi = np.sin(phi)
    sin_kappa = np.sin(kappa)

    adjusted = np.array([-0.23827,45.082413,90.241991])

    Rx = np.array([[1.0,0.0,0.0],
                  [0.0, np.cos(omega), -1. * np.sin(omega)],
                  [0.0, np.sin(omega), np.cos(omega)]])

    Ry = np.array([[np.cos(phi), 0.0,  np.sin(phi)],
                  [0.0, 1.0, 0.0],
                  [-1. * np.sin(phi), 0.0, np.cos(phi)]])

    Rz = np.array([[np.cos(kappa), np.sin(kappa), 0.0],
                  [-1.* np.sin(kappa),np.cos(kappa), 0.0],
                  [0.0, 0.0 ,1.0]
                   ])

    R_new = np.array([[np.cos(phi) * np.cos(kappa), -1.* np.cos(phi) * np.sin(kappa), np.sin(phi)],
                      [np.cos(omega) * np.sin(kappa) + np.sin(omega) * np.cos(kappa), np.cos(omega) * np.cos(kappa) - np.sin(omega) * np.sin(phi) * np.sin(kappa), -1.* np.sin(omega) * np.cos(phi)],
                      [np.sin(omega) * np.sin(kappa) - np.cos(omega) * np.cos(kappa), np.sin(omega) * np.cos(kappa) + np.cos(omega) * np.sin(phi) * np.sin(kappa), np.cos(omega) * np.cos(phi)]])

    R = Rz @ Ry @ Rx

    correct_R = np.array([
        [-0.00298218550975265896, 0.99999487023268418540,-0.00116879335309075217],
        [0.70608266689260990034, 0.00127802044912532996, -0.70812826110638393828],
        [-0.70812313483136868353, -0.00293703456711447977, -0.70608285614688770515]
    ])

    D = np.array([[1., 1., -1.],
                  [-1., -1., 1.],
                  [1., 1., -1.]])

    R_new = R * D

    print(R)


def get_K(f, cx, cy):
    """
    get intrinsic matrix
    :param f: focal length
    :param cx: princple point x
    :param cy: principle point y
    :return: return np.array format get intrinsic matrix
    """
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]])
    return K



def get_offset(img_id):
    """
    offset of 3D coordinates
    :return: offset in x,y,z dimension
    """
    offset_file = f'../../import/offset.xyz'

    img_type = img_id.strip().split("_")[0]

    with open(offset_file) as file:
        for line in file:
            parts = line.strip().split(" ")

            if parts[0] == img_type:
                offset_x = parts[1]
                offset_y = parts[2]
                offset_z = parts[3]

                return float(offset_x), float(offset_y), float(offset_z)

    # could be modified to adapt to various type of txt file.


def get_camera_parameters(img_id):
    """
    :param img_id: image id
    :return: return original R and t
    """
    cp_file = f'../../import/calibrated_camera_parameters.txt'

    if_img = False
    parameters_count = 0
    camera_parameters = []
    with open(cp_file) as lines:
        for line in lines:
            parts = line.strip().split(" ")
            if parts[0] == img_id:
                if_img = True
                continue
            elif if_img == True and parameters_count != 9:
                camera_parameters.append(parts)
                parameters_count+=1
            elif parameters_count == 9:
                break

    K = np.array([camera_parameters[0], camera_parameters[1], camera_parameters[2]]).astype(np.float32)
    t = np.array([camera_parameters[5]]).astype(np.float32)
    R = np.array([camera_parameters[6], camera_parameters[7], camera_parameters[8]]).astype(np.float32)

    t = t.reshape(3, 1)
    t1 = -1. * R @ t
    Rt = np.hstack((R, t1.reshape(3, 1)))
    KRt = K @ Rt

    return KRt


def get_KRt(img_id):
    """
    :param img_id: image id
    :return: return original R and t
    """
    cp_file = f'../../import/calibrated_camera_parameters.txt'

    if_img = False
    parameters_count = 0
    camera_parameters = []
    with open(cp_file) as lines:
        for line in lines:
            parts = line.strip().split(" ")
            if parts[0] == img_id:
                if_img = True
                continue
            elif if_img == True and parameters_count != 9:
                camera_parameters.append(parts)
                parameters_count+=1
            elif parameters_count == 9:
                break

    K = np.array([camera_parameters[0], camera_parameters[1], camera_parameters[2]]).astype(np.float32)
    t = np.array([camera_parameters[5]]).astype(np.float32)
    R = np.array([camera_parameters[6], camera_parameters[7], camera_parameters[8]]).astype(np.float32)

    return K, R, t


def projection(img_id, P):

    M = get_camera_parameters(img_id)

    offset_x, offset_y, offset_z = get_offset(img_id)

    Ps = []
    for line in P:
        P_new = [0, 0, 0]
        P_new[0] = line[0] - offset_x
        P_new[1] = line[1] - offset_y
        P_new[2] = line[2] - offset_z
        # reshape 3D point coordinates
        P_new = np.hstack((P_new, np.array([1])))
        Ps.append(P_new)

    # Ps = np.hstack((P,np.array([1,1,1,1]).reshape(4,1)))

    point = []
    for pt in Ps:
        P_proj = M @ pt.reshape(4, 1)
        x = P_proj[:2] / P_proj[2]
        point.append([int(x[0][0]), int(x[1][0])])

    # Print result

    # for i in range(len(P)):
    #     print("original 3D Point: ({}, {}, {}), ->> ({}, {})".format(P[i][0], P[i][1], P[i][2], point[i][0],
    #                                                                  point[i][1]))
    return point


def draw_points(points, img_id):
    image = cv2.imread('../../image/' + img_id)
    cv2.imshow('Image', image)
    # Create a window to display the image
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    # points = np.array([[8359, 4986], [8367, 4986], [8366, 4986], [8359, 4986]])
    # Draw a polygon on the image
    # points = np.array([[6872, 3293], [6865, 3834], [6395, 3914], [6393, 3392]], np.int32)
    cv2.polylines(image, [points], True, (0, 0, 255), thickness=5)
    cv2.imwrite("../../result/wrong_example.jpg", image)
    # Show the image
    # cv2.imshow('Image', image)
    # # Wait for a key press and then close the window
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def line_detection(path):
    # line detection
    img1 = cv2.imread(path)

    # Convert to grayscale
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Apply HoughLinesP to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    # Draw the detected lines on the original image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img1, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show the result
    cv2.imshow('Result', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def distortion_calibration(f, cx, cy, k1, k2, p1, p2, k3):
    """
    :param f: focal length
    :param cx: ppx
    :param cy: ppy
    :param k1: radial distortion k1
    :param k2: radial distortion k2
    :param p1: tangent distortion p1
    :param p2: tangent distortion p2
    :param k3: radial distortion k3
    :return: undistored image
    """
    img = cv2.imread('distorted_image.png')
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    dist_coeff = np.array([k1, k2, p1, p2, k3])
    img_undistorted = cv2.undistort(img, K, dist_coeff)

    return img_undistorted


def project_3D(img_id):

    pt_2d = np.array([[0., 0.],
                      [0., 10640.],
                      [14192., 10640.],
                      [14192., 0.]])

    K, R, t = get_KRt(img_id)

    offset_x, offset_y, offset_z = get_offset()

    for pt in pt_2d:
        # Define the 2D point coordinates
        u = pt[0]
        v = pt[1]

        K_inv = np.linalg.inv(K)

        # Compute the camera extrinsic matrix
        extrinsic = np.hstack((R, t.reshape(-1, 1)))

        # Compute the inverse of the camera extrinsic matrix
        extrinsic_inv = np.linalg.inv(extrinsic)

        # Compute the 3D point coordinates
        pt_3d = np.dot(extrinsic_inv, np.dot(K_inv, pt_2d))
        print("3D point coordinates: ", pt_3d)


import numpy as np


def pixel_to_3d_coordinates (x, y, z, projection_matrix):
    P_inv = np.linalg.inv(projection_matrix)
    pixel_homogeneous = np.array([x, y, 1])
    coordinates_3D = P_inv @ pixel_homogeneous
    coordinates_3D_normalized = coordinates_3D / coordinates_3D[-1]
    coordinates_3D_final = coordinates_3D_normalized * z

    return coordinates_3D_final[:3]

if __name__ == '__main__':

    img_id = '405_0035_00124455.tif'
    project_3D(img_id)

    # Example usage:
    projection_matrix = get_camera_parameters(img_id)

    x, y = 0,0
    z = 0

    result = pixel_to_3d_coordinates(x, y, z, projection_matrix)
    print(result)
#
#     P = np.array([[143590.562500, 487118.562500, 15.406000],
#                  [143642.093750, 487131.625000, 14.878000]])
#
#     img_id = '405_0035_00124455.tif'
#     projection(img_id, P)


#
#     # image: 404_0031_00130735
#
#     P = np.array([
#         [143336.17, 487572.916, 24.388],
#         [143336.17, 487572.916, -4.781],
#         [143330.841, 487585.665, -4.781],
#         [143330.841, 487585.665, 24.388]
#     ])
#
#
#     f = 29754.99132637640650500543
#     cx = 7034.54172294926956965355
#     cy = 5302.60023224232736538397
#
#     projection(f, cx, cy, P)
#     #
#     path = "/Users/katherine/Desktop/ro/crop1.jpg"
#     img = cv2.imread('/Users/katherine/Desktop/404_0031_00130735.tif')
#     print(img.shape)
#     cropped = img[1610:2752, 8442:9040]  # 裁剪坐标为[y0:y1, x0:x1]
#     cv2.imwrite(path, cropped)



