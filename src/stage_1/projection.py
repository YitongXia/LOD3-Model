import numpy as np
import cv2
from PIL import Image


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

def get_offset():
    """
    offset of 3D coordinates
    :return: offset in x,y,z dimension
    """
    offset_file = f'../../Rotterdam/pix4d_offset.xyz'

    with open(offset_file) as file:
        for line in file:
            parts = line.strip().split(" ")
            offset_x = parts[0]
            offset_y = parts[1]
            offset_z = parts[2]

    return float(offset_x), float(offset_y) , float(offset_z)


def get_Rt(img_id):
    """
    :param img_id: image id
    :return: return original R and t
    """
    Rt_file = f'../../Rotterdam/pix4d_camera_parameters.txt'
    if_img = False
    if_t = 0
    if_R = 0
    t = []
    R = []
    with open(Rt_file) as lines:
        for line in lines:
            parts = line.strip().split(" ")
            if parts[0] == img_id:
                if_img = True
                continue
            elif if_img == True and if_t == 0:
                t = np.array([float(parts[0]),float(parts[1]),float(parts[2])])
                if_t = 1
            elif if_img == True and if_t == 1 and if_R != 3:
                R.append([float(parts[0]),float(parts[1]),float(parts[2])])
                if_R += 1
            elif if_img == True and if_R == 3:
                return np.array(R),np.array(t)


def projection(f, cx, cy, img_id, P):
    """
    :param f: optimized focal length
    :param cx: optimized ppx
    :param cy: optimized ppy
    :param img_id: image id
    :param P: 3D point list (np.array([[],[],[]]), in RD28992
    :return: 2D image coodinates
    """
    K = get_K(f, cx, cy)

    offset_x, offset_y, offset_z = get_offset()

    R,t = get_Rt(img_id)
    t1 = -1. * R @ t
    Rt = np.hstack((R, t1.reshape(3, 1)))

    for line in P:
        line[0] = line[0] - offset_x
        line[1] = line[1] - offset_y
        line[2] = line[2] - offset_z

    Ps = np.hstack((P,np.array([1,1,1,1]).reshape(4,1)))

    M = K @ Rt

    point = []
    for pt in Ps:
        P_proj = M @ pt.reshape(4, 1)
        x = P_proj[:2] / P_proj[2]
        point.append([int(x[0][0]), int(x[1][0])])

    # Print result

    for i in range(len(Ps)):
        print("original 3D Point: ({}, {}, {}), ->> ({}, {})".format(P[i][0], P[i][1], P[i][2], point[i][0],
                                                                     point[i][1]))

    return point



def draw_points():
    image = cv2.imread('/Users/katherine/Desktop/ro/a_23_07132_b_rgb.jpg')
    cv2.imshow('Image', image)
    # Create a window to display the image
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    # points = np.array([[8359, 4986], [8367, 4986], [8366, 4986], [8359, 4986]])
    # Draw a polygon on the image
    points = np.array([[6872, 3293], [6865, 3834], [6395, 3914], [6393, 3392]], np.int32)
    cv2.polylines(image, [points], True, (255, 0, 0), thickness=10)
    # Show the image
    cv2.imshow('Image', image)
    # Wait for a key press and then close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()



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


if __name__ == '__main__':

    # image: 23_07133_b_rgb.jpg

    P = np.array([
        [92488.375000 , 437400.843750, 64.726997],
        [92488.453125 , 437398.656250, 41.040001],
        [92494.210938, 437380.000000, 42.681999],
        [92493.96875, 437382.593750, 66.226997]
    ])

    f = 32804.179
    cx = 7067.704
    cy = 6135.600

    projection(f, cx, cy, '23_07133_b_rgb.jpg', P)


