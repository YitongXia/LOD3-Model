import cv2
import numpy
import math

# from projection_almere import *
from src.stage_1.building_determination import *
from src.stage_1.perspective_projection import *

def get_corner(proj_2d):

    x = []
    y = []

    for pt in proj_2d:
        x.append(pt[0])
        y.append(pt[1])

    x0 = np.amin(x)
    x1 = np.amax(x)
    y0 = np.amin(y)
    y1 = np.amax(y)

    return y0, y1, x0, x1


def order_points(pts):
    # initialization
    pts = np.array(pts)
    rect = np.zeros((4, 2), dtype='float32')

    # Get the top-left and bottom-right coordinate points
    s = pts.sum(axis=1)  # Pixel values are summed for each row; if axis=0, pixel values are summed for each column
    rect[0] = pts[np.argmin(s)]  # top_left, return s first minimum index, eg.[1,0,2,0], return value is 1
    rect[2] = pts[np.argmax(s)]  # bottom_left, return s first maximum index, eg.[1,0,2,0], return value is 2

    # Calculate the discrete difference between the upper left corner and the lower right corner, respectively
    diff = np.diff(pts, axis=1)  # Column i+1 - column i
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def compute_distance(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def four_point_transform(image, bbox, pts_2d):
    #  Get the coordinate points and separate them
    # bbox = [[xmax, ymax, zmax], [xmax, ymax, zmin], [xmin, ymin, zmin], [xmin, ymin, zmax]]
    # [max_pt, [max_pt[0], max_pt[1], min_pt[2]], min_pt, [min_pt[0], min_pt[1], max_pt[2]]]
    maxHeight = int((bbox[0][2] - bbox[1][2]) * 100)

    maxWidth = int((compute_distance(bbox[1], bbox[2])) * 100)

    # Construct the 4 coordinate points of the new image, with the top left corner as the origin
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    rect = order_points(pts_2d)
    # Get the perspective transformation matrix
    # still a problem.
    M = cv2.getPerspectiveTransform(rect, dst)
    # Perform Perspective Transformations
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the result
    return warped


def extract_facade (img_id, building_id, facade_corner_2d):

    y0, y1, x0, x1 = get_corner(facade_corner_2d)

    # crop image
    path = "../../image/" + img_id + building_id + ".jpg"
    input_img = '/Users/yitongxia/Desktop/pipeline/'+img_id
    img = cv2.imread(input_img)
    print(img.shape)
    cropped = img[y0-800:y1-800, x0:x1]  # 裁剪坐标为[y0:y1, x0:x1]
    cv2.imwrite(path, cropped)


def perspective(img_id, bbox, facade_corner_2d, calibrating_output):
    # crop image
    path = f"/Users/yitongxia/Desktop/pipeline/"+ img_id
    # input image data
    calibrating_img = cv2.imread(path)

    # Restore Pixel Position
    warped = four_point_transform(calibrating_img, bbox, facade_corner_2d)

    # output image data
    cv2.imwrite(calibrating_output, warped)


def facade_extraction(visible_group, img_id, f, cx, cy):

    for i in range(len(visible_group)):

        # (max, maxy, minx, miny for each facade)
        max_pt, min_pt = find_merge_bbox(visible_group, 4)

        # four corner 3D coordinates
        P = np.array([[max_pt[0], max_pt[1], max_pt[2]],
                      [max_pt[0], max_pt[1], min_pt[2]],
                      [min_pt[0], min_pt[1], min_pt[2]],
                      [min_pt[0], min_pt[1], max_pt[2]]])

        # four corner 2D coordinates
        proj_2d = projection(f, cx, cy, P)

        # cropping parameters
        y0,y1,x0,x1 = get_corner(proj_2d)

        # crop image
        path = "../../image/"+img_id+i+".jpg"
        img = cv2.imread('/Users/yitongxia/Desktop/pipeline/404_0031_00130735.tif')
        print(img.shape)
        cropped = img[y0:y1, x0:x1]  # 裁剪坐标为[y0:y1, x0:x1]
        cv2.imwrite(path, cropped)

        # input image data
        calibrating_img = cv2.imread(path)
        shape = img.shape

        # Restore Pixel Position
        warped = four_point_transform(calibrating_img, proj_2d, P)

        calibrating_output = "../../image/"+img_id+i+"calibrated.jpg"
        # output image data
        cv2.imwrite(calibrating_output, warped)


def crop_image():
    path = "/Users/katherine/Desktop/ro/crop1.jpg"
    img = cv2.imread('/Users/katherine/Desktop/ro/23_07133_b_rgb.jpg')
    print(img.shape)
    cropped = img[1850:2400, 4650:5160]  # 裁剪坐标为[y0:y1, x0:x1]
    cv2.imwrite(path, cropped)


def draw_points(pts_2d, img_id):
    img = cv2.imread('/Users//Users/yitongxia/Desktop/pipeline/'+ img_id)

    # Create a copy of the image
    img_copy = img.copy()

    # Define the points

    points = np.array(pts_2d)

    # Loop through the points and draw circles
    for point in points:
        cv2.circle(img_copy, tuple(point), 50, (0, 255, 0), -1)

    # Display the resulting image
    cv2.imshow('Image with points', img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()