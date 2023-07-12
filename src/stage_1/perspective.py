import numpy as np
import cv2


def order_points(pts):
    # initialization
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

# pts:  [[5705.1216 2671.401 ], [7685.2334 4267.521 ], [8099.2563 2218.967 ], [8099.2563 2218.967 ]]
# rect: [[5705.1216 2671.401 ], [8099.2563 2218.967 ], [7685.2334 4267.521 ], [5705.1216 2671.401 ]]
def four_point_transform(image, pts):
    #  Get the coordinate points and separate them
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Calculate the width value of the new image and select the maximum value of the horizontal difference
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Calculate the height value of the new image and select the maximum value of the vertical difference
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Construct the 4 coordinate points of the new image, with the top left corner as the origin
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Get the perspective transformation matrix
    M = cv2.getPerspectiveTransform(rect, dst)
    # Perform Perspective Transformations
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the result
    return warped


def on_mouse(event, x, y, flags, param):

    """Click on the original image to get the coordinates of the four corner points of the rectangle"""

    global timg, points
    img2 = timg.copy()
    p0 = (0, 0)  # initialization
    if event == cv2.EVENT_LBUTTONDOWN:
        p1 = (x, y)
        points.append([x, y])
        print(p1)

        # Draw a circle at the clicked image
        # cv2.circle(image, center_coordinates, radius, color, thickness)
        cv2.circle(img2, p1, 50, (0, 255, 0), 4)
        cv2.imshow('origin', img2)
    return p0


if __name__ == "__main__":
    global points, timg
    xscale, yscale =  0.7, 0.7 # If the image is not precise enough, can enlarge it by adjusting the parameters
    points = []
    # input image data
    img = cv2.imread('/Users/yitongxia/Desktop/pipeline/405_0035_00124457.tif')
    shape = img.shape
    timg = cv2.resize(img, (int(shape[1] / xscale), int(shape[0] / yscale)))  # enlarge image
    print(timg.shape)
    cv2.imshow('origin', timg)

    cv2.setMouseCallback('origin', on_mouse)  # The name of the image to be displayed here must be the same as the one set in the previous sentence and in the on_mouse function
    cv2.waitKey(0)  # After the four corner dots are clicked, press the keyboard randomly to end the operation
    cv2.destroyAllWindows()

    # Restore Pixel Position
    points = np.array(points, dtype=np.float32)
    points[:,0] *= shape[1] / int(shape[1]/xscale)
    points[:,1] *= shape[0] / int(shape[0]/yscale)
    warped = four_point_transform(img, points)

    # if user wanna show the result, use this
    cv2.imshow('results', warped)
    # output image data
    cv2.imwrite("/Users/katherine/Desktop/facade3.jpg", warped)
    cv2.waitKey(0)