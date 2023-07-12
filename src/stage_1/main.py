import os
import time
# # from multiprocessing import Process, Manager
# # from shapely.geometry import Point, Polygon
# # import pandas as pd
import cv2
import numpy as np

from src.stage_1.extract_texture import *
from src.stage_1.image import *
from src.stage_1.building_determination import *
from projection_almere import *
from extract_texture import *


if __name__ == '__main__':

    path = f"../../visualization/building10_wall.off"

    vertices = read_mesh_vertex(path)
    faces, colors = read_mesh_faces_1(path)
    print("finish read")

    wallsurface, new_color_list1 = wallsurface_filter_bynormal(faces, colors, vertices)

    grouped_faces, new_color_list = merge_surface(faces, colors, vertices)

    rectangles = get_off_3Dfootprint(grouped_faces, vertices)
    rect_for_projection = rectangles.copy()

    # draw_rectangles(rectangles)

    img_id = '402_0030_00131344.tif'

    pts_groundtruthlist = []
    pts_original = []
    id_count = 0
    for rectangle in rectangles:
        rect = rectangle.copy()

        new_facede_2d = projection(img_id, rect)

        pts_original.append(np.array(new_facede_2d))

        # new_facede_2d = facade_offset(new_facede_2d, img_id)

        pts = np.array(new_facede_2d, np.int32)

        name = "../../result3/calibration_403_0032_00130207_" + str(id_count) + ".jpg"

        pts_groundtruthlist.append(pts)

        # perspective(img_id, rectangle, pts, name)

        id_count += 1

    # draw multiple rectangles

    # x = []
    # y = []
    #
    # for points in pts_original:
    #     for pt in points:
    #         x.append(pt[0])
    #
    # for points in pts_groundtruthlist:
    #     for pt in points:
    #         y.append(pt[0])

    # draw_scatter(np.array(x), np.array(y))

    image = cv2.imread('/Users/yitongxia/Desktop/pipeline/' + img_id)
    for rectangle in pts_original:
        cv2.polylines(image, [rectangle], True, (0, 0, 255), 5)
    # for rectangle in pts_groundtruthlist:
    #     cv2.polylines(image, [rectangle], True, (255, 0, 0), 5)
    # cv2.imwrite(img_id +"_2_projectonly_building10wall.jpg", image)
    cv2.imshow("Image with Rectangles", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    mybbox = rectangles[1]
    visual = [mybbox]
    rect = order_rectangle_points(mybbox)
    print(rect)
    # draw_single_rectangle(rect)

    file = f'../../import_Almere/openings.json'
    w = get_openings(file)
    avg_size = size_regularization(w)

    image = cv2.imread("../../result/calibration_405_0035_00124454_1.jpg")
    height, width, channels = image.shape

    dist_xy = compute_distance(mybbox[1], mybbox[2])
    dist_x = mybbox[0][0] - mybbox[2][0]
    dist_y = mybbox[0][1] - mybbox[2][1]
    dist_z = mybbox[0][2] - mybbox[1][2]

    xy_ratio = dist_xy / width
    z_ratio = dist_z / height

    all_3d_windows = []

    for i in range(len(w)):

        for window in w[i]:
            window_3d = []
            xy_ratio = dist_xy / width
            z_ratio = dist_z / height

            top_left_3d = [rect[0][0] + (window[0] / width) * dist_x,
                           rect[0][1] + (window[0] / width) * dist_y,
                           rect[0][2] - window[1] * z_ratio]

            bottom_left_3d = [top_left_3d[0], top_left_3d[1], top_left_3d[2] - avg_size[i][1] * (dist_z / height)]

            top_right_3d = [top_left_3d[0] + avg_size[i][0] * (dist_x / width),
                            top_left_3d[1] + avg_size[i][0] * (dist_y / width),
                            top_left_3d[2]]

            bottom_right_3d = [top_right_3d[0], top_right_3d[1], bottom_left_3d[2]]

            window_3d.append([top_left_3d, bottom_left_3d, bottom_right_3d, top_right_3d])

            # visual.append(window_3d[0])
            # draw_multi_rectangle(visual)
            all_3d_windows.append(window_3d[0])

    print("see magic")
    # draw_multi_rectangle(visual)
    print("hey")

    new_window_3d = []
    for window in all_3d_windows:
        # new window
        new_corner0 = inner_corner_cal_and_val(np.array(window[0]), np.array(window[1]), np.array(window[2]))
        new_corner1 = inner_corner_cal_and_val(np.array(window[1]), np.array(window[2]), np.array(window[3]))
        new_corner2 = inner_corner_cal_and_val(np.array(window[2]), np.array(window[3]), np.array(window[0]))
        new_corner3 = inner_corner_cal_and_val(np.array(window[3]), np.array(window[0]), np.array(window[1]))

        new_window_surface = [new_corner0, new_corner1, new_corner2, new_corner3]

        # new connecting surfaces
        connecting_0 = [window[0], new_corner0, new_corner3, window[3]]
        connecting_1 = [window[1], new_corner1, new_corner0, window[0]]
        connecting_2 = [window[2], new_corner2, new_corner1, window[1]]
        connecting_3 = [window[3], new_corner3, new_corner2, window[2]]

        new_window_3d.append(new_window_surface)
        new_window_3d.append(connecting_0)
        new_window_3d.append(connecting_1)
        new_window_3d.append(connecting_2)
        new_window_3d.append(connecting_3)

    fo = open("building6_wall.off", "w")

    vertex_index = len(vertices)

    fo.write("{}{}".format("OFF", "\n"))
    fo.write("{} {} {}{}".format(len(vertices) + len(new_window_3d) * 4, len(faces) + len(new_window_3d), 0, "\n"))
    fo.write("\n")
    for i in range(len(vertices)):
        fo.write("{} {} {}{}".format(vertices[i][0], vertices[i][1], vertices[i][2], "\n"))

    window_index_list = []

    for window in new_window_3d:
        print(window)
        for corner in window:
            print(corner)
            fo.write("{} {} {}{}".format(corner[0], corner[1], corner[2], "\n"))

        window_index_list.append([vertex_index, vertex_index + 1, vertex_index + 2, vertex_index + 3])
        vertex_index += 4

    for i in range(len(faces)):
        fo.write("{}  {} {} {}{}".format(3, wallsurface[i][0], wallsurface[i][1], wallsurface[i][2], "\n"))
    # for j in range(len(wallsurface)):
    #     for i in range(len(wallsurface[j])):
    #         fo.write(
    #             "{}  {} {} {}{}".format(3, wallsurface[j][i][0], wallsurface[j][i][1], wallsurface[j][i][2], "\n"))

    for window in window_index_list:
        print(window[0], window[1], window[2], window[3])
        fo.write("{}  {} {} {} {}{}".format(4, window[0], window[1], window[2], window[3], "\n"))
    # 关闭打开的文件
    fo.close()

    print("the number of wallsurface: {}".format(len()))





