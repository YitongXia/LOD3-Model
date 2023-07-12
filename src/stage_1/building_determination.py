import json

from matplotlib import pyplot as plt
import numpy as np

np.set_printoptions(suppress=True)
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from src.stage_1.extract_texture import *
from src.stage_1.projection_almere import *
import math
from src.stage_1.surface_merge import *
from src.layout import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from matplotlib.font_manager import FontProperties
from src.stage_1.integration import *

#
# ""
# cityjson file import function
# """
# def import_cityjson(file):
#     """
#     read CityJSON and store as json format
#     :param cityjson_file: filename
#     :return: json
#     """
#     with open(file, "r") as file:
#         # Read the contents of the file
#         json_data = file.read()
#
#     # Decode the JSON string to a Python object
#     input_cityjson = json.loads(json_data)
#
#     # Print the data
#     return input_cityjson
#
# """
# compute the centroid of a building using the footprint
# """
# def compute_center(id, j):
#     """
#     calculate the center coordinates of the selected building using footprint (LOD0)
#     :param id: id of the building
#     :param j: the whole cityjson file
#     :return: center coordinates in RD28992
#     """
#     for each in j["CityObjects"]:
#         if 'geometry' in j['CityObjects'][each]:
#             for geom in j['CityObjects'][each]['geometry']:
#                 if each == id:
#                     if geom["lod"] == 0:
#                         x_sum, y_sum, x, y = 0, 0, 0, 0
#                         for i in geom["boundaries"][0][0]:
#
#                             x = float(j["vertices"][i][0]) * float(j["transform"]["scale"][0]) + j["transform"]["translate"][0]
#                             y = float(j["vertices"][i][1]) * float(j["transform"]["scale"][1]) + j["transform"]["translate"][1]
#
#                             x_sum = x + x_sum
#                             y_sum = y + y_sum
#
#                         x_sum = x_sum / len(geom["boundaries"][0][0])
#                         y_sum = y_sum / len(geom["boundaries"][0][0])
#
#
#                         return Point2(x_sum, y_sum)
#
#
# """
# compute the footprint
# """
# def get_footprint(id, j):
#     """
#     return footprint (LOD0) of the buildings
#     :param id: id of the building
#     :param j: the whole cityjson file
#     :return: footprints 2d coordinates list of building
#     """
#     # footprint = np.array([])
#     for each in j["CityObjects"]:
#         if each == id:
#             for geom in j["CityObjects"][each]['geometry']:
#                 if geom["lod"] == 0:
#                     footprint = geom["boundaries"][0][0]
#
#     footprint_2d = []
#     for i in footprint:
#         x = j["vertices"][i][0] * j["transform"]["scale"][0] + j["transform"]["translate"][0]
#         y = j["vertices"][i][1] * j["transform"]["scale"][1] + j["transform"]["translate"][1]
#         footprint_2d.append([x, y])
#
#     return footprint_2d
#
# """
# bbox is defined as (minx, maxx, miny, maxy)
# """
# def find_building_in_range(img_center, bbox, j):
#     """
#     find buildings in pre-defined range
#     :param img_center:
#     :param bbox:
#     :param j:
#     :return: tuple
#     """
#     id=[]
#     for each in j["CityObjects"]:
#         x, y = compute_center(each, j)
#         if x >= bbox[0] & x <= bbox[1] & y >= bbox[2] & y>=bbox[3]:
#             id.append(each)
#     return id
#
#
# """
# convert point pairs to edges (skgeometry)
# """
# def points_2_edges(pts):
#     """ Make edges """
#     edges = []
#     for i in range(1,len(pts)):
#         e = Segment2(Point2(pts[i-1][0], pts[i-1][1]),
#                      Point2(pts[i][0], pts[i][1]))
#         edges.append(e)
#
#     e1 = Segment2(Point2(pts[0][0], pts[0][1]),
#                   Point2(pts[len(pts)-1][0], pts[len(pts)-1][1]))
#     edges.append(e1)
#     return edges
#
#
# """
# compute a closer observer
# (still need to be modified, to make it self-adaptive)
# """
# def get_observer(pt1, pt2):
#     scale = 0.95
#     x_new = pt1.x() + scale * (pt2.x() - pt1.x())
#     y_new = pt1.y() + scale * (pt2.y() - pt1.y())
#     new_pt1 = Point2(x_new, y_new)
#     return new_pt1
#
#
# """
# draw arrangement (skgeometry) using matplotlib)
# """
# def draw_arrangement(arr, vx, q, save_file=False):
#     """
#     Draw 2D arrangement with buildings and initial
#     panoramic image location
#     """
#     from matplotlib import pyplot as plt
#
#     plt.figure(figsize=(10, 10))
#     plt.xlabel("X Position")
#     plt.ylabel("Y Position")
#
#     # Draw walls and buildings
#     for he in arr.halfedges:
#         plt.plot([he.curve().source().x(), he.curve().target().x()],
#                      [he.curve().source().y(), he.curve().target().y()], "b")
#
#     # Draw Visibility Polygon
#     for he in vx.halfedges:
#         plt.plot([he.curve().source().x(), he.curve().target().x()],
#                      [he.curve().source().y(), he.curve().target().y()], "r:")
#
#     plt.scatter(q.x(), q.y(), color="b")
#     if save_file:
#         plt.savefig("vispol.eps", format="eps")
#     else:
#         plt.show()
#
#
# """
# get the camera position in real world
# """
# def get_camera_pos(img_id, city):
#     """
#     read and obtain camera parameters.txt file
#     :return: tuple of intrinsic parameters
#     """
#     extrinsic_param_almere = f'../../metadata/EO_obliek_verwerkt.txt'
#     extrinsic_param_rotterdam = f'../../Rotterdam/external.txt'
#     extrinsic_param = []
#     if city == "Almere":
#         with open(extrinsic_param_almere) as file:
#             for line in file:
#                 parts = line.strip().split("    ")
#                 if parts[0] == img_id:
#                     for item in parts[1:]:
#                         extrinsic_param.append(float(item))
#                     return extrinsic_param
#     elif city == "Rotterdam":
#         with open(extrinsic_param_rotterdam) as file:
#             for line in file:
#                 parts = line.strip().split(",")
#                 if parts[0] == img_id:
#                     for item in parts[1:]:
#                         extrinsic_param.append(item)
#                     return extrinsic_param
#
#
# """
# compute visibility range using skgeometry
# """
# def get_visibility_arrangement(j, id, img_id, city):
#     """ Create a 2D arrangement using skgeom """
#     arr = arrangement.Arrangement()
#
#     extrinsic_param = get_camera_pos(img_id, city)
#
#     camera_pos = Point2(float(extrinsic_param[0]), float(extrinsic_param[1]))
#
#     building_center = compute_center(id, j)
#
#     observer = get_observer(camera_pos, building_center)
#
#     footprint_2d = get_footprint(id, j)
#
#     # polygon = Polygon(footprint_2d)
#     # bbox = polygon.bbox()
#
#     edges = points_2_edges(footprint_2d)
#     for s in edges:
#         arr.insert(s)
#
#     building_arr = arrangement.Arrangement()
#
#     BUILDING_RANGE = 80
#     left_x = observer.x() - BUILDING_RANGE
#     left_y = observer.y() - BUILDING_RANGE
#     right_x = observer.x() + BUILDING_RANGE
#     right_y = observer.y() + BUILDING_RANGE
#
#     viewing_area = [
#         Segment2(Point2(left_x, left_y), Point2(left_x, right_y)),
#         Segment2(Point2(left_x, right_y), Point2(right_x, right_y)),
#         Segment2(Point2(right_x, right_y), Point2(right_x, left_y)),
#         Segment2(Point2(right_x, left_y), Point2(left_x, left_y))
#     ]
#
#     for s in viewing_area:
#         arr.insert(s)
#
#     """
#     Compute the visibility from a specific point inside
#     the arrangement
#     """
#     vs = TriangularExpansionVisibility(arr)
#     q = observer
#     face = arr.find(q)
#     vx = vs.compute_visibility(q, face)
#
#     # Get all edges of Visibility Polygon
#     allEdges = [v.point() for v in vx.vertices]
#
#
# # get visible points
#     visible_points = []
#     for x,y in footprint_2d:
#         position = Polygon(allEdges).oriented_side(Point2(x, y))
#         if position == Sign.ZERO or position == Sign.POSITIVE:
#             visible_points.append([x, y])
#
#     new_arr = arrangement.Arrangement()
#     new_edges = []
#
#     # condition at beginning point and ending point should be different
#     size = len(visible_points)
#     if size == 1:
#         for e in edges:
#             for x, y in visible_points:
#                 if x == e.source().x() and y == e.source().y():
#                     new_edges.append(e)
#                     new_arr.insert(e)
#                 if x == e.target().x() and y == e.target().y():
#                     new_edges.append(e)
#                     new_arr.insert(e)
#     else:
#         for e in edges:
#             for x1, y1 in visible_points:
#                 if x1 == e.source().x() and y1 == e.source().y():
#                     for x2, y2 in visible_points:
#                         if x2 == e.target().x() and y2 == e.target().y():
#                             new_edges.append(e)
#                             new_arr.insert(e)
#
#     """
#     draw visibility range and building outline
#     """
#
#     building_arr = arrangement.Arrangement()
#
#     for s in edges:
#         building_arr.insert(s)
#
#     #
#     # draw.draw(camera_pos, color='slateblue', linewidth = 1)
#     # draw.draw(observer, color='indigo',linewidth = 1)
#     # draw.draw(building_center, color='blue',linewidth = 1)
#     #
#     # for he in building_arr.halfedges:
#     #     draw.draw(he.curve(), color='royalblue')
#     #
#     # for v in vx.halfedges:
#     #     draw.draw(v.curve(), color='peru', visible_point=False)
#     # #
#     # for he in new_arr.halfedges:
#     #     draw.draw(he.curve(), color='gold')
#
#     return new_arr
#
#
#
# def get_visibility_arrangement_2(j, id, img_id, city):
#     """ Create a 2D arrangement using skgeom """
#     arr = arrangement.Arrangement()
#
#     extrinsic_param = get_camera_pos(img_id, city)
#
#     camera_pos = Point2(float(extrinsic_param[0]), float(extrinsic_param[1]))
#
#     building_center = compute_center(id, j)
#
#     observer = get_observer(camera_pos, building_center)
#
#     footprint_2d = get_footprint(id, j)
#
#     # polygon = Polygon(footprint_2d)
#     # bbox = polygon.bbox()
#
#     edges = points_2_edges(footprint_2d)
#     for s in edges:
#         arr.insert(s)
#
#     building_arr = arrangement.Arrangement()
#
#     BUILDING_RANGE = 80
#     left_x = observer.x() - BUILDING_RANGE
#     left_y = observer.y() - BUILDING_RANGE
#     right_x = observer.x() + BUILDING_RANGE
#     right_y = observer.y() + BUILDING_RANGE
#
#     viewing_area = [
#         Segment2(Point2(left_x, left_y), Point2(left_x, right_y)),
#         Segment2(Point2(left_x, right_y), Point2(right_x, right_y)),
#         Segment2(Point2(right_x, right_y), Point2(right_x, left_y)),
#         Segment2(Point2(right_x, left_y), Point2(left_x, left_y))
#     ]
#
#     for s in viewing_area:
#         arr.insert(s)
#
#     """
#     Compute the visibility from a specific point inside
#     the arrangement
#     """
#     vs = TriangularExpansionVisibility(arr)
#     q = observer
#     face = arr.find(q)
#     vx = vs.compute_visibility(q, face)
#
#     # Get all edges of Visibility Polygon
#     allEdges = [v.point() for v in vx.vertices]
#
# # get visible points
#     visible_points_x = []
#     visible_points_y = []
#     visible_points = []
#     for x, y in footprint_2d:
#         position = Polygon(allEdges).oriented_side(Point2(x, y))
#         if position == Sign.ZERO or position == Sign.POSITIVE:
#             visible_points_x.append(x)
#             visible_points_y.append(y)
#             visible_points.append([x, y])
#
#     xmax, ymax, xmin, ymin = find_arr_endpoint(visible_points)
#     return xmax, ymax, xmin, ymin
#
#
# """
# obtain pts from arrangement
# """
# def get_pts_from_arr(arr):
#
#     pt_list = []
#     final_list = []
#     for halfedge in arr.halfedges:
#         pt_list.append([[halfedge.source().point().x(), halfedge.source().point().y()],
#                         [halfedge.target().point().x(), halfedge.target().point().y()]])
#
#     length = int(len(pt_list) / 2)
#     for i in range(length):
#         final_list.append(pt_list[i*2])
#
#     return final_list
#
#
# """
# not used
# """
# def point_in_visibility_polygon(footprint, alledges):
#     """ Check wether a given point is inside the visibility polygon """
#     visible_points = []
#
#     for x,y in footprint:
#         position = alledges.oriented_side(Point2(x, y))
#         if position == Sign.ZERO or position == Sign.POSITIVE:
#             visible_points.append([x, y])
#
#     return visible_points


"""
return facade list (Wallsurface) [[[point1],[point2],[point3],[point4]],[...],[....]] with real coordinates in certain LOD
"""


def get_facades_list(j, id, lod):
    """
    select facades (if type is Wallsurface) of target building in tagert lod
    :param j:
    :param lod:
    :return:
    """
    facades_list = []
    id = id + "-0"
    for each in j["CityObjects"]:
        if each == id:
            if 'geometry' in j['CityObjects'][each]:
                for geom in j['CityObjects'][each]['geometry']:
                    if geom["lod"] == lod:
                        for i in range(len(geom['semantics']['values'][0])):
                            if geom['semantics']['values'][0][i] == 2:
                                for list in geom['boundaries'][0][i]:
                                    facade_list = []
                                    for i in list:
                                        pt = j["vertices"][i]
                                        x = float(pt[0]) * j["transform"]["scale"][0] + j["transform"]["translate"][0]
                                        y = float(pt[1]) * j["transform"]["scale"][1] + j["transform"]["translate"][1]
                                        z = float(pt[2]) * j["transform"]["scale"][2] + j["transform"]["translate"][2]
                                        facade_list.append([x, y, z])
                                    facades_list.append(facade_list)

    return facades_list


"""
get bbox for each facade
"""


def get_bbox(facade):
    max_pt = np.array([0.0, 0.0, -5.0])
    min_pt = np.array([999999.0, 999999.0, 999999.0])
    for point in facade:

        if point[0] > max_pt[0]: max_pt[0] = point[0]
        if point[0] < min_pt[0]: min_pt[0] = point[0]
        if point[1] > max_pt[1]: max_pt[1] = point[1]
        if point[1] < min_pt[1]: min_pt[1] = point[1]
        if point[2] > max_pt[2]:
            max_pt[2] = point[2]
        if point[2] < min_pt[2]:
            min_pt[2] = point[2]

    if_bbox_correct = False

    for point in facade:
        if point[0] == max_pt[0] and point[1] == max_pt[1] and point[2] == max_pt[2]:
            if_bbox_correct = True
        if point[0] == min_pt[0] and point[1] == min_pt[1] and point[2] == min_pt[2]:
            if_bbox_correct = True

    if not if_bbox_correct:
        temp = max_pt[1]
        max_pt[1] = min_pt[1]
        min_pt[1] = temp

    return max_pt, min_pt


"""
return starting point and the ending point of the visible arrangement
"""


def find_endpoints(visible_pts):
    minx = np.amin([visible_pts[0][0], visible_pts[1][0]])
    maxx = np.amax([visible_pts[0][0], visible_pts[1][0]])
    miny = np.amin([visible_pts[0][1], visible_pts[1][1]])
    maxy = np.amax([visible_pts[0][1], visible_pts[1][1]])

    return [minx, miny], [maxx, maxy]


"""
just a backup
"""


def find_endpoints2(visible_pts):
    x = []
    y = []
    for pt in visible_pts:
        x.append(pt[0])
        y.append(pt[1])

    minx = np.amin(np.array(x))
    maxx = np.amax(np.array(x))
    miny = np.amin(np.array(y))
    maxy = np.amax(np.array(y))

    flag = False
    for x, y in visible_pts:
        if x == minx and y == miny:
            flag = True
        if x == maxx and y == maxy:
            flag = True
    if not flag:
        temp = maxy
        maxy = miny
        miny = temp

    return minx, miny, maxx, maxy


#
# """
# compute visible facade using the visibility arrangement we've got before
# return list of visible facade
# """
# def get_visible_facede(arr, facade_list):
#
#     visible_pt_pair = get_pts_from_arr(arr)
#     visible_facade = []
#
#     facade_x = []
#     facade_y = []
#
#     visible_pt_x = []
#     visible_pt_y = []
#
#     visible_group = []
#     for i in range(len(visible_pt_pair)):
#         visible_group.append([])
#
#     for visible_pts in visible_pt_pair:
#         visible_pt_x.append([visible_pts[0][0], visible_pts[1][0]])
#         visible_pt_y.append([visible_pts[0][1], visible_pts[1][1]])
#
#     for facade in facade_list:
#         max_pt, min_pt = get_bbox(facade)
#         # facade_x.append([min_pt[0], max_pt[0]])
#         # facade_y.append([min_pt[1], max_pt[1]])
#
#         fixed_gradient = float(max_pt[0] - min_pt[0]) / float(max_pt[1]/ min_pt[1])
#         # if fixed_gradient > 0:
#         #     fixed_gradient = 1
#         # else:fixed_gradient = -1
#
#         for visible_pts in visible_pt_pair:
#             start_pt, end_pt = find_endpoints(visible_pts)
#             gradient = (float(end_pt[0] - start_pt[0]) / float(end_pt[1] / start_pt[1]))
#             # if gradient>0:
#             #     gradient = 1
#             # else:
#             #     gradient = -1
#             if start_pt[0] <= min_pt[0] and min_pt[0] <= end_pt[0]:
#                 if start_pt[0] <= max_pt[0] and max_pt[0] <= end_pt[0]:
#                     if start_pt[1] <= min_pt[1] and min_pt[1] <= end_pt[1]:
#                         if start_pt[1] <= max_pt[1] and max_pt[1] <= end_pt[1]:
#                             if fixed_gradient-0.1 <= gradient <=fixed_gradient+0.1:
#                                 visible_facade.append(facade)
#                                 visible_group[i].append(facade)
#                                 facade_x.append([min_pt[0], max_pt[0]])
#                                 facade_y.append([min_pt[1], max_pt[1]])
#
#
#     # plot some figures
#     plt.figure(figsize=(6,6))
#
#     for i in range(len(visible_pt_x)):
#         plt.scatter(visible_pt_x[i], visible_pt_y[i], color='gold', linewidths = 3)
#         plt.plot(visible_pt_x[i], visible_pt_y[i], color = 'gold', linewidth = 5)
#         plt.xlabel('X')
#         plt.ylabel('Y')
#
#     # for j in range(len(visible_facade)):
#     #     plt.scatter(visible_facade[j][0], visible_facade[j][1], color='royalblue', s=5)
#     #     plt.plot(visible_facade[j][0], visible_facade[j][1], color = 'royalblue')
#
#     for j in range(len(facade_x)):
#         plt.scatter(facade_x[j], facade_y[j], color='royalblue', s=5)
#         plt.plot(facade_x[j], facade_y[j], color = 'royalblue')
#
#     print("nice")
#
#     return visible_facade,visible_group
#
#
# """
# similar to get_visible_facede, but also return a facade group grouped by arrangement
# """
# def merge_co_planar(arr, facade_list):
#
#     visible_pt_pair = get_pts_from_arr(arr)
#     visible_facade = []
#
#     facade_x = []
#     facade_y = []
#
#     visible_group = []
#     for i in range(len(visible_pt_pair)):
#         visible_group.append([])
#
#     for facade in facade_list:
#         max_pt, min_pt = get_bbox(facade)
#         facade_x.append([min_pt[0], max_pt[0]])
#         facade_y.append([min_pt[1], max_pt[1]])
#         fixed_gradient = float(float(max_pt[0] - min_pt[0]) / float(max_pt[1] / min_pt[1]))
#
#         fixed_centroid = (float((max_pt[0]+min_pt[0])/2), float((max_pt[1]+min_pt[1])/2))
#
#         for i in range(len(visible_pt_pair)):
#             start_pt, end_pt = find_endpoints(visible_pt_pair[i])
#             # gradient = float((float(end_pt[0] - start_pt[0]) / float(end_pt[1] / start_pt[1])))
#             centroid = (float((start_pt[0] + end_pt[0]) / 2), float((start_pt[1] + end_pt[1]) / 2))
#             dist = compute_2d_distance(fixed_centroid,centroid)
#             if start_pt[0] <= min_pt[0] and min_pt[0] <= end_pt[0]:
#                 if start_pt[0] <= max_pt[0] and max_pt[0] <= end_pt[0]:
#                     if start_pt[1] <= min_pt[1] and min_pt[1] <= end_pt[1]:
#                         if start_pt[1] <= max_pt[1] and max_pt[1] <= end_pt[1]:
#                             if dist < 40:
#                                 visible_facade.append(facade)
#                                 visible_group[i].append(facade)
#                                 # plt.scatter([min_pt[0], max_pt[0]], [min_pt[1], max_pt[1]], color='gold', linewidths=3)
#                                 # plt.plot([min_pt[0], max_pt[0]], [min_pt[1], max_pt[1]], color='gold', linewidth=5)
#                                 # print("fixed gradient is: {} and gradient is {}".format(fixed_gradient, gradient))
#
#
#     return visible_facade, visible_group
#

"""
check whether two given facade are coplanar
check by their touching point
"""


def check_coplanar_backup(facade_1, facade_2):
    max_pt1, min_pt1 = get_bbox(facade_1)
    max_pt2, min_pt2 = get_bbox(facade_2)
    if_coplanar = False
    if min_pt1[0] == max_pt2[0] and min_pt1[1] == max_pt2[1] and max_pt1[2] == max_pt2[2]:
        if_coplanar = True
    if min_pt2[0] == max_pt1[0] and min_pt2[1] == max_pt1[1] and max_pt2[2] == max_pt1[2]:
        if_coplanar = True

    return if_coplanar


def check_coplanar(facade_1, facade_2):
    max_pt1, min_pt1 = get_bbox(facade_1)
    max_pt2, min_pt2 = get_bbox(facade_2)
    if_coplanar = False
    if max_pt1[2] == max_pt2[2]:
        if_coplanar = True

    return if_coplanar


"""
part of coplanar test
"""


def coplanar_subset_test(i, j, merge, facade_list):
    flag = False
    for k in range(len(merge[i])):

        for l in range(len(merge[j])):
            if merge[i][k] == merge[j][l]:
                continue
            else:
                facade_1 = facade_list[merge[i][k]]
                facade_2 = facade_list[merge[j][l]]

                if_coplanar = check_coplanar(facade_1, facade_2)

                if flag == False and if_coplanar == False:
                    continue

                elif if_coplanar and flag == False:
                    flag = True
                    print("yes surface {} and {} are coplanar!".format(merge[i], merge[j]))
                    if len(merge[i]) >= len(merge[j]):
                        merge[i].append(j)
                        merge[j] = merge[i]
                    elif len(merge[i]) < len(merge[j]):
                        merge[j].append(i)
                        merge[i] = merge[j]

                    return True

    return False


"""
check coplanar situation of all the visible facades
return coplanar groups
"""


def coplanar(facade_list):
    # initialise id_list
    merge = []
    for i in range(len(facade_list)):
        merge.append([i])

    for i in range(len(merge)):
        for j in range(len(merge)):
            flag = False
            if i == j or merge[i] == merge[j]:
                continue
            elif flag == False:
                flag = coplanar_subset_test(i, j, merge, facade_list)
            else:
                break

    # delete duplicate merged list
    merge_index = [merge[0]]
    for item in merge:
        flag = False
        for existing in merge_index:
            if item == existing:
                flag = True
                break
        if not flag:
            merge_index.append(item)

    merge_surfaces = []
    for item in merge_index:
        merge_surfaces.append([])

    for i in range(len(merge_index)):
        for index in merge_index[i]:
            merge_surfaces[i].append(facade_list[index])

    return merge_index, merge_surfaces


"""
sub-function to show each group of merged facades
"""


def draw_facade_subfunction(group):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for facade in group:
        faces = [[]]
        for i in range(len(facade)):
            faces[0].append(i)

        # get verts for faces
        poly3d = [[facade[vert_id] for vert_id in face] for face in faces]

        # draw the vertices
        x, y, z = zip(*facade)
        ax.scatter(x, y, z)
        # draw the faces
        ax.add_collection3d(Poly3DCollection(poly3d, facecolors='y', linewidths=1, alpha=0.9))
        ax.add_collection3d(Line3DCollection(poly3d, colors='k', linewidths=0.5, linestyles='-'))

    plt.show()


"""
split showing every groups
"""


def draw_facade(facades):
    for group in facades:
        draw_facade_subfunction(group)


"""
draw 3D facade to check the merge result
"""


def draw_all_3d_facades(facades):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for group in facades:
        for facade in group:
            faces = [[]]
            for i in range(len(facade)):
                faces[0].append(i)
            # get verts for faces
            poly3d = [[facade[vert_id] for vert_id in face] for face in faces]
            # draw the vertices
            x, y, z = zip(*facade)
            ax.scatter(x, y, z)
            # draw the faces
            ax.add_collection3d(Poly3DCollection(poly3d, facecolors='y', linewidths=1, alpha=1))
            ax.add_collection3d(Line3DCollection(poly3d, colors='k', linewidths=0.5, linestyles='-'))

        plt.show()


def old_coplanar_save(facade_list):
    # initialise id_list
    merge = []
    for i in range(len(facade_list)):
        merge.append([i])

    for i in range(len(merge)):

        for j in range(len(merge)):

            flag = False
            if i == j or merge[i] == merge[j]:
                continue
            elif flag == False:

                for k in range(len(merge[i])):

                    for l in range(len(merge[j])):
                        if merge[i][k] == merge[j][l]:
                            continue
                        else:
                            facade_1 = facade_list[merge[i][k]]
                            facade_2 = facade_list[merge[j][l]]

                            if_coplanar = check_coplanar(facade_1, facade_2)

                            if flag == False and if_coplanar == False:
                                continue

                            elif if_coplanar and flag == False:
                                flag = True
                                print("yes surface {} and {} are coplanar!".format(merge[i], merge[j]))
                                merge[i].append(j)
                                merge[j] = merge[i]
                                break

                        break

    merge_result = [merge[0]]
    for item in merge:
        flag = False
        for existing in merge_result:
            if item == existing:
                flag = True
                break

        if not flag:
            merge_result.append(item)

    return merge_result


def new_coplanar(facade_list):
    # initialise id_list
    merge = []
    for i in range(len(facade_list)):
        merge.append([i])

    for i in range(len(merge)):
        for j in range(i, len(merge)):
            if i == j or merge[i] == merge[j]:
                continue
            else:
                for k in range(len(merge[i])):
                    for l in range(len(merge[j])):
                        if merge[i][k] == merge[j][l]:
                            continue
                        else:
                            facade_1 = facade_list[merge[i][k]]
                            facade_2 = facade_list[merge[j][l]]

                            if_coplanar = check_coplanar(facade_1, facade_2)
                            if if_coplanar:
                                print("yes surface {} and {} are coplanar!".format(merge[i], merge[j]))
                                merge[i].append(j)
                                merge[j] = merge[i]
                            else:
                                continue

    merge_result = [merge[0]]
    for item in merge:
        flag = False
        for existing in merge_result:
            if item == existing:
                flag = True
                break

        if not flag:
            merge_result.append(item)

    return merge_result


def find_merge_bbox(visible_groups, index):
    max_pt = np.array([0.0, 0.0, -10.0])
    min_pt = np.array([999999.0, 999999.0, 999999.0])

    visible_facades = visible_groups[index]
    for facade in visible_facades:
        max, min = get_bbox(facade)
        if max[0] > max_pt[0]: max_pt[0] = max[0]
        if min[0] < min_pt[0]: min_pt[0] = min[0]
        if max[1] > max_pt[1]: max_pt[1] = max[1]
        if min[1] < min_pt[1]: min_pt[1] = min[1]
        if max[2] > max_pt[2]: max_pt[2] = max[2]
        if min[2] < min_pt[2]: min_pt[2] = min[2]

    if_bbox_correct = False

    for facade in visible_facades:
        for point in facade:
            if point[0] == max_pt[0] and point[1] == max_pt[1] and point[2] == max_pt[2]:
                if_bbox_correct = True
            if point[0] == min_pt[0] and point[1] == min_pt[1] and point[2] == min_pt[2]:
                if_bbox_correct = True

    if not if_bbox_correct:
        temp = max_pt[1]
        max_pt[1] = min_pt[1]
        min_pt[1] = temp

    return max_pt, min_pt


def get_merged_facade_bbox(visible_group, group_id):
    max_pt, min_pt = find_merge_bbox(visible_group, group_id)

    # order: top-left, bottom-left, bottom-right, top-right
    P = np.array([[max_pt[0], max_pt[1], max_pt[2]],
                  [max_pt[0], max_pt[1], min_pt[2]],
                  [min_pt[0], min_pt[1], min_pt[2]],
                  [min_pt[0], min_pt[1], max_pt[2]]])
    return P


def compute_distance(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def compute_2d_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    return float(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))


def get_height_difference(visible_facade):
    zmax = -10
    zmin = 99999
    for facade in visible_facade:
        for point in facade:
            if point[2] > zmax: zmax = point[2]
            if point[2] < zmin: zmin = point[2]
    return zmax, zmin


def find_arr_endpoint(point_lists):
    xmax, ymax = 0., 0.
    xmin, ymin = 999999., 999999.

    for pts in point_lists:
        pt = pts[0]
        if xmax < pt[0]: xmax = float(pt[0])
        if ymax < pt[1]: ymax = float(pt[1])
        if xmin > pt[0]: xmin = float(pt[0])
        if ymin > pt[1]: ymin = float(pt[1])

    if_bbox_correct = False

    for points in point_lists:
        point = points[0]
        if float(point[0]) == float(xmin):
            if float(point[1]) == ymin:
                if_bbox_correct = True
        if float(point[0]) == xmax and float(point[1]) == ymax:
            if_bbox_correct = True

    if not if_bbox_correct:
        temp = ymax
        ymax = ymin
        ymin = temp
    return xmax, ymax, xmin, ymin


# def get_facade_bbox(visible_facade):
#
#     facade_point_list = []
#
#     for facade in visible_facade:
#         for point in facade:
#             facade_point_list.append(point)
#
#     max_pt, min_pt = get_bbox(facade_point_list)
#
#     bbox_list = [[max_pt[0], max_pt[1], max_pt[2]],[max_pt[0], max_pt[1], min_pt[2]],[min_pt[0], min_pt[1], min_pt[2]],[min_pt[0], min_pt[1], max_pt[2]]]
#     return bbox_list


# for .off file!

def create_3d_footprint(each, vertices):
    max_pt = [0.0, 0.0, -100.0]
    min_pt = [999999.0, 999999.0, 999999.0]

    for facade in each:
        for i in facade:

            if vertices[i][0] > max_pt[0]: max_pt[0] = vertices[i][0]
            if vertices[i][0] < min_pt[0]: min_pt[0] = vertices[i][0]
            if vertices[i][1] > max_pt[1]: max_pt[1] = vertices[i][1]
            if vertices[i][1] < min_pt[1]: min_pt[1] = vertices[i][1]
            if vertices[i][2] > max_pt[2]: max_pt[2] = vertices[i][2]
            if vertices[i][2] < min_pt[2]: min_pt[2] = vertices[i][2]

    if_bbox_correct = False

    for facade in each:
        for i in facade:
            if vertices[i][0] == max_pt[0] and vertices[i][1] == max_pt[1] and vertices[i][2] == max_pt[2]:
                if_bbox_correct = True
            if vertices[i][0] == min_pt[0] and vertices[i][1] == min_pt[1] and vertices[i][2] == min_pt[2]:
                if_bbox_correct = True

    if not if_bbox_correct:
        temp = max_pt[1]
        max_pt[1] = min_pt[1]
        min_pt[1] = temp

    return [max_pt, [max_pt[0], max_pt[1], min_pt[2]], min_pt, [min_pt[0], min_pt[1], max_pt[2]]]


def get_off_3Dfootprint(grouped_faces, vertices):
    count = 0
    total_3Dfootprint = []
    else_polygon = []
    for each in grouped_faces:
        area_sum = 0.

        for tri in each:
            area = triangle_area(tri, vertices)
            area_sum += area

        if area_sum > 5:
            each_3Dfootprint = create_3d_footprint(each, vertices)
            total_3Dfootprint.append(each_3Dfootprint)
        else:
            for tri in each:
                else_polygon.append(tri)
        count += 1

    return total_3Dfootprint, else_polygon


def create_2d_footprint(each, vertices):
    max_pt = np.array([0.0, 0.0, -100.0])
    min_pt = np.array([999999.0, 999999.0, 999999.0])

    for facade in each:
        for i in facade:

            if vertices[i][0] > max_pt[0]: max_pt[0] = vertices[i][0]
            if vertices[i][0] < min_pt[0]: min_pt[0] = vertices[i][0]
            if vertices[i][1] > max_pt[1]: max_pt[1] = vertices[i][1]
            if vertices[i][1] < min_pt[1]: min_pt[1] = vertices[i][1]
            if vertices[i][2] > max_pt[2]: max_pt[2] = vertices[i][2]
            if vertices[i][2] < min_pt[2]: min_pt[2] = vertices[i][2]

    if_bbox_correct = False

    for facade in each:
        for i in facade:
            if vertices[i][0] == max_pt[0] and vertices[i][1] == max_pt[1] and vertices[i][2] == max_pt[2]:
                if_bbox_correct = True
            if vertices[i][0] == min_pt[0] and vertices[i][1] == min_pt[1] and vertices[i][2] == min_pt[2]:
                if_bbox_correct = True

    if not if_bbox_correct:
        temp = max_pt[1]
        max_pt[1] = min_pt[1]
        min_pt[1] = temp

    return [max_pt[0], max_pt[1]], [min_pt[0], min_pt[1]]


def get_off_2Dfootprint(grouped_faces, vertices):
    total_2Dfootprint = []
    count = 0

    for each in grouped_faces:

        pt_max, pt_min = create_2d_footprint(each, vertices)
        dist = compute_2d_distance(pt_max, pt_min)
        area_sum = 0
        for face in each:
            area_sum += triangle_area(face, vertices)

        if area_sum > 2 and dist > 5:
            total_2Dfootprint.append([pt_max, pt_min])

        count += 1

    return total_2Dfootprint


def footprint_in_image_range(image_range, footprint_2d):
    inrange_list = []
    for i in range(len(footprint_2d)):
        if image_range[0][0] <= footprint_2d[i][0][0] <= image_range[1][0]:
            if image_range[0][0] <= footprint_2d[i][1][0] <= image_range[1][0]:
                if image_range[0][1] <= footprint_2d[i][0][1] <= image_range[1][1]:
                    if image_range[0][1] <= footprint_2d[i][1][1] <= image_range[1][1]:
                        inrange_list.append(i)

    return inrange_list


def compute_off_centroid(footprint_2d):
    pass


def draw_single_rectangle(rectangle):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = [p[0] for p in rectangle]
    y = [p[1] for p in rectangle]
    z = [p[2] for p in rectangle]
    ax.plot(x, y, z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


def draw_multi_rectangle(rectangles):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for rectangle in rectangles:
        x = [p[0] for p in rectangle]
        y = [p[1] for p in rectangle]
        z = [p[2] for p in rectangle]
        ax.plot(x, y, z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


def draw_footprint(points):
    # extract x and y coordinates from the list of points
    x = [[point[0] for point in sub_list] for sub_list in points]
    y = [[point[1] for point in sub_list] for sub_list in points]

    # plot the points
    for i in range(len(x)):
        plt.plot(x[i], y[i], 'o-')

    # add labels and title    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Points')

    # show the plot
    plt.show()


def is_point_on_triangle_plane(point, triangle):
    # Calculate edge vectors of the triangle
    edge1 = np.subtract(triangle[1], triangle[0])
    edge2 = np.subtract(triangle[2], triangle[0])

    # Calculate the normal vector of the triangle's plane
    normal = np.cross(edge1, edge2)

    # Calculate a vector connecting the point to one of the triangle's vertices
    connecting_vector = np.subtract(point, triangle[0])

    # Calculate the dot product of the normal and the connecting vector
    dot_product = np.dot(normal, connecting_vector)

    # Check if the dot product is close to zero
    return np.isclose(dot_product, 0)


def calculate_y_on_triangle_plane(x, z, triangle):
    # Compute edge vectors of the triangle
    edge1 = np.subtract(triangle[1], triangle[0])
    edge2 = np.subtract(triangle[2], triangle[0])

    # Compute the normal vector of the triangle's plane
    normal = np.cross(edge1, edge2)
    nx, ny, nz = normal

    # Calculate the plane equation constant (D)
    D = -np.dot(normal, triangle[0])

    # Substitute the known x and z values into the plane equation and solve for y
    if ny == 0:
        raise ValueError("The normal vector's y component is zero, cannot solve for y.")
    y = (-D - nx * x - nz * z) / ny

    return y


def inner_corner(point_a, point_b, point_c):
    # Calculate vectors on the plane of the rectangle
    vector1 = point_b - point_a
    vector2 = point_c - point_a

    # Find the normal vector of the rectangle
    normal_vector = np.cross(vector1, vector2)

    # Normalize the normal vector
    normalized_normal_vector = normal_vector / np.linalg.norm(normal_vector)

    # Calculate point E
    point_e = point_a + 1 * normalized_normal_vector

    perpendicular(point_a, point_b, point_c, point_e)

    # Output point E
    print("Coordinates of point E:", point_e)
    return point_e


def perpendicular(point_a, point_b, point_c, point_e):
    # Calculate direction vector of the line AE
    direction_vector = point_e - point_a

    # Calculate vectors on the plane of the rectangle
    vector1 = point_b - point_a
    vector2 = point_c - point_a

    # Find the normal vector of the plane
    normal_vector = np.cross(vector1, vector2)

    # Calculate the dot product between the direction vector and the normal vector
    dot_product = np.dot(direction_vector, normal_vector)

    # Check if the dot product is close to zero
    epsilon = 1e-6
    is_perpendicular = abs(dot_product) < epsilon

    # Output result
    print("Is line AE perpendicular to the plane ABCD?", is_perpendicular)


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def inner_corner_cal_and_val(point_a, point_b, point_c):
    # Calculate vectors on the plane of the rectangle
    vector1 = point_b - point_a
    vector2 = point_c - point_a

    # Find the normal vector of the plane
    normal_vector = np.cross(vector1, vector2)

    # Normalize the normal vector
    normal_unit_vector = unit_vector(normal_vector)

    # Calculate point E such that AE is perpendicular to the plane and AE = 1 unit
    distance = -0.2
    point_e = point_a + distance * normal_unit_vector

    # Calculate direction vector of the line AE
    direction_vector = point_e - point_a

    # Calculate the dot product between the direction vector and the normal vector
    dot_product = np.dot(direction_vector, (point_c - point_b))

    # Check if the dot product is close to zero
    epsilon = 1e-6
    is_perpendicular = abs(dot_product) < epsilon

    # Output results
    print("Point E:", point_e)
    print("Is line AE perpendicular to the plane ABCD?", is_perpendicular)
    return point_e


def scalar_triple_product(a, b, c):
    return np.abs(np.dot(a, np.cross(b, c)))


def coplanar_check(point_a, point_b, point_c):
    # Calculate vectors AB, AC, and BC
    vector_ab = point_b - point_a
    vector_ac = point_c - point_a
    vector_bc = point_c - point_b

    # Calculate the scalar triple product
    volume = scalar_triple_product(vector_ab, vector_ac, vector_bc)

    # Check if the volume is close to zero
    epsilon = 1e-6
    is_coplanar = volume < epsilon

    # Output result
    print("Are points A, B, and C coplanar?", is_coplanar)


def draw_scatter(x, y):
    # Reshape x array to a 2D array
    x_array = np.array(x).reshape(-1, 1)

    # Fit a linear regression model
    lr = LinearRegression()
    lr.fit(x_array, y)

    # Calculate predicted values and R^2
    y_pred = lr.predict(x_array)
    r2 = r2_score(y, y_pred)

    # Create the scatter plot and trend line
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.plot(x, y_pred, color='red', linestyle='--', label='Trend line')

    TimesNewRoman = FontProperties()
    TimesNewRoman.set_family('Times New Roman')

    # Annotate the plot with the regression function and R^2 value
    regression_function = f"y = {lr.coef_[0]:.2f}x + {lr.intercept_:.2f}"

    ax.annotate(regression_function, xy=(0.5, 0.9), xycoords='axes fraction', fontsize=12, fontproperties=TimesNewRoman)
    ax.annotate(f"R^2 = {r2:.2f}", xy=(0.5, 0.8), xycoords='axes fraction', fontsize=12, fontproperties=TimesNewRoman)

    # Set axis labels font
    ax.set_xlabel('x', fontproperties=TimesNewRoman, fontsize=12)
    ax.set_ylabel('y', fontproperties=TimesNewRoman, fontsize=12)

    plt.legend()
    plt.show()


def facade_offset(new_facede_2d, img_id):
    img_type = img_id.strip().split("_")[0]

    if img_type == "405":
        for i in range(len(new_facede_2d)):
            new_facede_2d[i][0] = new_facede_2d[i][0] * 1.0063 - 33.253
            new_facede_2d[i][1] = new_facede_2d[i][1] * 1.0132 - 23.084
    elif img_type == "402":
        for i in range(len(new_facede_2d)):
            new_facede_2d[i][0] = new_facede_2d[i][0] * 0.9911 + 143.03
            new_facede_2d[i][1] = new_facede_2d[i][1] * 0.9899 + 51.635

    elif img_type == "404":
        for i in range(len(new_facede_2d)):
            new_facede_2d[i][0] = new_facede_2d[i][0] * 0.9932 + 128.6
            new_facede_2d[i][1] = new_facede_2d[i][1] * 0.9895 + 108.61

    elif img_type == "403":
        for i in range(len(new_facede_2d)):
            new_facede_2d[i][0] = new_facede_2d[i][0] * 1.0155 + 265.09
            new_facede_2d[i][1] = new_facede_2d[i][1] * 1.0119 - 914.57

    return new_facede_2d


def order_rectangle_points(points):
    sorted_points = sorted(points, key=lambda p: (p[0], p[2]))  # Sort by Y, then by X

    top_left = sorted_points[1]
    top_right = sorted_points[3]
    bottom_left = sorted_points[0]
    bottom_right = sorted_points[2]

    return [top_left, bottom_left, bottom_right, top_right]


"""integration for single facade/rectangle!"""


def integration(rectangle, w, d, facade_img_id):

    footprint_3d = []
    new_window_3d = []
    new_door_3d = []
    connecting_wall = []

    mybbox = rectangle
    visual = []
    visual.append(rectangle)
    rect = order_rectangle_points(mybbox)
    print(rect)
    # draw_single_rectangle(rect)

    image = cv2.imread(facade_img_id)
    height, width, channels = image.shape

    dist_xy = compute_distance(mybbox[1], mybbox[2])

    dist_x = mybbox[0][0] - mybbox[2][0]
    dist_y = mybbox[0][1] - mybbox[2][1]
    dist_z = mybbox[0][2] - mybbox[1][2]

    for i in range(len(w)):
        if len(w[i]) > 0:
            for window in w[i]:
                window_3d = []
                xy_ratio = dist_xy / width
                z_ratio = dist_z / height

                top_left_3d = [rect[0][0] + (window[0] / width) * dist_x,
                               rect[0][1] + (window[0] / width) * dist_y,
                               rect[0][2] - window[1] * z_ratio]

                bottom_left_3d = [top_left_3d[0], top_left_3d[1], top_left_3d[2] - window[3] * (dist_z / height)]

                top_right_3d = [top_left_3d[0] + window[2] * (dist_x / width),
                                top_left_3d[1] + window[2] * (dist_y / width),
                                top_left_3d[2]]

                bottom_right_3d = [top_right_3d[0], top_right_3d[1], bottom_left_3d[2]]

                # window_3d.append([top_left_3d, bottom_left_3d, bottom_right_3d, top_right_3d])
                window_3d.append([top_right_3d, bottom_right_3d, bottom_left_3d, top_left_3d])

                # visual.append(window_3d[0])
                # draw_multi_rectangle(visual)
                footprint_3d.append(window_3d[0])
        else:
            continue

    for window in footprint_3d:
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
        connecting_wall.append(connecting_0)
        connecting_wall.append(connecting_1)
        connecting_wall.append(connecting_2)
        connecting_wall.append(connecting_3)

    single_3d_doors = []
    for i in range(len(d)):
        if len(d[i]) > 0:
            for door in d[i]:
                door_3d = []
                xy_ratio = dist_xy / width
                z_ratio = dist_z / height

                top_left_3d = [rect[0][0] + (door[0] / width) * dist_x,
                               rect[0][1] + (door[0] / width) * dist_y,
                               rect[0][2] - door[1] * z_ratio]

                bottom_left_3d = [top_left_3d[0], top_left_3d[1], top_left_3d[2] - door[3] * (dist_z / height)]

                top_right_3d = [top_left_3d[0] + door[2] * (dist_x / width),
                                top_left_3d[1] + door[2] * (dist_y / width),
                                top_left_3d[2]]

                bottom_right_3d = [top_right_3d[0], top_right_3d[1], bottom_left_3d[2]]

                # window_3d.append([top_left_3d, bottom_left_3d, bottom_right_3d, top_right_3d])
                door_3d.append([top_right_3d, bottom_right_3d, bottom_left_3d, top_left_3d])

                # visual.append(window_3d[0])
                # draw_multi_rectangle(visual)
                single_3d_doors.append(door_3d[0])
                footprint_3d.append(door_3d[0])
        else:
            continue

    for door in single_3d_doors:
        # new window
        new_corner0 = inner_corner_cal_and_val(np.array(door[0]), np.array(door[1]), np.array(door[2]))
        new_corner1 = inner_corner_cal_and_val(np.array(door[1]), np.array(door[2]), np.array(door[3]))
        new_corner2 = inner_corner_cal_and_val(np.array(door[2]), np.array(door[3]), np.array(door[0]))
        new_corner3 = inner_corner_cal_and_val(np.array(door[3]), np.array(door[0]), np.array(door[1]))

        new_door_surface = [new_corner0, new_corner1, new_corner2, new_corner3]

        # new connecting surfaces
        connecting_0 = [door[0], new_corner0, new_corner3, door[3]]
        connecting_1 = [door[1], new_corner1, new_corner0, door[0]]
        connecting_2 = [door[2], new_corner2, new_corner1, door[1]]
        connecting_3 = [door[3], new_corner3, new_corner2, door[2]]

        new_door_3d.append(new_door_surface)
        connecting_wall.append(connecting_0)
        connecting_wall.append(connecting_1)
        connecting_wall.append(connecting_2)
        connecting_wall.append(connecting_3)

    return footprint_3d, new_window_3d, new_door_3d, connecting_wall


def write_non_wall(j, non_wall_surface, vertices):

    json_vertex_count = len(j['vertices'])

    for each in j["CityObjects"]:
        for geom in j["CityObjects"][each]['geometry']:
            if geom["lod"] == 2.2:
                for wall in non_wall_surface:
                    single_boundary = []
                    for vertex in wall:
                        print(vertex)
                        print([vertices[vertex][0], vertices[vertex][1], vertices[vertex][2]])
                        j["vertices"].append([vertices[vertex][0], vertices[vertex][1], vertices[vertex][2]])
                        single_boundary.append(json_vertex_count)
                        json_vertex_count += 1
                    geom['boundaries'].append([single_boundary])
                    geom['semantics']['values'].append(1)
    return j


def write_gap(j, non_wall_surface, vertices):

    json_vertex_count = len(j['vertices'])

    for each in j["CityObjects"]:
        for geom in j["CityObjects"][each]['geometry']:
            if geom["lod"] == 2.2:
                for wall in non_wall_surface:
                    single_boundary = []
                    for vertex in wall:
                        print(vertex)
                        print([vertices[vertex][0], vertices[vertex][1], vertices[vertex][2]])
                        j["vertices"].append([vertices[vertex][0], vertices[vertex][1], vertices[vertex][2]])
                        single_boundary.append(json_vertex_count)
                        json_vertex_count += 1
                    geom['boundaries'].append([single_boundary])
                    geom['semantics']['values'].append(2)
    return j


def output_openings_wall(j, rectangle, single_3d_windows, new_window_3d, new_door_3d, connecting_wall):
    json_vertex_count = len(j['vertices'])
    for each in j["CityObjects"]:
        for geom in j["CityObjects"][each]['geometry']:
            if geom["lod"] == 2.2:
                facade_boundaries = []
                single_boundary = []
                # rect = order_rectangle_points(rectangle)
                for vertex in rectangle:
                    j["vertices"].append([vertex[0], vertex[1], vertex[2]])
                    single_boundary.append(json_vertex_count)
                    json_vertex_count += 1
                facade_boundaries.append(single_boundary)
                geom['semantics']['values'].append(2)

                for window in single_3d_windows:
                    win_boundaries = []
                    for vertex in window:
                        j["vertices"].append([vertex[0], vertex[1], vertex[2]])
                        win_boundaries.append(json_vertex_count)
                        json_vertex_count += 1
                    facade_boundaries.append(win_boundaries)
                geom['boundaries'].append(facade_boundaries)

                for each in connecting_wall:
                    single_boundary = []
                    for vertex in each:
                        j['vertices'].append([vertex[0], vertex[1], vertex[2]])
                        single_boundary.append(json_vertex_count)
                        json_vertex_count += 1
                    geom['boundaries'].append([single_boundary])
                    geom['semantics']['values'].append(2)

                for each_window in new_window_3d:
                    single_boundary = []
                    for vertex in each_window:
                        j['vertices'].append([vertex[0], vertex[1], vertex[2]])
                        single_boundary.append(json_vertex_count)
                        json_vertex_count += 1
                    geom['boundaries'].append([single_boundary])
                    geom['semantics']['values'].append(4)

                for each_door in new_door_3d:
                    single_boundary = []
                    for vertex in each_door:
                        j['vertices'].append([vertex[0], vertex[1], vertex[2]])
                        single_boundary.append(json_vertex_count)
                        json_vertex_count += 1
                    geom['boundaries'].append([single_boundary])
                    geom['semantics']['values'].append(5)

    # json_object = json.dumps(j, indent=4)
    # with open('output_1.json', 'w') as json_file:
    #     json_file.write(json_object)

    return j


def output_off(rectangle, new_window_3d):
    i = 0
    index_list = []
    fo = open("mayday_wall.off", "w")

    for vertex in rectangle:
        fo.write("{} {} {}{}".format(vertex[0], vertex[1], vertex[2], '\n'))
    index_list.append([i, i + 1, i + 2, i + 3])
    i += 4

    for windows in new_window_3d:
        for vertex in windows:
            fo.write("{} {} {}{}".format(vertex[0], vertex[1], vertex[2], '\n'))
        index_list.append([i, i + 1, i + 2, i + 3])
        i += 4

    for index in index_list:
        fo.write("{} {} {} {} {}{}".format(4, index[0], index[1], index[2], index[3], '\n'))
    fo.close()


def no_openings_walls(id_count, image_index):

    filter_index = []
    for i in range(id_count):
        flag = False
        for opening_img in image_index:
            parts = opening_img.strip().split('_')[4]
            index = int(parts.strip().split('.')[0])
            print(i)
            print(index)
            if i == index:
                flag = True
                continue

        if not flag:
            filter_index.append(i)
        elif flag:
            print("hey")

    return filter_index


def add_walls(j, filter_result, rectangles):

    json_vertex_count = len(j['vertices'])
    for each in j["CityObjects"]:
        for geom in j["CityObjects"][each]['geometry']:
            if geom["lod"] == 2.2:
                for i in filter_result:
                    single_boundary = []
                    rect = order_rectangle_points(rectangles[i])
                    for vertex in rect:
                        j["vertices"].append([vertex[0], vertex[1], vertex[2]])
                        single_boundary.append(json_vertex_count)
                        json_vertex_count += 1
                    geom['boundaries'].append([single_boundary])
                    geom['semantics']['values'].append(2)

    return j


if __name__ == '__main__':

    """read json template file"""
    json_template = f'../../visualization/template.json'
    with open(json_template, "r") as json_file:
        json_data = json_file.read()
        j = json.loads(json_data)

    """read .off file"""
    # offpath = f"../../visualization/building10_wall.off"
    #
    # vertices = read_mesh_vertex(offpath)
    # faces, colors = read_mesh_faces_1(offpath)
    # print("finish read")
    #
    # wallsurface, new_color_list1, non_wall_surface1, non_wall_color1 = wallsurface_filter_bynormal(faces, colors, vertices)
    # grouped_faces, new_color_list = merge_surface(faces, colors, vertices)
    # grouped_non_wall, non_wall_color = merge_surface(non_wall_surface1, non_wall_color1, vertices)
    #
    # """obtain 3D rectangles"""
    # rectangles = get_off_3Dfootprint(grouped_faces, vertices)
    # rect_for_projection = rectangles.copy()
    path = f"../../visualization/building10_wall.off"

    vertices = read_mesh_vertex(path)
    faces, colors = read_mesh_faces_1(path)
    print("finish read")

    wallsurface, new_color_list1 = wallsurface_filter_bynormal(faces, colors, vertices)
    grouped_faces, new_color_list = merge_surface (faces, colors, vertices)
    non_wall_surface, non_wall_color = obtain_non_wall()

    rectangles, else_polygon = get_off_3Dfootprint(grouped_faces, vertices)
    rect_for_projection = rectangles.copy()

    # draw_rectangles(rectangles)

    img_id = '404_0030_00131330.tif'

    pts_groundtruthlist = []
    pts_original = []
    id_count = 0
    img_list = []
    for rectangle in rectangles:
        rect = rectangle.copy()

        new_facede_2d = projection(img_id, rect)

        pts_original.append(np.array(new_facede_2d))

        new_facede_2d = facade_offset(new_facede_2d, img_id)

        pts = np.array(new_facede_2d, np.int32)

        name = "../../collection/calibration_404_0030_00131330_" + str(id_count) + ".jpg"
        img_list.append(name)
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

    """for save and view the projection result """
    # image = cv2.imread('/Users/yitongxia/Desktop/pipeline/' + img_id)
    # for rectangle in pts_original:
    #     cv2.polylines(image, [rectangle], True, (0, 0, 255), 5)
    # for rectangle in pts_groundtruthlist:
    #     cv2.polylines(image, [rectangle], True, (255, 0, 0), 5)
    # cv2.imwrite(f"404_0030_00131330_optimization_many_buildings.jpg", image)
    # cv2.imshow("Image with Rectangles", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    visible_facade = []

    cocofile = f'../../import_Almere/facade-2.json'
    image_index, window_by_img, d = get_openings(cocofile)
    size_regularizations(image_index, window_by_img, d)

    for i in range(len(image_index)):
        parts = image_index[i].strip().split("_")
        img_direction = parts[1]
        rectangle_index = int(parts[4].strip().split(".")[0])
        rectangle = rectangles[rectangle_index]
        img_name = "../../collection/" + image_index[i]
        single_3d_windows, new_window_3d, new_door_3d, connecting_wall = integration(rectangle, window_by_img[i], d[i], img_name)
        j = output_openings_wall(j, rectangle, single_3d_windows, new_window_3d, new_door_3d, connecting_wall)

    filter_result = no_openings_walls(id_count, image_index)
    j = write_gap(j, else_polygon, vertices)
    j = add_walls(j, filter_result, rectangles)
    j = write_non_wall(j, non_wall_surface, vertices)

    json_object = json.dumps(j, indent=4)
    with open('complete.json', 'w') as json_file:
        json_file.write(json_object)



