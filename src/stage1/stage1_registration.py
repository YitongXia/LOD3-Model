import json

from matplotlib import pyplot as plt
import numpy as np

np.set_printoptions(suppress=True)
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from extract_texture import *
from perspective_projection import *
import math
from surface_merge import *
from src.layout import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from matplotlib.font_manager import FontProperties
from integration import *
from PIL import Image
from scipy import stats

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
    min_pt = [9999999.0, 9999999.0, 999999.0]

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


def inner_corner_cal_and_val(point_a, point_b, point_c, img_type):
    # Calculate vectors on the plane of the rectangle
    vector1 = point_b - point_a
    vector2 = point_c - point_a

    # Find the normal vector of the plane
    normal_vector = np.cross(vector1, vector2)

    # Normalize the normal vector
    normal_unit_vector = unit_vector(normal_vector)

    # Calculate point E such that AE is perpendicular to the plane and AE = 1 unit
    distance = 0
    if  img_type == "404":
        distance = 0.2
    else:
        distance = 0.2
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


def registration(new_facede_2d, slope_x, intercept_x, slope_y, intercept_y):
    for i in range(len(new_facede_2d)):
        new_facede_2d[i][0] = new_facede_2d[i][0] * slope_x + intercept_x
        new_facede_2d[i][1] = new_facede_2d[i][1] * slope_y + intercept_y

    return new_facede_2d


def registration_model():
    regis_model = []
    with open('../../visualization/registration.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(" ")
            regis_model.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])

    return regis_model


def facade_offset(new_facede_2d, img_id, regis_model):
    img_type = img_id.strip().split("_")[0]

    if img_type == "405":
        for i in range(len(new_facede_2d)):
            new_facede_2d[i][0] = new_facede_2d[i][0] * regis_model[3][0] + regis_model[3][1]
            new_facede_2d[i][1] = new_facede_2d[i][1] * regis_model[3][2] + regis_model[3][3]
    elif img_type == "402":
        for i in range(len(new_facede_2d)):
            new_facede_2d[i][0] = new_facede_2d[i][0] * regis_model[0][0] + regis_model[0][1]
            new_facede_2d[i][1] = new_facede_2d[i][1] * regis_model[0][2] + regis_model[0][3]

    elif img_type == "404":
        for i in range(len(new_facede_2d)):
            new_facede_2d[i][0] = new_facede_2d[i][0] * regis_model[2][0] + regis_model[2][1]
            new_facede_2d[i][1] = new_facede_2d[i][1] * regis_model[2][2] + regis_model[2][3]

    elif img_type == "403":
        for i in range(len(new_facede_2d)):
            new_facede_2d[i][0] = new_facede_2d[i][0] * regis_model[1][0] + regis_model[1][1]
            new_facede_2d[i][1] = new_facede_2d[i][1] * regis_model[1][2] + regis_model[1][3]

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
    print(mybbox)
    rect = order_rectangle_points(mybbox)

    print(rect)
    # draw_single_rectangle(rect)

    image = cv2.imread(facade_img_id)
    height, width, channels = image.shape

    dist_xy = compute_distance(mybbox[1], mybbox[2])

    img_type = facade_img_id.strip().split("_")[1]

    if img_type == "402" or img_type == "403":

        dist_x = mybbox[2][0] - mybbox[0][0]
        dist_y = mybbox[2][1] - mybbox[0][1]
        dist_z = mybbox[0][2] - mybbox[1][2]

        for i in range(len(w)):
            if len(w[i]) > 0:
                for window in w[i]:
                    window_3d = []
                    xy_ratio = dist_xy / width
                    z_ratio = dist_z / height

                    top_left_3d = [rect[3][0] + (window[0] / width) * dist_x,
                                   rect[3][1] + (window[0] / width) * dist_y,
                                   rect[3][2] - window[1] * z_ratio]

                    bottom_left_3d = [top_left_3d[0], top_left_3d[1], top_left_3d[2] - window[3] * (dist_z / height)]

                    top_right_3d = [top_left_3d[0] + window[2] * (dist_x / width),
                                    top_left_3d[1] + window[2] * (dist_y / width),
                                    top_left_3d[2]]

                    bottom_right_3d = [top_right_3d[0], top_right_3d[1], bottom_left_3d[2]]

                    # window_3d.append([top_left_3d, bottom_left_3d, bottom_right_3d, top_right_3d])
                    window_3d.append([top_right_3d, bottom_right_3d, bottom_left_3d, top_left_3d])

                    visual.append(window_3d[0])
                    # draw_multi_rectangle(visual)
                    footprint_3d.append(window_3d[0])
            else:
                continue

    elif img_type == "405" or img_type == "404":
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

                    visual.append(window_3d[0])
                    # draw_multi_rectangle(visual)
                    footprint_3d.append(window_3d[0])
            else:
                continue

    # print("see magic")
    # draw_multi_rectangle(visual)
    # print("hey")

    for window in footprint_3d:
        # new window
        new_corner0 = inner_corner_cal_and_val(np.array(window[0]), np.array(window[1]), np.array(window[2]), img_type)
        new_corner1 = inner_corner_cal_and_val(np.array(window[1]), np.array(window[2]), np.array(window[3]), img_type)
        new_corner2 = inner_corner_cal_and_val(np.array(window[2]), np.array(window[3]), np.array(window[0]), img_type)
        new_corner3 = inner_corner_cal_and_val(np.array(window[3]), np.array(window[0]), np.array(window[1]), img_type)

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

    if img_type == "402" or img_type == "403":
        for i in range(len(d)):
            dist_x = mybbox[2][0] - mybbox[0][0]
            dist_y = mybbox[2][1] - mybbox[0][1]
            dist_z = mybbox[0][2] - mybbox[1][2]
            if len(d[i]) > 0:
                for door in d[i]:
                    door_3d = []
                    xy_ratio = dist_xy / width
                    z_ratio = dist_z / height

                    top_left_3d = [rect[3][0] + (door[0] / width) * dist_x,
                                   rect[3][1] + (door[0] / width) * dist_y,
                                   rect[3][2] - door[1] * z_ratio]

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

    elif img_type == "405" or img_type == "404":
        for i in range(len(d)):
            dist_x = mybbox[0][0] - mybbox[2][0]
            dist_y = mybbox[0][1] - mybbox[2][1]
            dist_z = mybbox[0][2] - mybbox[1][2]
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
        new_corner0 = inner_corner_cal_and_val(np.array(door[0]), np.array(door[1]), np.array(door[2]), img_type)
        new_corner1 = inner_corner_cal_and_val(np.array(door[1]), np.array(door[2]), np.array(door[3]), img_type)
        new_corner2 = inner_corner_cal_and_val(np.array(door[2]), np.array(door[3]), np.array(door[0]), img_type)
        new_corner3 = inner_corner_cal_and_val(np.array(door[3]), np.array(door[0]), np.array(door[1]), img_type)

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
                    if_ground = False
                    for vertex in wall:
                        print(vertices[vertex-1])
                        if vertices[vertex-1][2] < 0:
                            if_ground = True
                        j["vertices"].append([vertices[vertex][0], vertices[vertex][1], vertices[vertex][2]])
                        single_boundary.append(json_vertex_count)
                        json_vertex_count += 1
                    geom['boundaries'].append([single_boundary])
                    if if_ground:
                        geom['semantics']['values'].append(0)
                    else:
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


def output_openings_wall_original(j, rectangle, single_3d_windows, new_window_3d, new_door_3d, connecting_wall):
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
                new_boundary = [single_boundary[3], single_boundary[2], single_boundary[1], single_boundary[0]]
                facade_boundaries.append(new_boundary)
                geom['semantics']['values'].append(2)

                for window in single_3d_windows:
                    win_boundaries = []
                    for vertex in window:
                        j["vertices"].append([vertex[0], vertex[1], vertex[2]])
                        win_boundaries.append(json_vertex_count)
                        json_vertex_count += 1

                    new_boundary = [win_boundaries[3], win_boundaries[2], win_boundaries[1], win_boundaries[0]]
                    facade_boundaries.append(new_boundary)
                geom['boundaries'].append(facade_boundaries)

                for each in connecting_wall:
                    single_boundary = []
                    for vertex in each:
                        j['vertices'].append([vertex[0], vertex[1], vertex[2]])
                        single_boundary.append(json_vertex_count)
                        json_vertex_count += 1

                    new_boundary = [single_boundary[3], single_boundary[2], single_boundary[1], single_boundary[0]]
                    geom['boundaries'].append([new_boundary])
                    geom['semantics']['values'].append(2)

                for each_window in new_window_3d:
                    single_boundary = []
                    for vertex in each_window:
                        j['vertices'].append([vertex[0], vertex[1], vertex[2]])
                        single_boundary.append(json_vertex_count)
                        json_vertex_count += 1

                    new_boundary = [single_boundary[3], single_boundary[2], single_boundary[1], single_boundary[0]]
                    geom['boundaries'].append([new_boundary])
                    geom['semantics']['values'].append(4)

                for each_door in new_door_3d:
                    single_boundary = []
                    for vertex in each_door:
                        j['vertices'].append([vertex[0], vertex[1], vertex[2]])
                        single_boundary.append(json_vertex_count)
                        json_vertex_count += 1

                    new_boundary = [single_boundary[3], single_boundary[2], single_boundary[1], single_boundary[0]]
                    geom['boundaries'].append([new_boundary])
                    geom['semantics']['values'].append(5)

    # json_object = json.dumps(j, indent=4)
    # with open('output_1.json', 'w') as json_file:
    #     json_file.write(json_object)

    return j


def output_openings_wall(j, rectangle, single_3d_windows, new_window_3d, new_door_3d, connecting_wall, img_direction):
    json_vertex_count = len(j['vertices'])
    for each in j["CityObjects"]:
        for geom in j["CityObjects"][each]['geometry']:
            if geom["lod"] == 2.2:
                facade_boundaries = []
                single_boundary = []
                # rect = order_rectangle_points(rectangle)
                if img_direction == "405" or img_direction == "402":
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

                        new_boundary = [win_boundaries[3], win_boundaries[2], win_boundaries[1], win_boundaries[0]]
                        facade_boundaries.append(new_boundary)
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
                        new_boundary = [single_boundary[3], single_boundary[2], single_boundary[1], single_boundary[0]]
                        geom['boundaries'].append([new_boundary])
                        geom['semantics']['values'].append(4)

                    for each_door in new_door_3d:
                        single_boundary = []
                        for vertex in each_door:
                            j['vertices'].append([vertex[0], vertex[1], vertex[2]])
                            single_boundary.append(json_vertex_count)
                            json_vertex_count += 1
                        new_boundary = [single_boundary[3], single_boundary[2], single_boundary[1], single_boundary[0]]
                        geom['boundaries'].append([new_boundary])
                        geom['semantics']['values'].append(5)
                elif img_direction == "403" or img_direction == "404":
                    for vertex in rectangle:
                        j["vertices"].append([vertex[0], vertex[1], vertex[2]])
                        single_boundary.append(json_vertex_count)
                        json_vertex_count += 1
                    new_boundary = [single_boundary[3], single_boundary[2], single_boundary[1], single_boundary[0]]
                    facade_boundaries.append(new_boundary)
                    geom['semantics']['values'].append(2)

                    for window in single_3d_windows:
                        win_boundaries = []
                        for vertex in window:
                            j["vertices"].append([vertex[0], vertex[1], vertex[2]])
                            win_boundaries.append(json_vertex_count)
                            json_vertex_count += 1

                        new_boundary = [win_boundaries[3], win_boundaries[2], win_boundaries[1], win_boundaries[0]]
                        facade_boundaries.append(new_boundary)
                    geom['boundaries'].append(facade_boundaries)

                    for each in connecting_wall:
                        single_boundary = []
                        for vertex in each:
                            j['vertices'].append([vertex[0], vertex[1], vertex[2]])
                            single_boundary.append(json_vertex_count)
                            json_vertex_count += 1

                        new_boundary = [single_boundary[3], single_boundary[2], single_boundary[1], single_boundary[0]]
                        geom['boundaries'].append([new_boundary])
                        geom['semantics']['values'].append(2)

                    for each_window in new_window_3d:
                        single_boundary = []
                        for vertex in each_window:
                            j['vertices'].append([vertex[0], vertex[1], vertex[2]])
                            single_boundary.append(json_vertex_count)
                            json_vertex_count += 1

                        new_boundary = [single_boundary[3], single_boundary[2], single_boundary[1], single_boundary[0]]
                        geom['boundaries'].append([new_boundary])
                        geom['semantics']['values'].append(4)

                    for each_door in new_door_3d:
                        single_boundary = []
                        for vertex in each_door:
                            j['vertices'].append([vertex[0], vertex[1], vertex[2]])
                            single_boundary.append(json_vertex_count)
                            json_vertex_count += 1

                        new_boundary = [single_boundary[3], single_boundary[2], single_boundary[1], single_boundary[0]]
                        geom['boundaries'].append([new_boundary])
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


def Harris_corner():
    # Load the image
    image_path = '../../visualization/cornerdetection.jpg'
    # image = cv2.imread('../../visualization/coener.jpg', cv2.IMREAD_GRAYSCALE)

    # Load your image as grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Initialize the FAST object with default values
    fast = cv2.FastFeatureDetector_create()
    fast.setThreshold(40)
    # Find and draw the keypoints
    keypoints = fast.detect(image, None)

    # Extract pixel locations of the corners
    corner_locations = [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints]

    # Print corner locations
    print(corner_locations)

    # result = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))
    #
    # # Display the image with the detected corners
    # cv2.imshow('FAST Corners', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return corner_locations


def is_inside_bbox(corner, bbox):
    x, y = corner
    x_min, y_min, x_max, y_max = bbox
    return x_min <= x <= x_max and y_min <= y <= y_max


def cropping(pts):
    points = np.array([pts])

    ymin = np.min(points[:, :, 1])
    ymax = np.max(points[:, :, 1])
    xmin = np.min(points[:, :, 0])
    xmax = np.max(points[:, :, 0])

    return [xmin, ymin, xmax, ymax]


points = []
roi_points = []  # This list will store the clicking results
roi = None


def draw_rectangle(event, x, y, flags, param):
    global points, image, clone, roi

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        print(x, y)

        # If two points have been selected, draw the rectangle
        if len(points) == 2:

            if points[0][0] < points[1][0] and points[0][1] < points[1][1]:
                print("ROI select successful")
                cv2.rectangle(image, points[0], points[1], (0, 255, 0), 2)
                cv2.imshow("image", image)
                roi = clone[points[0][1]:points[1][1], points[0][0]:points[1][0]]
                cv2.imshow("ROI", roi)
                cv2.setMouseCallback("ROI", click_on_ROI)
                print(f"ROI Coordinates: {points}")
            else:
                print("please reselct roi")
                points = []


def click_on_ROI(event, x, y, flags, param):
    global points, roi, roi_points

    if event == cv2.EVENT_LBUTTONDOWN:
        # Adjust clicked position relative to original image
        original_x = points[0][0] + x
        original_y = points[0][1] + y
        print(f"Clicked on ROI at: {original_x}, {original_y}")
        roi_points.append([original_x, original_y])


def LSR(arr):
    projlist = np.array(arr[::2])  # odd-indexed elements
    gtlist = np.array(arr[1::2])  # even-indexed elements

    # Separate x and y coordinates
    projlist_x = projlist[:, 0]
    projlist_y = projlist[:, 1]
    gtlist_x = gtlist[:, 0]
    gtlist_y = gtlist[:, 1]

    # Perform regression on x
    slope_x, intercept_x, r_value_x, p_value_x, std_err_x = stats.linregress(projlist_x, gtlist_x)
    print(f"Regression line equation for x is: y = {slope_x}x + {intercept_x}")
    print(f"R-squared for x is: {r_value_x ** 2}")

    # Perform regression on y
    slope_y, intercept_y, r_value_y, p_value_y, std_err_y = stats.linregress(projlist_y, gtlist_y)
    print(f"Regression line equation for y is: y = {slope_y}x + {intercept_y}")
    print(f"R-squared for y is: {r_value_y ** 2}")

    return slope_x, intercept_x, slope_y, intercept_y



if __name__ == '__main__':

    """read json template file"""
    json_template = f'../../template/template.json'
    with open(json_template, "r") as json_file:
        json_data = json_file.read()
        j = json.loads(json_data)

    """read .off file"""

    path = f"../../template/LOD22_walls.off"
    non_wall_path = f"../../template/original_LOD22.off"
    vertices = read_mesh_vertex(path)
    faces, colors = read_mesh_faces_1(path)
    print("finish read")

    wallsurface, new_color_list1 = wallsurface_filter_bynormal(faces, colors, vertices)
    grouped_faces, new_color_list = merge_surface(faces, colors, vertices)
    non_wall_surface, non_wall_color = obtain_non_wall(non_wall_path)

    rectangles, else_polygon = get_off_3Dfootprint(grouped_faces, vertices)
    rect_for_projection = rectangles.copy()

    img_id = '404_0029_00131853.tif'
    img_type = img_id.strip().split("_")[0]
    print(img_type)
    pts_groundtruthlist = []
    pts_original = []
    id_count = 0
    img_list = []

    # """get the registration model"""
    # for rectangle in rectangles:
    #     rect = rectangle.copy()
    #
    #     new_facede_2d = projection(img_id, rect)
    #
    #     pts_original.append(np.array(new_facede_2d))
    #
    #     pts = np.array(new_facede_2d, np.int32)
    #
    #     img_list.append(name)
    #
    #     pts_groundtruthlist.append(pts)
    #
    #     id_count += 1
    #
    # # !!! the file route should be changed as well
    # image = cv2.imread('../../image/' + img_id)
    # for rectangle in pts_original:
    #     cv2.polylines(image, [rectangle], True, (0, 0, 255), 3)
    # for rectangle in pts_groundtruthlist:
    #     cv2.polylines(image, [rectangle], True, (255, 0, 0), 3)
    # cv2.imshow("Image with Rectangles", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # """obtain registration model"""
    for rectangle in rectangles:
        rect = rectangle.copy()
        new_facede_2d = projection(img_id, rect)
        pts = np.array(new_facede_2d, np.int32)
        pts_original.append(pts)
        id_count += 1

    image = cv2.imread('../../image/' + img_id)
    for rectangle in pts_original:
        cv2.polylines(image, [rectangle], True, (0, 0, 255), 2)

    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", draw_rectangle)

    while True:
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            print("let's reselect ROI")
            image = clone.copy()
            points = []

        elif key == ord("c"):
            print("end selecting")
            break

    cv2.destroyAllWindows()
    print(f"ROI Points: {roi_points}")  # Print out all stored clicking results
    cv2.destroyAllWindows()
    slope_x, intercept_x, slope_y, intercept_y = LSR(roi_points)

    for rectangle in rectangles:
        rect = rectangle.copy()

        new_facede_2d = projection(img_id, rect)

        pts_original.append(np.array(new_facede_2d))

        new_facede_2d = registration(new_facede_2d, slope_x, intercept_x, slope_y, intercept_y)

        pts = np.array(new_facede_2d, np.int32)

        name = "../../texture/texture_" + img_id.strip().split(".")[0] + "_" + str(id_count) + ".jpg"
        img_list.append(name)
        pts_groundtruthlist.append(pts)

        # perspective(img_id, rectangle, pts, name)

        id_count += 1

    image = cv2.imread('../../image/' + img_id)
    # for rectangle in pts_original:
    #     cv2.polylines(image, [rectangle], True, (0, 0, 255), 5)
    for rectangle in pts_groundtruthlist:
        cv2.polylines(image, [rectangle], True, (255, 0, 0), 5)
    cv2.imshow("Image with Rectangles", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    with open('../../result/registration.txt', 'w') as f:
        f.write(f'{img_type} {slope_x} {intercept_x} {slope_y} {intercept_y}\n')


    print("the registration model has been saved.")
