import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)
import re
import csv
import time

def read_mesh_vertex(path):

    vertices = []

    with open(path) as file:
        for line in file.readlines()[2:]:
            clean_parts = line.strip()
            parts = re.split(r'[,\s]+', clean_parts)
            if(len(parts)) == 3:
                vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return vertices


def read_mesh_faces(path):

    faces = []

    current_color = []
    current_faces = []
    with open(path) as file:
        for line in file.readlines():
            clean_parts = line.strip()
            parts = re.split(r'[,\s]+', clean_parts)
            if (len(parts)) == 7:
                face = [int(parts[1]), int(parts[2]), int(parts[3])]
                color = [int(parts[4]), int(parts[5]), int(parts[6])]
                if len(current_color) == 0:
                    current_color = color
                    current_faces.append(face)
                else:
                    if compare_color(color, current_color):
                        current_faces.append(face)
                    else:
                        current_color = color
                        faces.append(current_faces.copy())
                        current_faces = []
                        current_faces.append(face)
        faces.append(current_faces.copy())

    return faces


def read_mesh_faces_1(path):

    faces = []
    colors = []
    with open(path) as file:
        for line in file.readlines():
            clean_parts = line.strip()
            parts = re.split(r'[,\s]+', clean_parts)
            if (len(parts)) == 7:
                face = [int(parts[1]), int(parts[2]), int(parts[3])]
                color = [int(parts[4]), int(parts[5]), int(parts[6])]
                faces.append(face)
                colors.append(color)

    return faces,colors



def compare_color(color, current_color):
    if color[0] == current_color[0] and color[1] == current_color[1] and color[2] == current_color[2]:
        return True
    else: return False


def vertical_test(face, vertices):

    v1 = np.array(vertices[face[0]])
    v2 = np.array(vertices[face[1]])  # Coordinates of second vertex
    v3 = np.array(vertices[face[2]])  # Coordinates of third vertex

    # Calculate two vectors lying on the plane of the triangle
    u = v2 - v1
    v = v3 - v1

    # Calculate the normal vector by taking the cross product of the two vectors
    normal = np.cross(u, v)

    # Normalize the normal vector to have unit length
    normal = normal / np.linalg.norm(normal)

    # if normal[2] <= np.sin(10 * np.pi / 180. ) or normal[2] >= np.sin(-10 * np.pi / 180. ):
    perpendicular = np.array([0,0,1])
    if normal[2] == 0.:
        return True
    else:
        return False


def triangle_area(face, vertices):

    pt1 = np.array(vertices[face[0]])
    pt2 = np.array(vertices[face[1]])  # Coordinates of second vertex
    pt3 = np.array(vertices[face[2]])  # Coordinates of third vertex

    # calculate the vector from P1 to P2
    V1 = pt2 - pt1

    V2 = pt3 - pt1
    cross_product = np.cross(V1, V2)

    area = 0.5 * np.linalg.norm(cross_product)

    # print("The area of the triangle is:", area)
    return area


def wallsurface_filter_bynormal(faces, colors, vertices):

    wallsurface = []
    new_color_list = []

    for i in range(len(faces)):
        if vertical_test(faces[i], vertices):
            wallsurface.append(faces[i])
            new_color_list.append(colors[i])

    return wallsurface, new_color_list


def obtain_non_wall(path):

    vertices = read_mesh_vertex(path)
    faces, colors = read_mesh_faces_1(path)

    non_wall_surface = []
    non_wall_color = []

    for i in range(len(faces)):
        if not vertical_test(faces[i], vertices):
            non_wall_surface.append(faces[i])
            non_wall_color.append(colors[i])

    return non_wall_surface, non_wall_color


def merge_surface(faces, colors, vertices):

    group_faces = []
    new_colors = np.unique(np.array(colors), axis=0)
    for color in new_colors:
        group_faces.append([])

    for i in range(len(new_colors)):
        if new_colors[i][0] == 0 and new_colors[i][1] ==0 and new_colors[i][2] == 0:
            continue
        else:
            for j in range(len(colors)):
                if new_colors[i][0] == colors[j][0] and new_colors[i][1] == colors[j][1] and new_colors[i][2] == colors[j][2]:
                    group_faces[i].append(faces[j])

    return group_faces, new_colors


def group_surface(faces, colors, vertices):

    grouped_surface = []
    grouped_color = []

    i = 0
    j = 0

    print("start to merge co-planar")

    while i < len(colors):

        group_face = []

        group_face.append(faces[i])
        while j <= len(colors):
            if j == len(colors):
                grouped_surface.append(group_face)
                grouped_color.append(colors[i])
                i = j
                break
            elif i == j:
                j += 1
            elif colors[i] == colors[j]:
                group_face.append(faces[j])
                j += 1
            elif colors[i] != colors[j]:
                i = j
                grouped_surface.append(group_face)
                grouped_color.append(colors[i-1])
                break

    return grouped_surface, grouped_color


def wallsuface_filter_byarea(group_faces, colors, vertices):

    new_faces = []
    new_colors = []

    count = 0
    for i in range(len(group_faces)):
        sum_area = 0

        for face in group_faces[i]:
            if count == 678:
                print(group_faces[i])
                print("let's check what's happen")
                print(triangle_area(face, vertices))
            area = triangle_area(face, vertices)
            sum_area += area
            count+=1

        if sum_area >= 180:
            new_faces.append(group_faces[i])
            new_colors.append(colors[i])

    print("there are {} old groups and {} new groups of faces selected".format(len(group_faces), len(new_faces)))
    return new_faces, new_colors


def surface_filter_byarea(faces, colors, vertices):

    wallsurface = []
    new_color_list = []

    for i in range(len(faces)):
        if triangle_area(faces[i], vertices)>=4:
            wallsurface.append(faces[i])
            new_color_list.append(colors[i])
        else:
            continue

    return wallsurface, new_color_list



if __name__ == '__main__':
    start_time = time.time()

    off_path = f"../../visualization/almere_wall.off"

    vertices = read_mesh_vertex(off_path)
    faces, colors = read_mesh_faces_1(off_path)
    print("finish read")

    # wallsurface1, new_color_list1 = group_surface(faces, colors, vertices)

    wallsurface1, new_color_list1 = wallsurface_filter_bynormal(faces, colors, vertices)
    wallsurface, new_color_list = merge_surface(wallsurface1, new_color_list1, vertices)
    # wallsurface, new_color_list = wallsuface_filter_byarea(group_faces, new_colors, vertices)


    # Open a file in write mode
    count = 0

    for group in wallsurface:
        for each in group:
            count+=1

    fo = open("../../visualization/almere_walls.off", "w")

    fo.write("{}{}".format("COFF","\n"))
    fo.write("{} {} {}{}".format(len(vertices), count,0,"\n"))
    fo.write("\n")
    for i in range(len(vertices)):
        fo.write("{} {} {}{}".format(vertices[i][0],vertices[i][1], vertices[i][2],"\n"))

    # for i in range(len(wallsurface)):
    #     fo.write("{}  {} {} {}  {} {} {}{}".format(3, wallsurface[i][0], wallsurface[i][1], wallsurface[i][2],
    #                                                new_color_list[i][0], new_color_list[i][1],
    #                                                new_color_list[i][2], "\n"))
    for j in range(len(wallsurface)):
        for i in range(len(wallsurface[j])):
            fo.write("{}  {} {} {}  {} {} {}{}".format(3, wallsurface[j][i][0], wallsurface[j][i][1], wallsurface[j][i][2],
                                                       new_color_list[j][0], new_color_list[j][1],
                                                       new_color_list[j][2], "\n"))

    # 关闭打开的文件
    fo.close()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")

    print("the number of wallsurface: {}".format(len(wallsurface)))



