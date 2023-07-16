import json

from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
from src.stage_1.building_determination import *
from src.stage_1.perspective_projection import *
import cv2

def compute_distance(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

def hierarchical(w_width_height):

    data = np.array(w_width_height)
    agglomerative_clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0)
    agglomerative_labels = agglomerative_clustering.fit_predict(data)

    # Plot the dendrogram
    linked = linkage(data, method='ward')

    plt.figure(figsize=(10, 5))
    dendrogram(linked, labels=data)
    plt.title('Dendrogram for Hierarchical Clustering')
    plt.xlabel('Data Points')
    plt.ylabel('Euclidean Distance')
    plt.show()

    return agglomerative_labels


def dbscan(width_height):
    data = np.array(width_height)
    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=70, min_samples=1)
    dbscan_labels = dbscan.fit_predict(data)

    # unique_labels = set(dbscan_labels)
    # colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    #
    # for k, col in zip(unique_labels, colors):
    #     if k == -1:
    #         # Black used for noise.
    #         col = [0, 0, 0, 1]
    #
    #     class_member_mask = (dbscan_labels == k)
    #
    #     xy = data[class_member_mask]
    #     plt.scatter(xy[:, 0], xy[:, 1], c=[col], edgecolors='k', marker='o', s=100, alpha=0.5)
    #
    # plt.xlabel("Width")
    # plt.ylabel("Height")
    # plt.title("DBSCAN Clustering Results")
    # plt.show()

    return dbscan_labels


def kmeans(width_height):

    data = np.array(width_height)
    # Apply KMeans clustering (assuming 3 clusters)
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans_labels = kmeans.fit_predict(data)

    return kmeans_labels


def dbscan_kmeans(w_width_height):

    data = np.array(w_width_height)
    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=10, min_samples=1)
    dbscan_labels = dbscan.fit_predict(data)

    # Apply KMeans clustering (assuming 3 clusters)
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans_labels = kmeans.fit_predict(data)

    # # Plot the results
    # fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    #
    # ax[0].scatter(data[:, 0], data[:, 1], c=dbscan_labels, cmap='viridis')
    # ax[0].set_title('DBSCAN Clustering')
    #
    # ax[1].scatter(data[:, 0], data[:, 1], c=kmeans_labels, cmap='viridis')
    # ax[1].set_title('KMeans Clustering')
    #
    # plt.show()

    return dbscan_labels, kmeans_labels


def visualize_openings(rectangles, bbox):

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.cm as cm
    import matplotlib

    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['Palatino Linotype'] + matplotlib.rcParams['font.serif']


    maxHeight = int((bbox[0][2] - bbox[1][2]) * 100)
    maxWidth = int((compute_distance(bbox[1], bbox[2])) * 100)

    num_clusters = len(rectangles)
    colors = cm.rainbow(np.linspace(0, 1, num_clusters))

    fig, ax = plt.subplots()

    for cluster, color in zip(rectangles, colors):
        for rect in cluster:
            xmin, ymin, xmax, ymax = rect
            width = xmax - xmin
            height = ymax - ymin
            ax.add_patch(patches.Rectangle((xmin, ymin), width, height, edgecolor=color, facecolor='none'))

    ax.set_xlim(0, maxWidth)
    ax.set_ylim(0, maxHeight)
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()
    plt.show()


def get_openings(openings_file, rectangles):

    with open(openings_file, "r") as file:
        # Read the contents of the file
        json_data = file.read()

    # Decode the JSON string to a Python object
    openings = json.loads(json_data)

    window_by_img = []
    door_by_img = []

    windows_index = 0
    door_index = 0
    image_index = []

    facade_windows = []

    for each in openings['images']:
        img_name = each['file_name']
        image_index.append(img_name)

    image_dict = {}
    for each in openings['images']:
        image_dict[each['file_name']] = each['id']

    """save category"""
    for each in openings['categories']:
        if each['name'] == 'window':
            windows_index = each['id']
        if each['name'] == 'door':
            door_index = each['id']

    for i in range(len(image_index)):

        id = image_index[i]
        no = int(id.strip().split("_")[4].strip().split('.')[0])
        # bbox = rectangles[no]

        w_top_left_corner = []
        w_width_height = []
        d_top_left_corner = []
        d_width_height = []
        w_bbox = []
        d_bbox = []

        for each in openings['annotations']:
            print(each)
            #if each['image_id'] == image_dict[""]
            if each['image_id'] == id:
                if each['category_id'] == windows_index:
                    if each['score'] >= 0.5:
                        w_bbox.append(each['bbox'])
                        corner = [each['bbox'][0], each['bbox'][1]]
                        w_top_left_corner.append(corner)
                        # w_width_height.append([each['bbox'][2], each['bbox'][3]])
                        w_width_height.append([each['bbox'][2] - each['bbox'][0], each['bbox'][3] - each['bbox'][1]])

                elif each['category_id'] == door_index:
                    if each['score'] > 0.5:
                        d_bbox.append(each['bbox'])
                        corner = [each['bbox'][0], each['bbox'][1]]
                        d_top_left_corner.append(corner)
                        d_width_height.append([each['bbox'][2], each['bbox'][3]])

        if len(w_width_height) > 0:
            window_labels = dbscan(w_width_height)

            windows_type = np.unique(np.array(window_labels), axis=0)
            windows_group = []

            for each in windows_type:
                windows_group.append([])

            for j in range(len(window_labels)):
                for i in range(len(windows_type)):
                    if window_labels[j] == windows_type[i]:
                        windows_group[i].append(w_bbox[j])
                        break
            # visualize_openings(windows_group, bbox)

            window_by_img.append(windows_group)
        else:
            window_by_img.append([])


        if len(d_width_height) > 0:
            doors_labels = dbscan(d_width_height)
            doors_type = np.unique(np.array(doors_labels), axis=0)
            doors_group = []

            for each in doors_type:
                doors_group.append([])

            for j in range(len(doors_labels)):
                for i in range(len(doors_type)):
                    if doors_labels[j] == doors_type[i]:
                        doors_group[i].append(d_bbox[j])
                        break

            door_by_img.append(doors_group)
        else: door_by_img.append([])

    return image_index, window_by_img, door_by_img


def special_get_openings(openings_file, rectangles):

    with open(openings_file, "r") as file:
        # Read the contents of the file
        json_data = file.read()

    # Decode the JSON string to a Python object
    openings = json.loads(json_data)

    openings_belongs = []

    window_by_img = []
    door_by_img = []

    windows_index = 0
    door_index = 0
    image_index = []

    facade_windows = []

    for each in openings['images']:
        img_name = each['file_name']
        image_index.append(img_name)

    image_dict = {}
    for each in openings['images']:
        image_dict[each['id']] = each['file_name']

    """save category"""
    for each in openings['categories']:
        if each['name'] == 'window':
            windows_index = each['id']
        if each['name'] == 'door':
            door_index = each['id']

    for i in range(len(image_index)):

        id = image_index[i]
        no = int(id.strip().split("_")[4].strip().split('.')[0])
        bbox = rectangles[no]

        w_top_left_corner = []
        w_width_height = []
        d_top_left_corner = []
        d_width_height = []
        w_bbox = []
        d_bbox = []

        for each in openings['annotations']:
            print(image_dict[each["image_id"]])
            if id == image_dict[each["image_id"]]:
                if each['category_id'] == windows_index:
                    w_bbox.append(each['bbox'])
                    corner = [each['bbox'][0], each['bbox'][1]]
                    w_top_left_corner.append(corner)
                    w_width_height.append([each['bbox'][2], each['bbox'][3]])
                    # w_width_height.append([each['bbox'][2] - each['bbox'][0], each['bbox'][3] - each['bbox'][1]])

                elif each['category_id'] == door_index:
                    d_bbox.append(each['bbox'])
                    corner = [each['bbox'][0], each['bbox'][1]]
                    d_top_left_corner.append(corner)
                    d_width_height.append([each['bbox'][2], each['bbox'][3]])


        if len(w_width_height) > 0:
            window_labels = dbscan(w_width_height)

            windows_type = np.unique(np.array(window_labels), axis=0)
            windows_group = []

            for each in windows_type:
                windows_group.append([])

            for j in range(len(window_labels)):
                for i in range(len(windows_type)):
                    if window_labels[j] == windows_type[i]:
                        windows_group[i].append(w_bbox[j])
                        break
            #visualize_openings(windows_group, bbox)

            window_by_img.append(windows_group)
        else:
            window_by_img.append([])


        if len(d_width_height) > 0:
            doors_labels = dbscan(d_width_height)
            doors_type = np.unique(np.array(doors_labels), axis=0)
            doors_group = []

            for each in doors_type:
                doors_group.append([])

            for j in range(len(doors_labels)):
                for i in range(len(doors_type)):
                    if doors_labels[j] == doors_type[i]:
                        doors_group[i].append(d_bbox[j])
                        break

            door_by_img.append(doors_group)
        else: door_by_img.append([])

    return image_index, window_by_img, door_by_img


def sub_size_regularizations(image_index, window_by_img, door_by_img):
    for i in range(len(image_index)):
        if len(window_by_img[i]) > 0:
            window_by_img[i] = sub_size_regularization2(window_by_img[i])
        if len(door_by_img[i]) > 0:
            door_by_img[i] = sub_size_regularization2(door_by_img[i])


def size_regularizations(openings_groups):

    for openings_group in openings_groups:
        for w_bbox in openings_group:
            avg_width = 0
            avg_height = 0
            for each in w_bbox:
                avg_width += each[2] - each[0]
                avg_height += each[3] - each[1]

            avg_width = int(avg_width / len(w_bbox))
            avg_height = int(avg_height / len(w_bbox))

            for each in w_bbox:
                each[2] = avg_width
                each[3] = avg_height

    return openings_groups


def sub_size_regularization2(openings_group):

    for w_bbox in openings_group:
        avg_width = 0
        avg_height = 0
        for each in w_bbox:
            avg_width += each[2]
            avg_height += each[3]

        avg_width = int(avg_width / len(w_bbox))
        avg_height = int(avg_height / len(w_bbox))

        for each in w_bbox:
            each[2] = avg_width
            each[3] = avg_height

    return openings_group



if __name__ == '__main__':

    file = f'../import_Almere/facade-3.json'
    rectangles = []
    image_index, window_by_img, door_by_img = get_openings(file, rectangles)
    size_regularizations(image_index, window_by_img, door_by_img)




