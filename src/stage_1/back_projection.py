import numpy as np
import projection_almere

def back_projection(camera_matrix, rotation_matrix, translation_vector, image_point, depth):
    # Homogeneous coordinates of the 2D image point
    homogeneous_image_point = np.array([image_point[0], image_point[1], 0.01])

    # Compute the inverse of the camera matrix
    camera_matrix_inv = np.linalg.inv(camera_matrix)

    # Unproject the 2D image point to 3D camera coordinates
    unprojected_point = depth * np.dot(camera_matrix_inv, homogeneous_image_point)

    # Convert the point from camera coordinates to world coordinates
    rotation_matrix_inv = np.linalg.inv(rotation_matrix)

    world_point = np.dot(rotation_matrix_inv, unprojected_point - translation_vector)

    return world_point


if __name__ == '__main__':

    image_points = [[0,0],
                    [0, 10640],
                    [14192, 0],
                    [14192, 10640]]

    # Example usage
    img_id = "404_0030_00131330.tif"
    camera_matrix, rotation_matrix, translation_vector = projection_almere.get_KRt(img_id)
    offset_x, offset_y, offset_z = projection_almere.get_offset()

    translation_vector = np.array(translation_vector[0])
    image_point = np.array([0, 0])
    depth = 800 # Example depth value

    for image_point in image_points:
        world_point = back_projection(camera_matrix, rotation_matrix, translation_vector, image_point, depth)
        world_point = [world_point[0] + offset_x, world_point[1] + offset_y, world_point[2]]
        print(world_point)