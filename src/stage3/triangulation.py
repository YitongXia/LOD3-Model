

def read_off_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    if lines[0].strip() != 'OFF':
        raise ValueError('Not a valid OFF file')

    num_vertices, num_faces, _ = map(int, lines[1].strip().split())

    vertices = []
    for i in range(2, 2 + num_vertices):
        vertex = list(map(float, lines[i].strip().split()))
        vertices.append(vertex)

    faces = []
    for i in range(2 + num_vertices, 2 + num_vertices + num_faces):
        face = list(map(int, lines[i].strip().split()))[
               1:]  # Ignore the first number, which is the number of vertices in the face
        faces.append(face)

    return vertices, faces


def off_to_triangle_input(vertices, faces):
    input_dict = {
        'vertices': vertices,
        'segments': []
    }

    for face in faces:
        for i in range(len(face)):
            input_dict['segments'].append([face[i], face[(i + 1) % len(face)]])

    return input_dict


if __name__ == '__main__':
    off_file = '../../src/stage_1/mayday_wall_2.off.off'
    vertices, faces = read_off_file(off_file)
    triangle_input = off_to_triangle_input(vertices, faces)
    triangulated = triangle.triangulate(triangle_input, 'p')
