import json
# import requests
from PIL import Image
import os

def get_metadata(img_id):
    """
    Get metadata of certain img from EO_obliek_verwerkt.txt
    :param img_id: string
    :return: an array containing all the metadata
    """

    # metadata file offered by AeroVision/Gementee Almere
    # metadata_file = f"../../metadata/EO_obliek_verwerkt.txt"

    cwd = os.getcwd()
    file_path = os.path.abspath("EO_obliek_verwerkt.txt")
    metadata_file = os.path.relpath(file_path, cwd)

    # format: ID  E: east  N:north  H:height O:omega P:phi  K:kappa
    with open(metadata_file) as file:
        for line in file:
            parts = line.strip().split(" ")
            if (parts[0] == img_id):
                return parts


def get_img_center(img_id):

    # image center file offered by AeroVision/Gementee Almere
    img_center_overview = f"../../metadata/EO_obliek_verwerkt.txt"

    # img_center: lat,lon
    with open(img_center_overview) as file:
        for line in file:
            img_center = line.strip().split(",")
            if (img_center[0] == img_id):
                return img_center


# coord_2d should be in top-left, right-bottom format
def crop_facades(img, coord_2d, save):

    path = "" + save
    img = Image.open(img)
    img2 = img.crop(coord_2d)
    img2.save(path)


def get_image(img):
    """
    read input tif file
    :param img: string
    :return: im
    """
    im = Image.open(img)
    return im


def visualize_img(im):
    im.show()