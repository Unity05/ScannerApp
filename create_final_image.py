import cv2 as cv
import math
import numpy as np
from os import listdir


def rotate_image(img, angle):
    '''Rotates given image through a given angle.

    Args:
        img (numpy.ndarray): The image to rotate
        angle (float): The rotation angle

    Returns:
        The image rotated through the given angle
    '''

    height, width, _ = img.shape
    M = cv.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    dst = cv.warpAffine(img, M, (width, height), borderValue=(255, 255, 255))
    return dst


def insert_image(insert_img, ground_img, box_coords, output_img_ratio, sheet_data, angle_dict):
    '''Inserts an image into a given ground image.

    Rotates a given image and inserts the rotated image
    into the given ground image.

    Args:
        insert_img (numpy.ndarray): The image to insert
        ground_img (numpy.ndarray): The image the insert_image insert into
        box_coords (list): The points of the insert image's position on the original image
        output_img_ratio (float): The ratio of the original image size and the the output image size
        sheet_data (tuple): The sheet's left upper corner x and y value (x, y)
        angle_dict (dict): A dictionnary with the full sheet angle data

    Returns:
        The ground image with the inserted insert_image.
    '''

    sheet_x_0, sheet_y_0 = sheet_data

    # ==== resize insert image ====
    insert_img_h = int(insert_img.shape[0] * output_img_ratio)
    insert_img_w = int(insert_img.shape[1] * output_img_ratio)
    insert_img = cv.resize(insert_img, (insert_img_w, insert_img_h))

    # ==== update insert coords ====
    x_0 = int((box_coords[2] - sheet_x_0) * output_img_ratio)
    y_0 = int((box_coords[0] - sheet_y_0) * output_img_ratio)
    x_1 = x_0 + insert_img_w
    y_1 = y_0 + insert_img_h

    mid_point = np.mean([y_0, y_1])
    ground_img_h = ground_img.shape[1]
    angle = 0

    if mid_point < int(ground_img_h*(1/3)):
        angle = angle_dict['angle_top']
    elif mid_point < int(ground_img_h*(2/3)):
        angle = angle_dict['angle']
    elif mid_point <= int(ground_img_h):
        angle = angle_dict['angle_bot']

    insert_img = rotate_image(img=insert_img, angle=-1*angle*0.75).transpose(2, 0, 1)
    insert_img = insert_img / 255.0

    for channel_index in range(3):
        ground_img[channel_index][y_0:y_1, x_0:x_1] = insert_img[channel_index]

    return ground_img


def get_final_image(box_coords, corner_pixel_coords, sheet_state_dict, out_img_path):
    '''Creates the final image.

    Inserts all single layout parts into the ground image (DinA4)
    and saves the final image. Also computes necessary
    ratios between all different images.

    Args:
        box_coords (list): The points of the insert image's position on the original image
        corner_pixel_coords (list): The points of the sheet's corners in the shape
            [top[left, right], bottom, left[top, not], right]
        sheet_state_dict (dict): contains all needed information about the image
        out_img_path (str): The path where to save the output image
    '''

    # ==== get necessary data ====
    img_width, img_height = 3120, 4160
    state_width, state_height = 300, 400

    # ==== process data ====
    img_state_ratio = img_height / state_height
    state_height_avg, state_width_avg = np.mean([sheet_state_dict['left'], sheet_state_dict['right']]), np.mean(
        [sheet_state_dict['top'], sheet_state_dict['bottom']])
    rl_sheet_width = img_state_ratio * state_width_avg
    rl_sheet_height = img_state_ratio * state_height_avg

    sheet_x_0 = int(np.mean([corner_pixel_coords[2][0][0], corner_pixel_coords[2][1][0]]) * img_state_ratio)
    sheet_y_0 = int(np.mean([corner_pixel_coords[0][0][1], corner_pixel_coords[0][1][1]]) * img_state_ratio)
    sheet_data = (sheet_x_0, sheet_y_0)

    width, height = 2480, 3508  # DinA4 --> output image size
    output_img_ratio = math.sqrt((width * height) / (rl_sheet_width * rl_sheet_height))

    # ==== create new image ====
    new_img = np.ones((height, width, 3), dtype=np.float32).transpose(2, 0, 1)

    layout_part_images = listdir('layout_parts')
    for i, layout_part_file in enumerate(layout_part_images):
        insert_img = cv.imread('layout_parts/' + layout_part_file)
        new_img = insert_image(insert_img=insert_img, ground_img=new_img, box_coords=box_coords[i],
                               output_img_ratio=output_img_ratio, sheet_data=sheet_data, angle_dict=sheet_state_dict)

    new_img = new_img * 255.0  # cv.imwrite saves it as 0-255 rgb values
    cv.imwrite(out_img_path, new_img.transpose(1, 2, 0))