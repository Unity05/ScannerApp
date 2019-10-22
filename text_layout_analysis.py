import cv2 as cv
import numpy as np
from itertools import permutations
import math


def box_in_box(boxes, sheet_compare=False):
    '''Checks if a box is inside of another box.

    Args:
        boxes (tuple): The two boxes
        sheet_compare (bool): Whether a box should be compared with the sheet

    Returns:
        Whether the box is inside of the other box as a boolean.
    '''

    if sheet_compare:
        sheet_box, box_1 = boxes
        # ==== sheet_box ====
        box_0_x0, box_0_y0, box_0_x1, box_0_y1 = sheet_box
        # ==== box_1 ====
        box_1_x0, box_1_y0, box_1_w, box_1_h = box_1
        box_1_x1 = box_1_x0 + box_1_w
        box_1_y1 = box_1_y0 + box_1_h
    else:
        box_0, box_1 = boxes
        # ==== box_0 ====
        box_0_x0, box_0_y0, box_0_w, box_0_h = box_0
        box_0_x1 = box_0_x0 + box_0_w
        box_0_y1 = box_0_y0 + box_0_h
        # ==== box_1 ====
        box_1_x0, box_1_y0, box_1_w, box_1_h = box_1
        box_1_x1 = box_1_x0 + box_1_w
        box_1_y1 = box_1_y0 + box_1_h

    # ==== logic ====
    is_in_box = box_0_x0 < box_1_x0 < box_0_x1 and box_0_y0 < box_1_y0 < box_0_y1 \
                and box_0_x0 < box_1_x1 < box_0_x1 and box_0_y0 < box_1_y1 < box_0_y1
    return is_in_box


def remove_nested_boxes(boxes, corner_points, image_model_ratio):
    '''Removes boxes that are inside of other boxes.

    Args:
        boxes (list): The two boxes
        corner_points (list): The points of the sheet's corners in the shape
            [top[left, right], bottom, left[top, not], right]
        image_model_ratio (float): The ratio between the size of the original image and
            the sheet size used to find the edges

    Returns:
        The filtered list of boxes.
    '''

    sheet_box = (corner_points[0][0][0], corner_points[0][0][1], corner_points[1][1][0], corner_points[1][1][1])
    sheet_box = [int(z * image_model_ratio) for z in sheet_box]

    boxes_combinations = list(permutations(boxes, 2))
    return_boxes = []
    collected_boxes = []
    for box_comb in boxes_combinations:
        box_is_in_box = box_in_box(box_comb)
        if not box_is_in_box:
            collected_boxes.append(box_comb[1])
    for box in boxes:
        if collected_boxes.count(box) == (len(boxes)-1):
            box_is_in_box = box_in_box([sheet_box, box], sheet_compare=True)
            if box_is_in_box:
                return_boxes.append(box)

    return return_boxes


def get_contour_points(img, volume_threshold, sheet_volume, sheet_model_ratio, image_sheet_ratio, corner_points, image_model_ratio):
    '''Computes the points of the layout parts.

    Args:
        img (numpy.ndarray): The image
        volume_threshold (float): The minimum volume a of layout part
        sheet_volume (int): The sheet's volume
        sheet_model_ratio (float): The ratio between the sheet size used for the layout analysis and
            the sheet size used to find the edges
        image_sheet_ratio (float): The ratio between the sheet size of the original image and
            the sheet size used for the layout analysis
        corner_points (list): The points of the sheet's corners in the shape
            [top[left, right], bottom, left[top, not], right]
        image_model_ratio (float): The ratio between the size of the original image and
            the sheet size used to find the edges

    Returns:
         The points of the layout parts.
    '''

    img = cv.blur(img, (5, 5), 0)  # blur seems to fit better than GaussianBlur in this case
                                   # blur for less noise after thresholding
    thresh = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((10, 10), np.uint8)
    dilation = cv.dilate(thresh, kernel, iterations=1)
    contours, _ = cv.findContours(dilation, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    return_contour_points = []
    volumes = []
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        box_volume = w * h
        if box_volume < volume_threshold or box_volume >= sheet_volume*0.95*sheet_model_ratio:
            continue
        return_contour_points.append([int(z * image_sheet_ratio) for z in[x, y, w, h]])
        volumes.append(box_volume)
    return_contour_points = remove_nested_boxes(boxes=return_contour_points, corner_points=corner_points, image_model_ratio=image_model_ratio)
    return return_contour_points


def get_layout(img_file, sheet_volume, corner_points):
    '''Computes all layout parts of a given image and saves them in 'layout_parts/'.

    Args:
        img_file (str): The path of the image
        sheet_volume (int): The volume of the sheet (from the edge detection)
        corner_points (list): The points of the sheet's corners in the shape
            [top[left, right], bottom, left[top, not], right]

    Returns:
        The points of every layout part as a rectangle.
    '''

    img_original = cv.imread(img_file)    # flag 0 --> IMREAD_GRAYSCALE
    avg_height = 800
    avg_width = 600
    new_height = int(img_original.shape[0] * (avg_height / img_original.shape[0]))
    new_width = int(img_original.shape[1] * (avg_width / img_original.shape[1]))
    img = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, (new_width, new_height))

    # ==== compute ratios ====
    model_volume = 400*300  # volume of denoise_edge.py image
    layout_sheet_volume = 800*600  # volume of image used for layout_analysis
    image_volume = 3120*4160    # volume of original image
    sheet_model_ratio = layout_sheet_volume/model_volume
    image_sheet_ratio = math.sqrt(image_volume/layout_sheet_volume)     # ratio for contour points
    image_model_ratio = math.sqrt(image_volume/model_volume)    #

    volume_threshold = sheet_volume * 0.025
    boxes = get_contour_points(img=img, volume_threshold=volume_threshold, sheet_volume=sheet_volume,
                               sheet_model_ratio=sheet_model_ratio, image_sheet_ratio=image_sheet_ratio,
                               corner_points=corner_points, image_model_ratio=image_model_ratio)
    layout_parts = []
    box_coords = []
    for rect in boxes:
        cv.rectangle(img, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 255, 0), 2)   #???????????????
        layout_parts.append(img_original[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])])
        box_coords.append([rect[1], rect[1]+rect[3], rect[0], rect[0]+rect[2]])

    for i, layout_part in enumerate(layout_parts):
        cv.imwrite('layout_parts/layout_part_{}.png'.format(str(i)), layout_part)

    return box_coords
