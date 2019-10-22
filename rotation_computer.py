import numpy as np
import math


def get_other_point(edge_img_tensor, start_point, is_right, is_top):
    '''Computes the other corner point depending on a given line.

    Args:
        edge_img_tensor (torch.Tensor): An image of the edges
        start_point (tuple): The point to start the computation from
        is_right (bool): Whether the line is on the right side
        is_top (bool): Whether the line is at the top

    Returns:
        The other corner point depending on the given line.

    Raises:
        IndexError: An error occurred trying finding a point out of the image --> sheet partly out of image
    '''

    x, y = start_point
    horizontal_direction = 1
    vertical_direction = 1
    if is_right:
        horizontal_direction = -1
    if not is_top:
        vertical_direction = -1

    same_x_counter = 0
    while True:
        try:
            if edge_img_tensor[y][x+horizontal_direction].item() == 1.0:
                same_x_counter = 0
                x += horizontal_direction
            elif edge_img_tensor[y+vertical_direction][x].item() == 1.0:
                same_x_counter += 1
                y += vertical_direction
            elif edge_img_tensor[y+vertical_direction][x+horizontal_direction].item() == 1.0:
                same_x_counter = 0
                x += horizontal_direction
                y += vertical_direction

            elif edge_img_tensor[y-vertical_direction][x].item() == 1.0:    # for the case of a bulge
                y -= vertical_direction
                same_x_counter += 1
            elif edge_img_tensor[y-vertical_direction][x-horizontal_direction].item() == 1.0:
                x -= horizontal_direction
                y -= vertical_direction
                same_x_counter += 1
            else:   # end-point found
                break
            if same_x_counter > 2:
                break
        except IndexError:
            break
    return (x, y)


def insert_new_point(corner_pixel_coords, is_right, point, line):
    '''Inserts the given new corner point into the given corner point list.

    Args:
        corner_pixel_coords (list): A list of corner points
        is_right (bool): Whether the point is on the right side
        point (tuple): The corner point to insert
        line (int): The line index (line: 0=top; 1=bot; 2=left; 3=right)

    Returns:
        The edited corner point list.
    '''

    if is_right:
        corner_pixel_coords[line].insert(0, point)
    else:
        corner_pixel_coords[line].append(point)
    return corner_pixel_coords


def get_sheet_state(corner_pixel_coords):
    '''Computes all data depending on the sheet's shape.

    Args:
        corner_pixel_coords (list): A list of corner points

    Returns:
        The updated corner points list and a dictionnary,
        containing the computed sheet data of shape
        {'top': x, 'bottom': x, 'left': x, 'right': x, 'angle': x,
        'angle_top': x, 'angle_bot': x, 'volume': x}
    '''

    sheet_state_dict = {'top': 0, 'bottom': 0, 'left': 0, 'right': 0, 'angle': 0,
                        'angle_top': 0, 'angle_bot': 0, 'volume': 0}
    angles = []
    sign_angles = []
    global_angle_sign = 1
    lens = []
    for i, key in enumerate(sheet_state_dict.keys()):
        height = corner_pixel_coords[i][0][1] - corner_pixel_coords[i][1][1]
        width = corner_pixel_coords[i][1][0] - corner_pixel_coords[i][0][0]
        len = math.sqrt(height**2 + width**2)
        lens.append(len)
        if i < 2:
            angle = math.degrees(math.atan(height / width))
            if i == 0:
                global_angle_sign = np.sign(angle)
            if abs(angle) > 0.5:
                angles.append(abs(angle))
            else:
                angles.append(0)
            sign_angles.append(angle)
        else:
            angle = math.degrees(abs(math.atan(width / height)))
            if angle > 0.5:
                angles.append(angle)
            else:
                angles.append(0)
        sheet_state_dict[key] = len
        if i == 3:
            break

    sheet_state_dict['angle'] = np.mean(angles[2:]) * global_angle_sign
    sheet_state_dict['angle_top'] = sign_angles[0]
    sheet_state_dict['angle_bot'] = sign_angles[1]
    sheet_state_dict['volume'] = int(((lens[0]*lens[1])+(lens[2]*lens[3])) / 2)

    return sheet_state_dict


def get_rotation(edge_img_tensor):
    '''Computes all sheet corner points as well as other data depending on the sheet's state.

    Args:
        edge_img_tensor (torch.Tensor): An image of the edges

    Returns:
        The corner points list of shape
        [top[left, right], bottom, left[top, bot], right]
        and other sheet data of shape
        {'top': x, 'bottom': x, 'left': x, 'right': x, 'angle': x,
        'angle_top': x, 'angle_bot': x, 'volume': x}
    '''

    width_mid = edge_img_tensor.size()[1] // 2
    top_is_right = False
    bot_is_right = False
    # ==== first pixel from top ====
    corner_pixel_coords = []
    for line_index in range(len(edge_img_tensor)):
        for pixel_index in range(len(edge_img_tensor[line_index])):
            pixel = edge_img_tensor[line_index][pixel_index]
            if pixel.item() == 1.0:
                corner_pixel_coords.append((pixel_index, line_index))
    corner_pixel_coords = [[corner_pixel_coords[0]], [corner_pixel_coords[-1]], [], []]
    if corner_pixel_coords[0][0][0] > width_mid:
        top_is_right = True
    if corner_pixel_coords[1][0][0] > width_mid:
        bot_is_right = True
    # ==== top ====
    corner_pixel_coords[0][0] = get_other_point(edge_img_tensor=edge_img_tensor, start_point=corner_pixel_coords[0][0],
                                                is_right=not top_is_right, is_top=True)
    corner_pixel_coords = insert_new_point(corner_pixel_coords=corner_pixel_coords, is_right=top_is_right,
                                           point=get_other_point(edge_img_tensor=edge_img_tensor,
                                                                 start_point=corner_pixel_coords[0][0],
                                                                 is_right=top_is_right, is_top=True),
                                           line=0)
    # ==== bottom ====
    corner_pixel_coords[1][0] = get_other_point(edge_img_tensor=edge_img_tensor, start_point=corner_pixel_coords[1][0],
                                                is_right=not bot_is_right, is_top=False)
    corner_pixel_coords = insert_new_point(corner_pixel_coords=corner_pixel_coords, is_right=bot_is_right,
                                           point=get_other_point(edge_img_tensor=edge_img_tensor,
                                                                 start_point=corner_pixel_coords[1][0],
                                                                 is_right=bot_is_right, is_top=False),
                                           line=1)
    # ==== left ====
    corner_pixel_coords[2].append(corner_pixel_coords[0][0])
    corner_pixel_coords[2].append(corner_pixel_coords[1][0])
    # ==== right ====
    corner_pixel_coords[3].append(corner_pixel_coords[0][1])
    corner_pixel_coords[3].append(corner_pixel_coords[1][1])

    sheet_state_dict = get_sheet_state(corner_pixel_coords=corner_pixel_coords)

    return corner_pixel_coords, sheet_state_dict
