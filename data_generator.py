import torch
import random as rnd
import math
import numpy as np
import cv2 as cv
import string


def uncomplete_image(batch, lines, p):
    '''Removes parts of the sheet's contour.

    Removes a part on every side of the sheet's contour.

    Args:
        batch (torch.Tensor): A batch of training images
        lines (list): Contains all points of every line, sorted into lines
        p (float): The percentage to be removed of every line

    Returns:
        The edited batch of training images.
    '''

    for image_index in range(batch.size()[0]):
        img_tensor = batch[image_index].squeeze(0)
        img_lines = lines[image_index]
        uncomplete_points = []
        for line_index in range(4):
            line = img_lines[line_index]
            uncomplete_len = int(p * len(line))
            uncomplete_start_index = rnd.randint(0, len(line)-uncomplete_len)
            uncomplete_end_index = uncomplete_start_index + uncomplete_len
            uncomplete_points.append(line[uncomplete_start_index:uncomplete_end_index])
        for line in uncomplete_points:
            for pixel in line:
                img_tensor[pixel[1]][pixel[0]] = 0
                batch[image_index] = img_tensor.unsqueeze(0)
    return batch


def create_batches(batch_size, number_of_batches, image_tensors, lines):
    '''Creates batches of a given batch_size and number_of_batches from a list of images.

    Args:
        batch_size (int): The batch size
        number_of_batches (int): The number of batches to create
        image_tensors (list): The images to create the batches from
        lines (list): Contains all points of every line, sorted into lines

    Returns:
        The created batches as torch.FloatTensor and
        the lines of the batches' images.
    '''

    file_counter = 0
    batches = []
    return_lines = []
    for batch in range(number_of_batches):
        batch_images = []
        batch_lines = []
        for i in range(batch_size):
            batch_images.append(image_tensors[file_counter][0].tolist())
            batch_lines.append(lines[file_counter])
            file_counter += 1
        return_lines.append(batch_lines)
        batches.append(batch_images)
    return torch.FloatTensor(batches), return_lines


def cat_lines(img):
    '''Concatenates a given image tensor's rows.

    Args:
        img (torch.Tensor): An image tensor

    Returns:
        The concatenated image tensor.
    '''

    x = img[0]
    for i in range(1, img.size()[0]):
        x = torch.cat((x, img[i]), 0)
    return x


def add_noise_bs(batch, p, color=False):
    '''Adds noise to images of a given batch.

    Args:
        batch (torch.Tensor): A batch of training images
        p (float): The percentage to get noised of the every image
        color (bool): Whether the noise shall be black or a color

    Returns:
        The batch with noised images.
    '''

    for image_index in range(batch.size()[0]):
        number_of_pixels = batch[image_index].size()[1] * batch[image_index].size()[2]
        number_of_noise = math.ceil(number_of_pixels * p)
        indices = rnd.choices(range(number_of_pixels), k=number_of_noise)
        image = cat_lines(batch[image_index].squeeze())
        if color:
            for index in indices:
                image[index] = rnd.uniform(0, 1)
        else:
            image[indices] = 1                                      # 0 --> black; 1 --> white
        batch[image_index] = image.reshape((batch[image_index].size()[1], batch[image_index].size()[2])).unsqueeze(0)
    return batch


def add_noise(img, p, color=False):
    '''Adds noise to an image.

    Args:
        img (torch.Tensor): A training image
        p (float): The percentage to get noised of the image
        color (bool): Whether the noise shall be black or a color

    Returns:
        The noised image.
    '''

    number_of_pixels = img.size()[0] * img.size()[1]
    number_of_noise = math.ceil(number_of_pixels * p)
    indices = rnd.choices(range(number_of_pixels), k=number_of_noise)
    image = cat_lines(img)
    if color:
        for index in indices:
            image[index] = rnd.uniform(0, 1)
    else:
        image[indices] = 1                                      # 0 --> black; 1 --> white
    img = image.reshape((img.size()[0], img.size()[1]))
    return img


def DDA(x_0, y_0, x_1, y_1, thickness):
    '''Computes all points of a line.

    Args:
        x_0 (int): First x value of the line
        y_0 (int): First y value of the line
        x_1 (int): Second x value of the line
        y_1 (int): Second y value of the line
        thickness (int): The thickness of the line

    Returns:
        All points of the line.
    '''

    points = []
    dx = x_1 - x_0
    dy = y_1 - y_0
    if abs(dx) >= abs(dy):
        step = abs(dx)
    else:
        step = abs(dy)
    dx = dx / step
    dy = dy / step
    x = x_0
    y = y_0
    i = 1
    while i <= step:
        for thickness_i in range(thickness):
            points.append((int(x+thickness_i), int(y+thickness_i)))
        x += dx
        y += dy
        i += 1
    return points


def draw_line(x_0, y_0, x_1, y_1, img_tensor, thickness, rgb):
    '''Draws the line points on an image.

    Args:
        x_0 (int): First x value of the line
        y_0 (int): First y value of the line
        x_1 (int): Second x value of the line
        y_1 (int): Second y value of the line
        img_tensor (torch.Tensor): A training image
        thickness (int): The thickness of the line
        rgb (bool): Whether the line shall be drawn as RGB

    Returns:
        The edited image tensor and all line points.

    Raises:
        IndexError: An error occurred trying drawing a point out of the image
    '''

    points = list(DDA(x_0, y_0, x_1, y_1, thickness=thickness))
    if not rgb:
        for (x, y) in points:
            img_tensor[y][x] = 1
    else:
        for (x, y) in points:
            for channel_index in range(3):
                try:
                    img_tensor[channel_index][y][x] = 0
                except IndexError:
                    img_tensor[channel_index][y-1][x-1] = 0

    return img_tensor, points


def generate_sheet(img_tensor, thickness=1, rgb=False):
    '''Creates a rectangle on a given image.

    Args:
        img_tensor (torch.Tensor): A training image tensor
        thickness (int): The thickness of the line
        rgb (bool): Whether the lines shall be drawn as RGB

    Returns:
        The edited image tensor and
        the points of the line in shape:
        [left, right, top, bottom]
    '''

    if not rgb:
        width = img_tensor.size()[1]
        height = img_tensor.size()[0]
    else:
        width = img_tensor.size()[2]
        height = img_tensor.size()[1]
    x_0_0 = rnd.randint(0, int(width*0.1))
    x_0_1 = rnd.randint(0, int(width*0.1))
    x_1_0 = rnd.randint(int(width*0.9), int(width-1))
    x_1_1 = rnd.randint(int(width*0.9), int(width-1))
    y_0_0 = rnd.randint(0, int(height*0.1))
    y_0_1 = rnd.randint(0, int(height*0.1))
    y_1_0 = rnd.randint(int(height*0.9), int(height-1))
    y_1_1 = rnd.randint(int(height*0.9), int(height-1))

    # ====left_side====
    img_tensor, points_left = draw_line(x_0_0, y_0_0, x_0_1, y_1_0, img_tensor, thickness=thickness, rgb=rgb)
    # ====right_side====
    img_tensor, points_right = draw_line(x_1_0, y_0_1, x_1_1, y_1_1, img_tensor, thickness=thickness, rgb=rgb)
    # ====top====
    img_tensor, points_top = draw_line(x_0_0, y_0_0, x_1_0, y_0_1, img_tensor, thickness=thickness, rgb=rgb)
    # ====bottom====
    img_tensor, points_bottom = draw_line(x_0_1, y_1_0, x_1_1, y_1_1, img_tensor, thickness=thickness, rgb=rgb)

    lines = [points_left, points_right, points_top, points_bottom]
    return img_tensor, lines


def generate_images(width, height, number_of_batches, batch_size, noisy=False, color=False, noisy_p=0.1,
                    save=True, return_batches=False, destination_folder=''):
    '''Creates, edits and creates batches from images.

    Args:
        width (int): The width of the images
        height (int): The height of the images
        number_of_batches (int): The number of batches to create
        batch_size (int): The batch size
        noisy (bool): Whether noise shall be added to the images
        color (bool): Whether the noise shall be black or a color
        noisy_p (float): The percentage to get noised of the image
        save (bool): Whether the batches shall be saved
        return_batches (bool): Whether the batches shall be returned
        destination_folder (str): In which directory the batches shall be saved

    Returns:
        The batches and all points of the lines in the batch images.
    '''

    images = []
    batch_lines = []
    for i in range(number_of_batches*batch_size):
        img_tensor = torch.zeros(width, height)
        img_tensor, lines = generate_sheet(img_tensor)
        batch_lines.append(lines)
        if noisy:
            img_tensor = add_noise(img_tensor, noisy_p, color)
        images.append(img_tensor.unsqueeze(0))
    batches, batch_lines = create_batches(batch_size=batch_size, number_of_batches=number_of_batches, image_tensors=images, lines=batch_lines)
    if save:
        torch.save(batches, destination_folder + 'image_tensor_batches.pt')
    if return_batches:
        return batches.unsqueeze_(2), batch_lines


def modified_sigmoid(x):
    '''Modified sigmoid function for color gradients.

    Args:
        x (float): An archetype

    Returns:
        An image of the modified sigmoid function.
    '''

    return -(2/(1+math.exp(-x)))+2


def add_shadow(batch_tensor):
    '''Adds shadow to a given batch of images.

    Args:
        batch_tensor (torch.Tensor): A batch tensor of training images

    Returns:
        The batch with edited images.
    '''

    for image_index in range(batch_tensor.size()[0]):
        width = int(batch_tensor[image_index][0].size()[1] / 2)
        height = int(batch_tensor[image_index][0].size()[0] / 2)
        #num_quadrants = rnd.randint(1, 1)
        num_quadrants = 2   # performance
        shadow_area = np.random.randint(1, 3)

        if shadow_area == 1 or shadow_area == 4:
            x_0 = 0
        else:
            x_0 = width
        if shadow_area >= 3 or num_quadrants == 2:
            y_0 = 0
        else:
            y_0 = height
        x_1 = x_0 + width
        if num_quadrants == 2:
            y_1 = height * 2
        else:
            y_1 = y_0 + height
        for channel_index in range(batch_tensor[image_index].size()[0]):
            shadow_section = batch_tensor[image_index][channel_index][y_0:y_1, x_0:x_1]
            d = shadow_section.size()[1] / 1
            for column_index in range(shadow_section.size()[1]):
                if shadow_area == 1:
                    section = shadow_section[:, width-column_index-1].clone()
                    section[shadow_section[:, width-column_index-1] != 0] = modified_sigmoid(column_index / d)
                    shadow_section[:, width-column_index-1] = section
                else:
                    section = shadow_section[:, column_index].clone()
                    section[shadow_section[:, column_index] != 0] = modified_sigmoid(column_index / d)
                    shadow_section[:, column_index] = section

            batch_tensor[image_index][channel_index][y_0:y_1, x_0:x_1] = shadow_section
    return batch_tensor


def rand_bool(probability):
    '''Computes a random boolean with a given probability.

    Args:
        probability (float): The probability 'True' will be computed

    Returns:
        A boolean depending on the given probability.
    '''

    return rnd.random() < probability


def recolor_lines(batch, lines, p):
    '''Recolors and noises lines.

    Args:
        batch (torch.Tensor): A batch of training images
        lines (list): Contains all points of every line, sorted into lines, of every image of the batch
        p (float): The percentage to get recolored or noised of the image

    Returns:
        The batch with the edited images.

    Raises:
        IndexError: An error occurred trying editing a point out of the image
    '''

    for image_index in range(batch.size()[0]):
        img_lines = lines[image_index]
        recolor_points = []
        for line_index in range(4):
            line = img_lines[line_index]
            recolor_len = int(p * len(line))
            recolor_start_index = rnd.randint(0, len(line)-recolor_len)
            uncomplete_end_index = recolor_start_index + recolor_len
            recolor_points += line[recolor_start_index:uncomplete_end_index]
        gray = bool(rnd.getrandbits(1))
        if gray:
            color = rnd.uniform(0.6, 0.8)
        else:
            color = 0
        for channel in range(batch[image_index].size()[0]):
            for point in recolor_points:
                try:
                    if rand_bool(0.075):
                        batch[image_index][channel][tuple(reversed(point))] = 1
                    else:
                        batch[image_index][channel][tuple(reversed(point))] = color
                except IndexError:
                    pass
    return batch


def noise_text_bs(batch, p, mode=1):
    '''Adds noise to the text on images of a given batch.

    Args:
        batch (torch.Tensor): A batch of training images
        p (float): The percentage to get noised of the every image's text
        mode (int): Whether the noise shall be white or black (salt/pepper)

    Returns:
        The batch with edited images.
    '''

    return_batch_indices = []
    for image_index in range(batch.size()[0]):
        number_of_pixels = batch[image_index].size()[1] * batch[image_index].size()[2]
        number_of_noise = math.ceil(number_of_pixels * p)
        indices = rnd.choices(range(number_of_pixels), k=number_of_noise)
        return_batch_indices.append(indices)
        for channel_index in range(batch[image_index].size()[0]):
            image = cat_lines(batch[image_index][channel_index].squeeze())
            image[indices] = mode  # 0 --> black; 1 --> white
            batch[image_index][channel_index] = image.reshape((batch[image_index].size()[1], batch[image_index].size()[2])).unsqueeze(0)
    return batch


def generate_random_text(img_tensor):
    '''Creates random text on a given image.

    Args:
        img_tensor (torch.Tensor): A training image tensor

    Returns:
        The edited image tensor.
    '''

    font_faces = [cv.FONT_HERSHEY_SIMPLEX, cv.FONT_HERSHEY_DUPLEX, cv.FONT_HERSHEY_TRIPLEX]
    font_face = rnd.choice(font_faces)
    figure_list = list(string.ascii_letters + string.digits + ' ')
    font_scale = rnd.uniform(0.5, 1.25)
    thickness = rnd.randint(1, 8)       # (1, 4) or (1, 8) --> depends on model
    letter_size = cv.getTextSize(text='W', fontFace=font_face, fontScale=font_scale, thickness=thickness)
    letters_horizontal = int(img_tensor.size()[1] / letter_size[0][0])
    letter_size_y = letter_size[0][1] + rnd.randint(5, 25)
    letters_vertical = int(img_tensor.size()[2] / letter_size_y)
    num_lines = rnd.randint((letters_vertical-5), letters_vertical)
    y = rnd.randint(10, 50)
    x = rnd.randint(10, 50)
    img_tensor = img_tensor.permute(1, 2, 0).numpy()
    for line in range(num_lines):
        letters_per_line = rnd.randint((letters_horizontal - 7), letters_horizontal)
        line_str = ''
        for letter in range(letters_per_line):
            line_str += rnd.choice(figure_list)
        img_tensor = cv.putText(img=img_tensor, text=line_str, org=(x, y), fontFace=font_face, fontScale=font_scale,
                                color=(0, 0, 0), thickness=thickness)
        y += letter_size_y
    return img_tensor.get().transpose(2, 0, 1)


def generate_text_images(width_range, height_range, channels, number_of_batches, batch_size, salt, pepper):
    '''Creates batches of images.

    Creates batches of training and label images and
    therefore also worsens images by adding noise or shadows
    and / or recoloring.

    Args:
        width_range (tuple): The range for the random image width
        height_range (tuple): The range for the random image height
        channels (int): Number of channels (i.e. RGB --> 3)
        number_of_batches (int): The number of batches
        batch_size (int): The batch size
        salt (bool): Whether white noise shall be added
        pepper (bool): Whether black noise shall be added

    Returns:
        The batches of training and label images.
    '''

    batches = []
    for batch_number in range(number_of_batches):
        batch = []
        batch_lines = []
        height = rnd.randrange(height_range[0], height_range[1], 100)
        width = rnd.randrange(width_range[0], width_range[1], 100)
        for img in range(batch_size):
            image = torch.ones(channels, height, width)
            #thickness = np.random.randint(1, 3)    # used in model_34.2
            thickness = 1
            image, lines = generate_sheet(img_tensor=image, thickness=thickness, rgb=True)
            image = generate_random_text(img_tensor=image)
            batch.append(image.tolist())
            batch_lines.append(lines)
        batch = torch.FloatTensor(batch)
        batch_train = batch.clone()
        batch_train = add_shadow(batch_train)
        batch_train = recolor_lines(batch=batch_train, lines=batch_lines, p=0.25)    # commented out in model_34.2
        if salt:
            batch_train = noise_text_bs(batch=batch_train, p=0.45, mode=1)
        if pepper:
            batch_train = noise_text_bs(batch=batch_train, p=0.45, mode=0)      # p=0.25 in model_48
        batches.append((batch_train, batch))
    return batches


def get_text_batches(num_batches, batch_size, salt, pepper):
    '''Gets batches of images depending on given parameters.

    Args:
        num_batches (int): The number of batches
        batch_size (int): The batch size
        salt (bool): Whether white noise shall be added
        pepper (bool): Whether black noise shall be added

    Returns:
        The created batches.
    '''

    batches = generate_text_images((600, 1000), (800, 1500), 3, num_batches, batch_size, salt, pepper)
    return batches
