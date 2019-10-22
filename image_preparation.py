from PIL import Image
from PIL import ImageFilter
import torchvision.transforms as transforms
from torch.autograd import Variable
import data_generator


transform = transforms.Compose([
    transforms.ToTensor(),
])


def hard_edges(img_tensor, threshold):
    '''Edits edges by 'heavisiding' color values with given threshold.

    Args:
        img_tensor (torch.Tensor): An edge image tensor
        threshold (float): The threshold to heaviside the edges

    Returns:
        The image tensor with thresholded edges.
     '''

    return_img = img_tensor
    for channel in range(img_tensor.size()[0]):
        tensor = data_generator.cat_lines(img_tensor[channel])
        for i in range(len(tensor)):
            if tensor[i] <= threshold:
                tensor[i] = 0
            else:
                tensor[i] = 1
        return_img[channel] = tensor.reshape(img_tensor.size()[1], img_tensor.size()[2])
    return return_img


def to_gray(img):
    '''Converts the given image into a one channel image.

    Args:
        img (PIL.Image.Image): The image to convert

    Returns:
        The edited image.
    '''

    alpha = img.convert('RGBA').split()[-1]
    back_g = Image.new('RGBA', img.size, (255, 255, 255)+(255,))
    back_g.paste(img, mask=alpha)
    return_img = back_g.convert('L')
    return return_img


def get_edged_image(img_file):
    '''Resizes and converts image. Also finds edges and 'heavisides' them.

    Args:
        img_file (str): The image to edit

    Returns:
        The edited image.
    '''

    avg_height = 400
    avg_width = 300
    img = Image.open(img_file)
    img = img.resize((avg_width, avg_height), Image.ANTIALIAS)
    img_gray = to_gray(img)
    img_with_edges = img_gray.filter(ImageFilter.FIND_EDGES)
    img_tensor = Variable(transform(img_with_edges))
    hard_img = hard_edges(img_tensor, 0.15)
    hard_img = hard_img.squeeze(0)

    return hard_img
