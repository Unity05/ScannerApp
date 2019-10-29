import torch
from torch.autograd import Variable
import image_preparation
import rotation_computer
from models import DenoisingAutoencoderSheetEdges


def get_edge_data(img_file, model_mode):
    '''Denoises an edged image and computes the sheet data out of it.

    Args:
        img_file (str): The path of the image

    Returns:
        The points of the sheet's corners and the sheet state of shape:
        {'top': x, 'bottom': x, 'left': x, 'right': x, 'angle': x,
        'angle_top': x, 'angle_bot': x, 'volume': x}
    '''

    model = DenoisingAutoencoderSheetEdges()
    model.load_state_dict(torch.load('models/model_2.pth').state_dict())
    model.threshold = 0.75
    model.hard_mode = True

    img = image_preparation.get_edged_image(img_file=img_file)

    with torch.no_grad():
        if model_mode == 0:
            img = Variable(img)
            model.eval()
        elif torch.cuda.is_available():
            img = Variable(img).cuda()
            model.cuda().eval()
        else:
            img = Variable(img)
            model.eval()

        new_img = model(img.unsqueeze(0).unsqueeze(0), model_mode)
        corner_pixel_coords, sheet_state_dict = rotation_computer.get_rotation(edge_img_tensor=new_img.squeeze(0).squeeze(0))

    return corner_pixel_coords, sheet_state_dict
