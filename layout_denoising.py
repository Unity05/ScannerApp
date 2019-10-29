import torch
from torch.autograd import Variable
import cv2 as cv
from os import listdir


def edit_layout_parts(model_mode):
    '''Edits saved layout part images.

    Reads saved layout part images and processes them through two nets.
    It replaces every original layout part image by the edited one.
    '''

    if model_mode == 0:
        model = torch.load('models/layout_parts_model_48.pth', map_location=torch.device('cpu'))
        model_1 = torch.load('models/layout_parts_model_34.2.pth', map_location=torch.device('cpu'))
        model.eval()
        model_1.eval()
    elif torch.cuda.is_available():
        model = torch.load('models/layout_parts_model_48.pth')
        model_1 = torch.load('models/layout_parts_model_34.2.pth')
        model.cuda().eval()
        model_1.cuda().eval()
    else:
        model = torch.load('models/layout_parts_model_48.pth', map_location=torch.device('cpu'))
        model_1 = torch.load('models/layout_parts_model_34.2.pth', map_location=torch.device('cpu'))
        model.eval()
        model_1.eval()

    layout_part_images = listdir('layout_parts')
    with torch.no_grad():
        for layout_part_file in layout_part_images:
            img = Variable(torch.Tensor(cv.imread('layout_parts/' + layout_part_file)).permute(2, 0, 1).unsqueeze(0))
            if torch.cuda.is_available() and model_mode == 1:
                img = img.cuda()
            new_img = model(img)
            new_img = model_1(new_img)
            new_img = new_img.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            new_img = new_img * 255.0   # cv.imwrite saves it as 0-255 rgb values
            cv.imwrite('layout_parts/' + layout_part_file, new_img)
