import denoise_edge
import text_layout_analysis
import layout_denoising
import create_final_image

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import threading

import sys
import subprocess
from os.path import dirname, abspath
from os import listdir, remove


class Window(QWidget):

    def __init__(self, screen_resolution, w_f, h_f):
        '''Initializes window with all widges

        Args:
            screen_resolution (PyQt5.QtCore.QRect): The screen's resolution
            w_f (float): The factor to multiply the real width with
            h_f (float): The factor to multiply the real height with
        '''

        super().__init__()
        self.model_mode = 0

        self.width = int(screen_resolution.width() * w_f)
        self.height = int(screen_resolution.height() * h_f)
        self.x_border_distance = int(self.width / 9)
        self.y_border_distance = int(self.height/20)

        self.ratio = int(self.width / 1920)

        QToolTip.setFont(QFont('Arial', 10))
        self.setWindowTitle('Scanner')
        self.setFixedSize(self.width, self.height)

        self.input_img_info = QLabel(self)
        self.input_img_info.setGeometry(self.x_border_distance, self.y_border_distance, 100*self.ratio, 32*self.ratio)
        self.input_img_info.setText('Image Path:')

        self.button_input = QPushButton('Ok', self)
        self.button_input.move(self.x_border_distance*2.5+100, self.y_border_distance)
        self.button_input.clicked.connect(self.get_input_image)

        self.input_img_path_line = QLineEdit(self)
        self.input_img_path_line.setGeometry(self.x_border_distance+(100*self.ratio), self.y_border_distance,
                                                               self.x_border_distance*1.5, 32*self.ratio)

        self.button_output = QPushButton('Edit', self)
        self.button_output.move(self.x_border_distance, self.height-self.y_border_distance-self.button_output.height())
        self.button_output.clicked.connect(self.get_output_image)

        self.model_mode_combo = QComboBox(self)
        self.model_mode_combo.setGeometry(self.x_border_distance-self.button_output.width(),
                                          self.height - self.y_border_distance - self.button_output.height(),
                                          75*self.ratio, 32*self.ratio)
        self.model_mode_combo.addItems(['CPU', 'GPU'])
        self.model_mode_combo.setCurrentIndex(self.model_mode)
        self.model_mode_combo.activated[int].connect(self.model_mode_changed)

        self.output_img_info = QLabel(self)
        self.output_img_info.setGeometry(self.x_border_distance + (125 * self.ratio),
                                         self.height-self.y_border_distance-self.button_output.height(), 150*self.ratio,
                                         32*self.ratio)
        self.output_img_info.setText('Output Image Name:')

        self.output_img_name = QLineEdit(self)
        self.output_img_name.setGeometry(self.x_border_distance + (280*self.ratio),
                                         self.height-self.y_border_distance-self.button_output.height(),
                                             self.x_border_distance * 1.5, 32*self.ratio)

        self.input_img_label = QLabel(self)
        self.output_img_label = QLabel(self)

        self.show()

    def model_mode_changed(self, i):
        self.model_mode = i

    def get_updated_image_shape(self, image_size):
        '''Computes the new shape for an image

        Computes the new shape to show an image in the app.

        Args:
            image_size (tuple): The image's resolution

        Returns:
            The new image size and the distance between the top of the app and the upper edge of the shown image.
        '''

        img_width, img_height = image_size
        img_new_width = int(self.width / 3)
        img_new_height = int((img_new_width * img_height) / img_width)
        y_border_distance = int((self.height - img_new_height) / 2)
        return img_new_width, img_new_height, y_border_distance

    def get_input_image(self):
        '''Shows input image in the app'''

        input_img = QPixmap(self.input_img_path_line.text())
        img_new_width, img_new_height, y_border_distance = self.get_updated_image_shape(image_size=
                                                                                        (input_img.size().width(),
                                                                                         input_img.size().height()))
        self.input_img_label.setGeometry(self.x_border_distance, y_border_distance,
                                         img_new_width, img_new_height)
        self.input_img_label.setPixmap(input_img.scaled(img_new_width, img_new_height))

        self.update()

    def get_output_thread(self):
        '''Edits input image and saves it'''

        output_img_name = self.output_img_name.text()
        if output_img_name == '':
            output_img_name = 'final_image'
        path = 'final_image/' + output_img_name + '.png'
        corner_pixel_coords, sheet_state_dict = denoise_edge.get_edge_data(img_file=self.input_img_path_line.text(),
                                                                           model_mode=self.model_mode)
        box_coords = text_layout_analysis.get_layout(img_file=self.input_img_path_line.text(),
                                                     sheet_volume=sheet_state_dict['volume'],
                                                     corner_points=corner_pixel_coords)
        layout_denoising.edit_layout_parts(model_mode=self.model_mode)
        create_final_image.get_final_image(box_coords=box_coords, corner_pixel_coords=corner_pixel_coords,
                                           sheet_state_dict=sheet_state_dict, out_img_path=path)

    def get_output_image(self):
        '''Shows the edited output image

        Computes new shape for the edited output image to show it in the app.
        Also opens the image in the photo viewer.
        '''
        output_img_name = self.output_img_name.text()
        if output_img_name == '':
            output_img_name = 'final_image'
        path = 'final_image/' + output_img_name + '.png'
        output_thread = threading.Thread(target=self.get_output_thread)
        output_thread.start()
        img_new_width, img_new_height, y_border_distance = self.get_updated_image_shape(image_size=(2480, 3508))
        self.output_img_label.setGeometry(self.width-self.x_border_distance-img_new_width, y_border_distance,
                                          img_new_width, img_new_height)
        output_thread.join()
        output_img = QPixmap(path)
        self.output_img_label.setPixmap(output_img.scaled(img_new_width, img_new_height))

        self.update()

        layout_part_files = listdir('layout_parts')
        for layout_part_file in layout_part_files:
            remove('layout_parts/' + layout_part_file)

        path = dirname(abspath(__file__)) + '\\' + path.replace('/', '\\')
        image_viewer_from_command_line = {'linux': 'xdg-open',
                                          'win32': 'explorer',
                                          'darwin': 'open'}[sys.platform]
        subprocess.run([image_viewer_from_command_line, path])


if __name__ == '__main__':
    app = QApplication(sys.argv)
    screen_resolution = app.desktop().screenGeometry()
    w = Window(screen_resolution=screen_resolution, w_f=0.5, h_f=0.5)
    sys.exit(app.exec_())
