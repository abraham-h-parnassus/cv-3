from tkinter import *
from tkinter import filedialog as fd
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
from numpy import random
from skimage.draw import line, circle_perimeter

from cv3 import process


class Application:

    def __init__(self, root):
        self.root = root
        self.frame = ttk.Frame(root, padding=10)
        self.image = None
        self.main_section = ttk.Labelframe(self.frame, text="Input Image")
        self.preview_section = ttk.Labelframe(self.frame, text="Input Preview")
        self.result_section_clean = ttk.Labelframe(self.frame, text="Result (No Noise)")
        self.result_section_noisier = ttk.Labelframe(self.frame, text="Result (Noiser)")
        self.result_section_noisy = ttk.Labelframe(self.frame, text="Result (Very Noisy)")

        self.choose_image_button = ttk.Button(self.main_section, text="Choose image",
                                              command=self.select_image)
        self.generate_lines_button = ttk.Button(self.main_section, text="Generate lines",
                                                command=self.generate_lines)
        self.generate_circles_button = ttk.Button(self.main_section, text="Generate circles",
                                                  command=self.generate_circles)

        self.detect_lines_button = ttk.Button(self.main_section, text="Detect Lines",
                                              command=self.detect_lines)
        self.detect_circles_button = ttk.Button(self.main_section, text="Detect Circles",
                                                command=self.detect_circles)
        self.preview_label = Label(self.preview_section, borderwidth=0)
        self.clean_result_label = Label(self.result_section_clean, borderwidth=0)
        self.noiser_result_label = Label(self.result_section_noisier, borderwidth=0)
        self.noisy_result_label = Label(self.result_section_noisy, borderwidth=0)

    def show(self):
        self.frame.grid()
        self.main_section.grid(column=0, row=0)
        self.choose_image_button.grid(column=0, row=0)
        self.generate_lines_button.grid(column=1, row=0)
        self.detect_lines_button.grid(column=1, row=1)
        self.generate_circles_button.grid(column=2, row=0)
        self.detect_circles_button.grid(column=2, row=1)

        self.preview_section.grid(column=0, row=1)
        self.preview_label.grid(column=0, row=0)

        self.result_section_clean.grid(column=1, row=1)
        self.result_section_noisier.grid(column=1, row=2)
        self.result_section_noisy.grid(column=1, row=3)

        self.clean_result_label.grid(column=0, row=0)
        self.noiser_result_label.grid(column=0, row=0)
        self.noisy_result_label.grid(column=0, row=0)

        self.root.mainloop()

    def select_image(self):
        path = fd.askopenfile()
        self.image = cv2.imread(path.name)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self._show_image(self.preview_label, self.image)

    def generate_lines(self):
        image = np.zeros((200, 200))
        for i in range(1, 5):
            points = random.randint(0, high=200, size=4, dtype=int)
            image[line(points[0], points[1], points[2], points[3])] = 255
        self.image = image
        self._show_image(self.preview_label, self.image)

    def generate_circles(self):
        image = np.zeros((200, 200))
        sizes = [31, 35, 60]
        for size in sizes:
            points = random.randint(size, high=200 - size, size=2, dtype=int)
            circy, circx = circle_perimeter(points[0], points[1], size,
                                            shape=image.shape)
            image[circy, circx] = 255
        self.image = image
        self._show_image(self.preview_label, image)

    def detect_lines(self):
        result = process(self.image)

        self._show_image(self.clean_result_label, self.down_scale(result[0]))
        self._show_image(self.noiser_result_label, self.down_scale(result[1]))
        self._show_image(self.noisy_result_label, self.down_scale(result[2]))

    def detect_circles(self):
        result = process(self.image, detect_circles=True)

        self._show_image(self.clean_result_label, self.down_scale(result[0]))
        self._show_image(self.noiser_result_label, self.down_scale(result[1]))
        self._show_image(self.noisy_result_label, self.down_scale(result[2]))

    def _show_image(self, panel: Label, image):
        photo_image = ImageTk.PhotoImage(image=Image.fromarray(image, None if len(image.shape) < 3 else 'RGB'))
        panel.photo = photo_image
        panel.configure(image=photo_image)
        self.root.update()

    def down_scale(self, image):
        scale_percent = 40
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        return cv2.resize(image, (width, height))


def show_ui():
    root = Tk()
    Application(root).show()
    root.mainloop()


if __name__ == '__main__':
    show_ui()
