from src.plugin_interface import PluginInterface
from PyQt6.QtWidgets import QWidget
from PyQt6 import QtGui
from .ui_main import Ui_Form
from skimage.segmentation import watershed
from scipy import ndimage
import os
import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
import shutil

class Controller(QWidget):
    def __init__(self, model):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.model = model
        self.image = None
        self.render_image = False   # variabel agar tidak bisa load img ketika sudah ada img
        self.path_img_save = "./plugins/moilapp-plugin-histologi-bat/img_tmp"
        self.set_stylesheet()

    def set_stylesheet(self):
        self.ui.label.setStyleSheet(self.model.style_label())
        self.ui.label_2.setStyleSheet(self.model.style_label())
        self.ui.label_4.setStyleSheet(self.model.style_label())
        self.ui.label_5.setStyleSheet(self.model.style_label())
        self.ui.label_6.setStyleSheet(self.model.style_label())
        self.ui.label_7.setStyleSheet(self.model.style_label())
        self.ui.label_8.setStyleSheet(self.model.style_font_12())
        self.ui.lbl_cell.setStyleSheet(self.model.style_font_12())

        self.ui.img_grafik.setStyleSheet(self.model.style_label())
        self.ui.img_ori.setStyleSheet(self.model.style_label())
        self.ui.img_dist.setStyleSheet(self.model.style_label())
        self.ui.img_morph.setStyleSheet(self.model.style_label())
        self.ui.img_canny.setStyleSheet(self.model.style_label())
        self.ui.img_label.setStyleSheet(self.model.style_label())

        self.ui.btn_load.setStyleSheet(self.model.style_pushbutton())
        self.ui.btn_clear.setStyleSheet(self.model.style_pushbutton())
        self.ui.btn_crop.setStyleSheet(self.model.style_pushbutton())
        self.ui.btn_save.setStyleSheet(self.model.style_pushbutton())

        self.ui.btn_load.clicked.connect(self.load_image_1)
        self.ui.btn_crop.clicked.connect(self.load_image_crop)
        self.ui.btn_clear.clicked.connect(self.clearImg)
        self.ui.btn_save.clicked.connect(self.save_img)

        # self.checkDir(f"./plugins/moilapp-plugin-histologi-bat/img_tmp")

        self.checkDir(self.path_img_save)

    def save_img(self):
        if self.render_image == False: return

        if os.path.exists("./plugins/moilapp-plugin-histologi-bat/img_save"):
            os.system("rm -R ./plugins/moilapp-plugin-histologi-bat/img_save")
        shutil.copytree(f"{self.path_img_save}", "./plugins/moilapp-plugin-histologi-bat/img_save/")

    def clearImg(self):
        self.ui.img_ori.clear()
        self.ui.img_morph.clear()
        self.ui.img_dist.clear()
        self.ui.img_label.clear()
        self.ui.img_grafik.clear()
        self.ui.img_canny.clear()
        self.ui.lbl_cell.clear()
        self.image_original = None
        self.image = self.image_original
        self.render_image = False

    def load_image_1(self):
        if self.render_image: return    # kalo true bakal kembali
        file = self.model.select_file()
        if file:
            if file:
                self.moildev = self.model.connect_to_moildev(parameter_name=file)
            if file.partition('.')[-1].upper() in ['APNG', 'AVIF', 'GIF', 'JPEG', 'JPG', 'PNG', 'SVG', 'TIFF', 'WEBP']:
                self.image_original = cv2.imread(file)
                self.image = self.image_original.copy()
                self.render_image = True
                self.show_to_ui_img_1(file)

    def load_image_crop(self):
        if self.render_image: return
        self.checkDir(f"{self.path_img_save}/crop")
        file = self.model.select_file()
        if file:
            if file:
                self.moildev = self.model.connect_to_moildev(parameter_name=file)
            self.image_original = cv2.imread(file)
            self.image = self.image_original.copy()
            self.render_image = True
            self.show_to_ui_img_crop(file)

    def show_to_ui_img_1(self, img_path):
        self.checkDir(f"{self.path_img_save}/img_processing")
        img = cv2.imread(img_path)
        size = 400

        self.morp_opr(img)

        switch_obj = cv2.imread(f"{self.path_img_save}/img_processing/switch-obj.png")

        self.labelling(switch_obj)
        self.count_cell()

        canny = cv2.imread(f"{self.path_img_save}/img_processing/canny.png")
        distace = cv2.imread(f"{self.path_img_save}/img_processing/distance.png")

        self.model.show_image_to_label(self.ui.img_ori, img, size)
        self.model.show_image_to_label(self.ui.img_morph, switch_obj, size)
        self.model.show_image_to_label(self.ui.img_canny, canny, size)
        self.model.show_image_to_label(self.ui.img_dist, distace, size)
        self.model.show_image_to_label(self.ui.img_label, self.image_original, size)

        self.graph()
        graph = cv2.imread(f"{self.path_img_save}/img_processing/graph.png")
        self.model.show_image_to_label(self.ui.img_grafik, graph, size)
        plt.close("all")

    def morp_opr(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        img_mop = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50)))
        img_mop = cv2.morphologyEx(img_mop, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)))
        cv2.imwrite(f"{self.path_img_save}/img_processing/morp.png", img_mop)

        # switch objek
        img_morp = img_mop.copy()
        for i in range(0, img_morp.shape[0]):
            for j in range(0, img_morp.shape[1]):
                px = 255 if img_morp[i][j] == 0 else 0
                img_morp[i][j] = px
        cv2.imwrite(f"{self.path_img_save}/img_processing/switch-obj.png", img_morp)

        # return img_mop

    def switch_obj(self):
        img_morp = cv2.imread(f"{self.path_img_save}/img_processing/morp.png", 0)
        for i in range(0, img_morp.shape[0]):
            for j in range(0, img_morp.shape[1]):
                px = 255 if img_morp[i][j] == 0 else 0
                img_morp[i][j] = px
        cv2.imwrite(f"{self.path_img_save}/img_processing/switch-obj.png", img_morp)
        # return  img_morp

    def labelling(self, switch_obj):
        # morph = cv2.imread(f"{self.path_img_save}morp.png")
        canny = cv2.Canny(switch_obj, 100, 200)
        cv2.imwrite(f"{self.path_img_save}/img_processing/canny.png", canny)

        kontur1, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(self.image_original, kontur1, -1, (0, 255, 0), 5)

    def count_cell(self):
        img_switch = cv2.imread(f"{self.path_img_save}/img_processing/switch-obj.png")
        gray = cv2.cvtColor(img_switch, cv2.COLOR_BGR2GRAY)
        threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        kernel = np.ones((3, 3), np.uint8)
        mop_open = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=2)

        # dilate = cv2.dilate(mop_open, kernel, iterations=2)

        distance_trans = cv2.distanceTransform(mop_open, cv2.DIST_L2, cv2.DIST_MASK_5)
        dist_thres = cv2.threshold(distance_trans, 0.24 * distance_trans.max(), 255, cv2.THRESH_BINARY)[1]
        cv2.imwrite(f"{self.path_img_save}/img_processing/distance.png", dist_thres)

        img_distace = cv2.imread(f"{self.path_img_save}/img_processing/distance.png", 0)
        kontur2, _ = cv2.findContours(img_distace, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # mc = 264.5833  # 1 px = 264.5833 micrometer
        self.ui.lbl_cell.setText(f"{len(kontur2)}")

        for i in range(0, len(kontur2)):
            ((x, y), r) = cv2.minEnclosingCircle(kontur2[i])
            wide = cv2.contourArea(kontur2[i], False)
            if wide == 0: continue
            cv2.putText(self.image_original, f"{int(wide)}px", (int(x) - 4, int(y)), cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 255), 1)
            # cv2.putText(img, f"{int(wide * mc)}μm", (int(x) - 4, int(y)), cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 255), 1)

    def show_to_ui_img_crop(self, img_path):
        dir_img_save_path = f"{self.path_img_save}/crop"

        self.checkDir(dir_img_save_path)

        self.model.show_image_to_label(self.ui.img_ori, self.image_original, 620)
        self.crop_img(dir_img_save_path, img_path)

    def checkDir(self, path_dir):
        if (os.path.exists(f"{path_dir}")):
            if (os.path.isdir(f"{path_dir}")):
                os.system(f"rm -R {path_dir}")
                os.mkdir(f"{path_dir}")
                # cv2.imwrite(img_save_path, image)
        else:
            os.mkdir(f"{path_dir}")

    def crop_img(self, dir_path, img_path):
        # jumlah potongan gambar
        jmh_crop = 4

        # gambar yg sudah di labeling
        img = cv2.imread(img_path)
        height, width = img.shape[:2]

        # menghitung ukuran gambar untuk dipotong
        row_start, row_end = self.count_crop_img(jmh_crop, height)
        col_start, col_end = self.count_crop_img(jmh_crop, width)

        # img_save_path = f"{dir_path}/count_cell.png"

        self.checkDir(dir_path)

        for i in range(0, jmh_crop):
            for j in range(0, jmh_crop):
                # memotong gambar
                cropped = img[row_start[i]:row_end[i], col_start[j]:col_end[j]]

                # menyimpan gambar
                cv2.imwrite(f"{dir_path}/img_crop_{i + 1}_{j + 1}.png", cropped)

    def count_crop_img(self, jmh_crop, size_img):
        # menghitung dimensi gambar yg akan dipotong sesuai kebutuhan
        start_crop = []
        end_crop = []

        # perhitungan ukuran potongan gambar
        size_crop = int(size_img / jmh_crop)
        for i in range(0, jmh_crop + 1):
            # fungsi append untuk menambahkan nilai list
            end_crop.append(i * size_crop)
            start_crop.append(end_crop[i] - size_crop)

        # fungsi pop untuk menhapus list index 0
        start_crop.pop(0)
        end_crop.pop(0)
        return start_crop, end_crop

    def graph(self):    # untuk grafik

        # Fixing random state for reproducibility
        np.random.seed(19680801)

        dt = 0.01
        t = np.arange(0, 30, dt)
        nse1 = np.random.randn(len(t))                 # white noise 1
        nse2 = np.random.randn(len(t))                 # white noise 2

        # Two signals with a coherent part at 10 Hz and a random part
        s1 = np.sin(2 * np.pi * 10 * t) + nse1
        s2 = np.sin(2 * np.pi * 10 * t) + nse2

        fig, axs = plt.subplots(2, 1, layout='constrained')
        axs[0].plot(t, s1, t, s2)
        axs[0].set_xlim(0, 2)
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('s1 and s2')
        axs[0].grid(True)

        cxy, f = axs[1].cohere(s1, s2, 256, 1. / dt)
        axs[1].set_ylabel('Coherence')

        plt.savefig(f"{self.path_img_save}/img_processing/graph.png")

        plt.show()

class HistologiBat(PluginInterface):
    def __init__(self):
        super().__init__()
        self.widget = None
        self.description = "This is a plugins application"

    def set_plugin_widget(self, model):
        self.widget = Controller(model)
        return self.widget

    def set_icon_apps(self):
        return "icon.png"

    def change_stylesheet(self):
        self.widget.set_stylesheet()
