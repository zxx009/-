from ultralytics import YOLO
import cv2
import multiprocessing
import sys
import os
import shutil
import threading
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from datetime import datetime
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog, QDialog, QLabel as QDialogLabel, QMessageBox, QProgressDialog, QTableWidgetItem
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtCore import Qt, pyqtSignal
from guinew3 import Ui_Form, ImageSizeDialog, ClickableLabel
from orb import process_orb_images
from sift import process_sift_images
from template import process_templated_images
from muban1 import template_pose_estimation
from muban2 import estimate_pose
from pose_estimator import deep_pose_estimation
import net
from config import Config
import argparse
import urso
import custom_dataset
import utils
import random

MODEL_DIR = os.path.abspath("./models")
DEFAULT_LOGS_DIR = os.path.join(MODEL_DIR, "logs")
DATA_DIR = os.path.abspath("./datasets")


def convert_ndarray_to_qpixmap(cv_img):
    # 将 BGR 转为 RGB
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

    # 获取图像的宽度、高度和每行字节数
    height, width, channels = rgb_image.shape
    bytes_per_line = 3 * width

    # 创建 QImage 对象
    q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

    # 转换为 QPixmap 并返回
    return QPixmap.fromImage(q_image)

import os

def validate_dir(file_path):
    # 支持的图片文件扩展名
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

    # 检查文件路径是否为空
    if not file_path:
        print("Error: The file path is empty.")
        return False

    # 检查路径是否存在且是目录
    if not os.path.isdir(file_path):
        print(f"Error: The path '{file_path}' is not a valid directory.")
        return False

    # 遍历目录查找图片文件
    has_image = any(
        os.path.splitext(file)[1].lower() in image_extensions
        for file in os.listdir(file_path)
    )

    if not has_image:
        print(f"Error: No image files found in directory '{file_path}'.")
        return False

    print(f"Success: Image files found in directory '{file_path}'.")
    return True

def validate_file(file_path, expected_extensions):
    # 检查文件路径是否为空
    if not file_path:
        print(file_path)
        print("Error: The file path is empty.")
        return False

    # 检查文件扩展名是否在预期的扩展名列表中
    if not any(file_path.endswith(ext) for ext in expected_extensions):
        print(f"Error: The file must have one of the following extensions: {', '.join(expected_extensions)}")
        return False

    # 检查文件是否存在
    if not os.path.isfile(file_path):
        print("Error: The file does not exist.")
        return False

    # 文件通过所有检查
    return True


class MyWindow(QWidget, Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.sample_img_filepath = ""
        self.source_img_filepath = ""
        self.model_filepath = ""
        self.recon_sample_dir = "./Recon/train/images"      #暂固定，可改为自己选择
        self.pose_sample_dir= './datasets/custom'
        self.output_path = ""
        self.class_filepath = "./Recon/train/labels/classes.txt"   #暂固定，可改为自己选择
        self.h5model_filepath = ""
        self.class_names = self.class_names = {0: 'main-body', 1: 'Left-Solar-Array', 2:'Right-Solar-Array',3: 'Radar', 4: 'LENS', 5: 'Antenna'}
        self.class_colors = {
        0: (255, 0, 0),  # 红色
        1: (0, 255, 0),  # 绿色
        2: (0, 0, 255),  # 蓝色
        3: (0, 255, 255),  # 黄色
        4: (255, 255, 0),  # 天蓝色
        5: (255, 0, 255),  # 品红
    }
        self.load_sample.clicked.connect(self.on_load_sample_dir)
        self.load_source_img.clicked.connect(self.on_load_source_img)
        self.load_model.clicked.connect(self.on_load_model)
        self.load_h5model.clicked.connect(self.on_load_h5model)
        self.pushButton_output_path.clicked.connect(self.on_load_output)
        # self.deep_learnin_Compldent.clicked.connect(self.on_deep_learning_generate)
        self.traditional_Pose.clicked.connect(self.on_traditional_Pose_generate)
        self.deep_learnin_pose.clicked.connect(self.on_deep_Learning_Pose_generate)
        self.SIFT_Compldent.clicked.connect(self.on_traditional_Compldent_SIFT_generate)
        self.ORB_Compldent.clicked.connect(self.on_traditional_Compldent_ORB_generate)
        self.sample_img.clicked.connect(self.show_image_size)  # 为 sample_img 连接点击信号
        self.source_img.clicked.connect(self.show_image_size)  # 为 source_img 连接点击信号
        self.deep_learning_img.clicked.connect(self.show_image_size)  # 为 source_img 连接点击信号
        self.traditional_Pose_img.clicked.connect(self.show_image_size)  # 为 source_img 连接点击信号
        self.traditional_Compldent_img.clicked.connect(self.show_image_size)  # 为 source_img 连接点击信号
        self.Templated_Compldent.clicked.connect(self.on_traditional_Compldent_Templated_generate)  # 为 Templated_Compldent 连接点击信号 todo
        self.PushBottom_choose_pose_template.clicked.connect(self.on_load_sample_img)  # 为 PushBottom_choose_pose_template 连接点击信号
        self.deep_learning_pose_img.clicked.connect(self.show_image_size)  # 为 source_img 连接点击信号

        # 创建一个 QFont 对象，设置字体和大小
        font = QFont("Arial", 20)  # 设置字体为 Arial，大小为 20
        self.setFont(font)  # 设置整个窗口的字体

        self.setWindowTitle("空间目标数据增广技术研究")  # 设置窗口标题

    def on_load_sample_img(self):
        # 这里是按钮被点击时调用的方法
        file_path, _ = QFileDialog.getOpenFileName(self, '选择样本图片', r".\\",
                                                   '图片 (*.png *.xpm *.jpg *.jpeg *.bmp)')
        if file_path:
            self.sample_img_filepath = file_path
            # 加载图片并显示在 QLabel 上
            pixmap = QPixmap(file_path)
            if pixmap.isNull():
                print(f"无法加载图片: {file_path}")
            else:
                self.sample_img.setPixmap(pixmap)
                self.sample_img.setScaledContents(True)
    # def on_load_sample_img(self):
    #     # 修改为选择文件夹
    #     dir_path = QFileDialog.getExistingDirectory(self, '选择样本图片文件夹', r".\\")
    #     if dir_path:
    #         self.sample_dir = dir_path  # 保存文件夹路径
    #
    #         # 显示文件夹中的第一张有效图片（可选）
    #         expected_img_extensions = ['.png', '.xpm', '.jpg', '.jpeg', '.bmp']
    #         for file in os.listdir(dir_path):
    #             if os.path.splitext(file)[1].lower() in expected_img_extensions:
    #                 img_path = os.path.join(dir_path, file)
    #                 pixmap = QPixmap(img_path)
    #                 if not pixmap.isNull():
    #                     self.sample_img.setPixmap(pixmap)
    #                     self.sample_img.setScaledContents(True)
    #                 break  # 只显示第一张有效图片

    def on_load_sample_dir(self):
        # 这里是按钮被点击时调用的方法
        file_path = QFileDialog.getExistingDirectory(self, '选择样本图片文件夹', r".\\")
        if file_path:
            self.recon_sample_dir = file_path
            self.label_choosed_dir.setText(file_path)
    def on_load_source_img(self):
        # 这里是按钮被点击时调用的方法
        file_path, _ = QFileDialog.getOpenFileName(self, '选择源图片', r".\\", '图片 (*.png *.xpm *.jpg *.jpeg *.bmp)')
        if file_path:
            self.source_img_filepath = file_path
            # 加载图片并显示在 QLabel 上
            pixmap = QPixmap(file_path)
            if pixmap.isNull():
                print(f"无法加载图片: {file_path}")
            else:
                self.source_img.setPixmap(pixmap)
                self.source_img.setScaledContents(True)

    def on_load_model(self):
        # 这里是按钮被点击时调用的方法
        file_path, _ = QFileDialog.getOpenFileName(self, '打开YOLO模型', r".\\", '模型 (*.pt)')
        if file_path:
            self.model_filepath = file_path

    def on_load_output(self):
        # 这里是按钮被点击时调用的方法
        file_path = QFileDialog.getExistingDirectory(self, '选择结果图片输出文件夹', r".\\")
        if file_path:
            self.output_path = file_path

    def on_load_h5model(self):
        # 这里是按钮被点击时调用的方法
        file_path, _ = QFileDialog.getOpenFileName(self, '打开YOLO模型', r".\\", '模型 (*.h5)')
        if file_path:
            self.h5model_filepath = file_path

    def on_deep_learning_generate(self):
        expected_img_extensions = ['.png', '.xpm', '.jpg', '.jpeg', '.bmp']
        expected_model_extensions = ['.pt']
        # validate_file(self.sample_img_path,expected_img_extensions)
        flag = validate_file(self.source_img_filepath, expected_img_extensions)
        if flag == False:
            QMessageBox.warning(self, '警告', '请选择源图片！')
            return
        flag = validate_file(self.model_filepath, expected_model_extensions)
        if flag == False:
            QMessageBox.warning(self, '警告', '请选择YOLO模型！')
            return
        output_dir = self.output_path
        if not os.path.isdir(self.output_path):
            reply = QMessageBox.question(self, '警告', '输出文件夹路径错误，是否直接使用当前路径下的OUTPUT文件夹？',
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                # 使用当前路径下的OUTPUT文件夹作为输出路径
                self.output_path = r".\output"
                # 如果output文件夹不存在，则创建它
                if not os.path.isdir(self.output_path):
                    os.makedirs(self.output_path)
                QMessageBox.information(self, '提示', f'已使用当前路径下的output文件夹：{self.output_path}')
            else:
                # 用户选择不重设，可能需要其他操作
                QMessageBox.warning(self, '警告', '请重新设置输出文件夹路径！')
            return
        #源代码
        # model = YOLO(self.model_filepath)
        #
        # img = cv2.imread(self.source_img_filepath)
        #
        # results = model.predict(
        #     source=self.source_img_filepath,
        #     save=False,
        #     show=False
        # )
        # recognized_class_ids = set()  # 记录识别到的类别ID
        # for r in results:
        #     boxes = r.boxes.xyxy.cpu().numpy()
        #     cls_ids = r.boxes.cls.cpu().numpy()  # 获取类别ID
        #     confs = r.boxes.conf.cpu().numpy()  # 获取置信度
        #
        #     for box, cls_id, conf in zip(boxes, cls_ids, confs):
        #         x1, y1, x2, y2 = map(int, box[:4])
        #         color = self.class_colors[int(cls_id)]
        #         label = f"{self.class_names[int(cls_id)]} {conf:.2f}"
        #         # 画框
        #         cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=1)
        #         # 写标签
        #         # cv2.putText(img, label, (x1, y1 - 5),
        #         #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        #         recognized_class_ids.add(int(cls_id))
        # # --- 添加右上角图例 ---
        # legend_x, legend_y = 10, 10
        # for idx, (cls_id, name) in enumerate(self.class_names.items()):
        #     color = self.class_colors[int(cls_id)]
        #     y_pos = legend_y + idx * 20
        #     cv2.rectangle(img, (legend_x, y_pos), (legend_x + 15, y_pos + 15), color, -1)
        #     cv2.putText(img, name, (legend_x + 20, y_pos + 12),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        #
        # # 获取分类概率
        # for result in results:
        #     boxes = result.boxes  # 所有类别的概率
        #     meanBoxesConf = boxes.conf.mean()
        #     print(f"{meanBoxesConf=}")
        #     # top1_prob = probs.top1  # 最高概率值
        #     # top1_class = probs.top1_indices  # 最高概率对应的类别ID
        #     # class_name = model.names[top1_class.item()]  # 类别名称
        #     # print(f"类别: {class_name}, 概率: {top1_prob:.4f}")
        #     # print("所有类别概率:", probs.data.tolist())  # 输出所有类别的概率列表
        #     # avg_prob = probs.mean().item()  # 转为 Python 浮点数
        #     # print(f"所有类别平均概率: {avg_prob:.4f}")
        #     # 打印时间信息
        #     speed = result.speed
        #     totalSpeed = speed.get('preprocess') + speed.get('inference') + speed.get('postprocess')
        #     # img = result.plot()  # 绘制带有检测框的图像
        #
        # # ✅ 计算识别率
        # total_class_count = len(self.class_names)
        # recognized_count = len(recognized_class_ids)
        # recognition_rate = (recognized_count / total_class_count) * 100 if total_class_count > 0 else 0
        # print(f"识别组件种类数量: {recognized_count} / 总组件种类: {total_class_count}")
        # print(f"识别率: {recognition_rate:.2f}%")
        #
        # pixmap = convert_ndarray_to_qpixmap(img)
        # self.deep_learning_img.setPixmap(pixmap)
        # self.deep_learning_img.setScaledContents(True)  # 使图片适应标签大小
        # # self.tableWidget_Compldent.setItem(0, 3, QTableWidgetItem(f"{recognition_rate:.0f}"))  # 设置第一行第三列的数据为roll
        # self.tableWidget_Compldent.setItem(0, 3, QTableWidgetItem(f"{recognized_count:.0f}"))  # 设置第一行第三列的数据为roll
        # self.tableWidget_Compldent.setItem(1, 3, QTableWidgetItem(f"{totalSpeed:.2f}"))  # 设置第二行第三列的数据为roll
        # # 获取当前时间，用于文件名
        # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        # # 获取源文件名（不包含路径）
        # file_name = f"{os.path.splitext(os.path.basename(self.source_img_filepath))[0]}_DeepLearning_{current_time}.jpg"
        #
        # # 保存图像到指定路径
        # output_image_path = os.path.join(self.output_path, file_name)
        # cv2.imwrite(output_image_path, img)

        #gpt修改后代码：
        # ⚠️ 此处已替换为模板匹配，不再加载YOLO模型
        # model = YOLO(self.model_filepath)
        # img = cv2.imread(self.source_img_filepath)
        # results = model.predict(
        #     source=self.source_img_filepath,
        #     save=False,
        #     show=False，conf = 0.25
        # )

        # ⚠️ 用模板匹配方法替代深度学习推理
        temimg, match_count, time = process_templated_images(
            self.source_img_filepath,
            self.recon_sample_dir,
            self.class_filepath,
            self.output_path,
            self.class_names,
            self.class_colors
        )

        # ⚠️ 不再进行YOLO结果绘制，保留原位置
        # recognized_class_ids = set()
        # for r in results:
        #     boxes = r.boxes.xyxy.cpu().numpy()
        #     cls_ids = r.boxes.cls.cpu().numpy()
        #     confs = r.boxes.conf.cpu().numpy()
        #     for box, cls_id, conf in zip(boxes, cls_ids, confs):
        #         x1, y1, x2, y2 = map(int, box[:4])
        #         color = self.class_colors[int(cls_id)]
        #         label = f"{self.class_names[int(cls_id)]} {conf:.2f}"
        #         cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=1)
        #         recognized_class_ids.add(int(cls_id))

        # ⚠️ 不绘制YOLO图例，保留原位置
        # legend_x, legend_y = 10, 10
        # for idx, (cls_id, name) in enumerate(self.class_names.items()):
        #     color = self.class_colors[int(cls_id)]
        #     y_pos = legend_y + idx * 20
        #     cv2.rectangle(img, (legend_x, y_pos), (legend_x + 15, y_pos + 15), color, -1)
        #     cv2.putText(img, name, (legend_x + 20, y_pos + 12),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # ⚠️ 不再获取YOLO推理时间，保留原位置
        # for result in results:
        #     boxes = result.boxes
        #     meanBoxesConf = boxes.conf.mean()
        #     print(f"{meanBoxesConf=}")
        #     speed = result.speed
        #     totalSpeed = speed.get('preprocess') + speed.get('inference') + speed.get('postprocess')

        # ⚠️ 改为模板匹配的组件数量和耗时
        recognized_count = match_count
        totalSpeed = time

        # ⚠️ 计算识别率（可选）
        total_class_count = len(self.class_names)
        recognition_rate = (recognized_count / total_class_count) * 100 if total_class_count > 0 else 0
        print(f"识别组件种类数量: {recognized_count} / 总组件种类: {total_class_count}")
        print(f"识别率: {recognition_rate:.2f}%")

        # ✅ 显示图像（用模板匹配结果）
        pixmap = convert_ndarray_to_qpixmap(temimg)
        self.deep_learning_img.setPixmap(pixmap)
        self.deep_learning_img.setScaledContents(True)

        # ✅ 更新表格
        self.tableWidget_Compldent.setItem(0, 3, QTableWidgetItem(f"{recognized_count:.0f}"))  # 组件数量
        self.tableWidget_Compldent.setItem(1, 3, QTableWidgetItem(f"{totalSpeed:.0f}"))  # 耗时

        # ✅ 保存图像
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{os.path.splitext(os.path.basename(self.source_img_filepath))[0]}_DeepLearning_{current_time}.jpg"
        output_image_path = os.path.join(self.output_path, file_name)
        cv2.imwrite(output_image_path, temimg)
    def on_traditional_Compldent_ORB_generate(self):
        expected_img_extensions = ['.png', '.xpm', '.jpg', '.jpeg', '.bmp']
        flag = validate_dir(self.recon_sample_dir)
        if flag == False:
            QMessageBox.warning(self, '警告', '模板图片路径错误，请重设！')
            return
        flag = validate_file(self.source_img_filepath, expected_img_extensions)
        if flag == False:
            QMessageBox.warning(self, '警告', '待识别源图片路径错误，请重设！')
            return
        if not os.path.isdir(self.output_path):
            reply = QMessageBox.question(self, '警告', '输出文件夹路径错误，是否直接使用当前路径下的OUTPUT文件夹？',
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                # 使用当前路径下的OUTPUT文件夹作为输出路径
                self.output_path = r".\output"
                # 如果output文件夹不存在，则创建它
                if not os.path.isdir(self.output_path):
                    os.makedirs(self.output_path)
                QMessageBox.information(self, '提示', f'已使用当前路径下的output文件夹：{self.output_path}')
            else:
                # 用户选择不重设，可能需要其他操作
                QMessageBox.warning(self, '警告', '请重新设置输出文件夹路径！')
            return
        # orbimg, orb_ratio, time = process_orb_images(self.source_img_filepath, self.sample_dir,
        #                                                  self.class_filepath, self.output_path, self.class_names, self.class_colors)
        orbimg, count, time = process_orb_images(self.source_img_filepath, self.recon_sample_dir,
                                                         self.class_filepath, self.output_path, self.class_names, self.class_colors)
        pixmap = convert_ndarray_to_qpixmap(orbimg)
        self.traditional_Compldent_img.setPixmap(pixmap)
        self.traditional_Compldent_img.setScaledContents(True)  # 使图片适应标签大小
        self.tableWidget_Compldent.setItem(0, 1, QTableWidgetItem(f"{count:.0f}"))  # 设置第二行第二列的数据为roll
        self.tableWidget_Compldent.setItem(1, 1, QTableWidgetItem(f"{time:.0f}"))  # 设置第二行第二列的数据为roll
        # 获取当前时间，用于文件名
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 获取源文件名（不包含路径）
        file_name = f"{os.path.splitext(os.path.basename(self.source_img_filepath))[0]}_ORB_{current_time}.jpg"

        # 保存图像到指定路径
        output_image_path = os.path.join(self.output_path, file_name)
        cv2.imwrite(output_image_path, orbimg)
    def on_traditional_Compldent_SIFT_generate(self):
        expected_img_extensions = ['.png', '.xpm', '.jpg', '.jpeg', '.bmp']
        flag = validate_dir(self.recon_sample_dir)
        if flag == False:
            QMessageBox.warning(self, '警告', '模板图片路径错误，请重设！')
            return
        flag = validate_file(self.source_img_filepath, expected_img_extensions)
        if flag == False:
            QMessageBox.warning(self, '警告', '待识别源图片路径错误，请重设！')
            return
        if not os.path.isdir(self.output_path):
            reply = QMessageBox.question(self, '警告', '输出文件夹路径错误，是否直接使用当前路径下的OUTPUT文件夹？',
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                # 使用当前路径下的OUTPUT文件夹作为输出路径
                self.output_path = r".\output"
                # 如果output文件夹不存在，则创建它
                if not os.path.isdir(self.output_path):
                    os.makedirs(self.output_path)
                QMessageBox.information(self, '提示', f'已使用当前路径下的output文件夹：{self.output_path}')
            else:
                # 用户选择不重设，可能需要其他操作
                QMessageBox.warning(self, '警告', '请重新设置输出文件夹路径！')
            return
        # siftimg, sift_ratio, time = process_sift_images(self.source_img_filepath, self.sample_dir,
        #                                                     self.class_filepath, self.output_path,self.class_names, self.class_colors)
        siftimg, count, time = process_sift_images(self.source_img_filepath, self.recon_sample_dir,
                                                            self.class_filepath, self.output_path,self.class_names, self.class_colors)
        pixmap = convert_ndarray_to_qpixmap(siftimg)
        self.traditional_Compldent_img.setPixmap(pixmap)
        self.traditional_Compldent_img.setScaledContents(True)  # 使图片适应标签大小
        self.tableWidget_Compldent.setItem(0, 0, QTableWidgetItem(f"{count:.0f}"))  
        self.tableWidget_Compldent.setItem(1, 0, QTableWidgetItem(f"{time:.0f}"))  
        # 获取当前时间，用于文件名
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 获取源文件名（不包含路径）
        file_name = f"{os.path.splitext(os.path.basename(self.source_img_filepath))[0]}_SIFt_{current_time}.jpg"

        # 保存图像到指定路径
        output_image_path = os.path.join(self.output_path, file_name)
        cv2.imwrite(output_image_path, siftimg)

    def on_traditional_Compldent_Templated_generate(self):
        expected_img_extensions = ['.png', '.xpm', '.jpg', '.jpeg', '.bmp']
        flag = validate_dir(self.recon_sample_dir)
        if flag == False:
            QMessageBox.warning(self, '警告', '模板图片路径错误，请重设！')
            return
        expected_model_extensions = ['.pt']
        # validate_file(self.sample_img_path,expected_img_extensions)
        flag = validate_file(self.source_img_filepath, expected_img_extensions)
        if flag == False:
            QMessageBox.warning(self, '警告', '请选择源图片！')
            return
        flag = validate_file(self.model_filepath, expected_model_extensions)
        if flag == False:
            QMessageBox.warning(self, '警告', '请选择YOLO模型！')
            return
        output_dir = self.output_path
        if not os.path.isdir(self.output_path):
            reply = QMessageBox.question(self, '警告', '输出文件夹路径错误，是否直接使用当前路径下的OUTPUT文件夹？',
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                # 使用当前路径下的OUTPUT文件夹作为输出路径
                self.output_path = r".\output"
                # 如果output文件夹不存在，则创建它
                if not os.path.isdir(self.output_path):
                    os.makedirs(self.output_path)
                QMessageBox.information(self, '提示', f'已使用当前路径下的output文件夹：{self.output_path}')
            else:
                # 用户选择不重设，可能需要其他操作
                QMessageBox.warning(self, '警告', '请重新设置输出文件夹路径！')
            return
        # temimg, tem_ratio, time = process_templated_images(self.source_img_filepath, self.sample_dir,
        #                                                  self.class_filepath, self.output_path, self.class_names, self.class_colors)
        temimg, match_count, time = process_templated_images(self.source_img_filepath, self.recon_sample_dir,
                                                         self.class_filepath, self.output_path, self.class_names, self.class_colors)
        pixmap = convert_ndarray_to_qpixmap(temimg)
        self.deep_learning_img.setPixmap(pixmap)
        self.deep_learning_img.setScaledContents(True)  # 使图片适应标签大小
        self.tableWidget_Compldent.setItem(0, 2, QTableWidgetItem(f"{match_count:.0f}"))  # 设置第二行第二列的数据为roll
        self.tableWidget_Compldent.setItem(1, 2, QTableWidgetItem(f"{time:.0f}"))  # 设置第二行第二列的数据为roll
        # 获取当前时间，用于文件名
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 获取源文件名（不包含路径）
        file_name = f"{os.path.splitext(os.path.basename(self.source_img_filepath))[0]}_TEMPLATE_{current_time}.jpg"

        # 保存图像到指定路径
        output_image_path = os.path.join(self.output_path, file_name)
        cv2.imwrite(output_image_path, temimg)

    def on_traditional_Pose_generate(self):
        expected_img_extensions = ['.png', '.xpm', '.jpg', '.jpeg', '.bmp']
        flag = validate_file(self.sample_img_filepath, expected_img_extensions)
        if flag == False:
            QMessageBox.warning(self, '警告', '模板图片路径错误，请重设！')
            return
        # 修改：验证模板图片目录
        # flag = validate_dir(self.pose_sample_dir)
        # if flag == False:
        #     QMessageBox.warning(self, '警告', '模板图片文件夹路径错误，请重设！')
        #     return
        # 验证源待识别图像
        flag = validate_file(self.source_img_filepath, expected_img_extensions)
        if flag == False:
            QMessageBox.warning(self, '警告', '待识别源图片路径错误，请重设！')
            return
        # 检查输出路径
        if not os.path.isdir(self.output_path):
            reply = QMessageBox.question(self, '警告', '输出文件夹路径错误，是否直接使用当前路径下的OUTPUT文件夹？',
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                # 使用当前路径下的OUTPUT文件夹作为输出路径
                self.output_path = r".\output"
                # 如果output文件夹不存在，则创建它
                if not os.path.isdir(self.output_path):
                    os.makedirs(self.output_path)
                QMessageBox.information(self, '提示', f'已使用当前路径下的output文件夹：{self.output_path}')
            else:
                # 用户选择不重设，可能需要其他操作
                QMessageBox.warning(self, '警告', '请重新设置输出文件夹路径！')
            return
        
        try:
            # cv2img, anger, angle_err, trans_err = template_pose_estimation(self.pose_sample_dir, self.source_img_filepath)
            cv2img, pitch, yaw, roll= template_pose_estimation(self.pose_sample_dir, self.source_img_filepath)
        except ValueError as e:
            QMessageBox.warning(self, '匹配失败', str(e))  # 弹窗显示错误信息
            return
        self.tableWidget_pose.setItem(0, 0, QTableWidgetItem(f"{pitch:.3f}"))  # 设置第三行第二列的数据为pitch
        self.tableWidget_pose.setItem(0, 1, QTableWidgetItem(f"{yaw:.3f}"))  # 设置第三行第三列的数据为yaw
        self.tableWidget_pose.setItem(0, 2, QTableWidgetItem(f"{roll:.3f}"))  # 设置第三行第四列的数据为roll
        # self.tableWidget_pose.setItem(0, 3, QTableWidgetItem(f"{trans_err:.3f}m"))  # 设置第三行第四列的数据为trans_err
        # self.tableWidget_pose.setItem(0, 4, QTableWidgetItem(f"{angle_err:.3f}°"))  # 设置第三行第四列的数据为angle_err
        pixmap = convert_ndarray_to_qpixmap(cv2img)
        self.traditional_Pose_img.setPixmap(pixmap)
        self.traditional_Pose_img.setScaledContents(True)  # 使图片适应标签大小
        # 获取当前时间，用于文件名
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 获取源文件名（不包含路径）
        file_name = f"{os.path.splitext(os.path.basename(self.source_img_filepath))[0]}_traditional_pose_{current_time}.jpg"

        # 保存图像到指定路径
        output_image_path = os.path.join(self.output_path, file_name)
        cv2.imwrite(output_image_path, cv2img)

    def on_deep_Learning_Pose_generate(self):
        expected_h5model_extensions = ['.h5']
        expected_img_extensions = ['.png', '.xpm', '.jpg', '.jpeg', '.bmp']
        flag = validate_file(self.h5model_filepath, expected_h5model_extensions)
        if flag == False:
            QMessageBox.warning(self, '警告', 'h5模型路径错误，请重设！')
            return
        flag = validate_file(self.source_img_filepath, expected_img_extensions)
        if flag == False:
            QMessageBox.warning(self, '警告', '源图片路径错误，请重设！')
            return
        if not os.path.isdir(self.output_path):
            reply = QMessageBox.question(self, '警告', '输出文件夹路径错误，是否直接使用当前路径下的OUTPUT文件夹？',
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                # 使用当前路径下的OUTPUT文件夹作为输出路径
                self.output_path = r".\output"
                # 如果output文件夹不存在，则创建它
                if not os.path.isdir(self.output_path):
                    os.makedirs(self.output_path)
                QMessageBox.information(self, '提示', f'已使用当前路径下的output文件夹：{self.output_path}')
            else:
                # 用户选择不重设，可能需要其他操作
                QMessageBox.warning(self, '警告', '请重新设置输出文件夹路径！')
            return

        parser = argparse.ArgumentParser()
        parser.add_argument("command", metavar="<command>", nargs="?", default="test",
                            help="'train', 'evaluate', or 'test' (default: 'test')")
        parser.add_argument('--backbone', required=False, default='resnet50', help='Backbone architecture')
        parser.add_argument('--dataset', required=False, default="custom",
                            help='Dataset name (soyuz_hard,dragon_hard,default: soyuz_hard)')
        parser.add_argument('--epochs', required=False, default=50, type=int, help='Number of epochs')
        parser.add_argument('--image_scale', required=False, default=0.5, type=float, help='Resize scale')
        parser.add_argument('--ori_weight', required=False, default=1.0, type=float, help='Loss weight')
        parser.add_argument('--loc_weight', required=False, default=1.0, type=float, help='Loss weight')
        parser.add_argument('--bottleneck', required=False, default=128, type=int, help='Bottleneck width')
        parser.add_argument('--branch_size', required=False, default=1024, type=int, help='Branch input size')
        parser.add_argument('--learn_rate', required=False, default=0.0005, type=float, help='Learning rate')
        parser.add_argument('--batch_size', required=False, default=1, type=int, help='Number of images per GPU')
        parser.add_argument('--rot_aug', dest='rot_aug', action='store_true')
        parser.set_defaults(rot_aug=True)
        parser.add_argument('--rot_image_aug', dest='rot_image_aug', action='store_true')
        parser.set_defaults(rot_image_aug=True)
        parser.add_argument('--classify_ori', dest='regress_ori', action='store_false')
        parser.add_argument('--regress_ori', dest='regress_ori', action='store_true')
        parser.set_defaults(regress_ori=False)
        parser.add_argument('--classify_loc', dest='regress_loc', action='store_false')
        parser.add_argument('--regress_loc', dest='regress_loc', action='store_true')
        parser.set_defaults(regress_loc=True)
        parser.add_argument('--regress_keypoints', dest='regress_keypoints',
                            action='store_true')  # Experimental: Overrides options above
        parser.set_defaults(regress_keypoints=False)
        parser.add_argument('--sim2real', dest='sim2real', action='store_true')
        parser.set_defaults(sim2real=False)
        parser.add_argument('--clr', dest='clr', action='store_true')
        parser.set_defaults(clr=False)
        parser.add_argument('--f16', dest='f16', action='store_true')
        parser.set_defaults(f16=False)
        parser.add_argument('--square_image', dest='square_image', action='store_true')
        parser.set_defaults(square_image=True)
        parser.add_argument('--ori_param', required=False, default='quaternion',
                            help="'quaternion' 'euler_angles' 'angle_axis'")
        parser.add_argument('--ori_resolution', required=False, default=24, type=int,
                            help="Number of bins assigned to each angle")
        parser.add_argument('--weights', required=False, default="soyuz_hard",
                            help="Path to weights .h5 file or 'coco' or 'imagenet' for coco pre-trained weights (default: dragon_hard)")
        parser.add_argument('--logs', required=False, default=DEFAULT_LOGS_DIR,
                            help='Logs and checkpoints directory (default=logs/)')
        parser.add_argument('--image', required=False, metavar="path or URL to image", help='Image to evaluate')
        parser.add_argument('--video', required=False, metavar="path or URL to video", help='Video to evaluate')

        args = parser.parse_args()
        config = Config()
        config.ORIENTATION_PARAM = args.ori_param  # only used in regression mode
        config.ORI_BINS_PER_DIM = args.ori_resolution  # only used in classification mode
        config.NAME = args.dataset
        config.EPOCHS = args.epochs
        config.NR_DENSE_LAYERS = 1  # Number of fully connected layers used on top of the feature network
        config.LEARNING_RATE = args.learn_rate  # 0.001
        config.BOTTLENECK_WIDTH = args.bottleneck
        config.BRANCH_SIZE = args.branch_size
        config.BACKBONE = args.backbone
        config.ROT_AUG = args.rot_aug
        config.F16 = args.f16
        config.SIM2REAL_AUG = args.sim2real
        config.CLR = args.clr
        config.ROT_IMAGE_AUG = args.rot_image_aug
        config.OPTIMIZER = "SGD"
        config.REGRESS_ORI = args.regress_ori
        config.REGRESS_LOC = args.regress_loc
        config.REGRESS_KEYPOINTS = args.regress_keypoints
        config.LOSS_WEIGHTS['loc_loss'] = args.loc_weight
        config.LOSS_WEIGHTS['ori_loss'] = args.ori_weight

        # Set up resizing & padding if needed
        if args.square_image:
            config.IMAGE_RESIZE_MODE = 'square'
        else:
            config.IMAGE_RESIZE_MODE = 'pad64'

        # === 4️⃣ 设置原始图像尺寸 ===
        if args.dataset == "custom":
            width_original = custom_dataset.Camera.width
            height_original = custom_dataset.Camera.height
        else:
            width_original = urso.Camera.width
            height_original = urso.Camera.height

        config.IMAGE_MAX_DIM = round(width_original * args.image_scale)
        print("config.IMAGE_MAX_DIM:", config.IMAGE_MAX_DIM)    
        if config.IMAGE_MAX_DIM % 64 > 0:
            raise Exception("Scale problem. Image maximum dimension must be divisible by 64.")

        # n.b: assumes height is less than width
        height_scaled = round(height_original * args.image_scale)
        if height_scaled % 64 > 0:
            config.IMAGE_MIN_DIM = height_scaled - height_scaled % 64 + 64
        else:
            config.IMAGE_MIN_DIM = height_scaled
        config.IMAGE_MIN_DIM = 480
        print("config.IMAGE_MIN_DIM:", config.IMAGE_MIN_DIM) 
        # n.b: assumes height is less than width
        config.IMAGES_PER_GPU = 1
        config.BATCH_SIZE = config.IMAGES_PER_GPU * config.GPU_COUNT
        config.update()
        config.display()  # ✅ 添加显示参数，确保一致

        model = net.UrsoNet(mode="inference", config=config, model_dir=args.logs)
        # 根据权重参数选择要加载的权重文件
        # 加载权重
        model.load_weights(self.h5model_filepath, None, by_name=True)

        dataset_dir = os.path.join(DATA_DIR, args.dataset)

        # Load validation dataset
        # dataset = urso.Urso()
        dataset = custom_dataset.CustomDataset()
        dataset.load_dataset(dataset_dir, config, "test")

        try:
            # cv2img, roll, pitch, yaw, angle_err, trans_err = deep_pose_estimation(self.pose_sample_dir, self.source_img_filepath)
            cv2img, pitch, yaw, roll= deep_pose_estimation(self.source_img_filepath,model,dataset)
        except ValueError as e:
            QMessageBox.warning(self, '检测失败', str(e))  # 弹窗显示错误信息
            return
        self.tableWidget_pose.setItem(1, 0, QTableWidgetItem(f"{pitch:.6f}"))  # 设置第二行第二列的数据为pitch
        self.tableWidget_pose.setItem(1, 1, QTableWidgetItem(f"{yaw:.6f}"))  # 设置第二行第三列的数据为yaw
        self.tableWidget_pose.setItem(1, 2, QTableWidgetItem(f"{roll:.6f}"))  # 设置第二行第四列的数据为roll
        # self.tableWidget_pose.setItem(1, 3, QTableWidgetItem(f"{trans_err:.3f}m"))  #
        # self.tableWidget_pose.setItem(1, 4, QTableWidgetItem(f"{angle_err:.3f}°"))  #
        pixmap = convert_ndarray_to_qpixmap(cv2img)
        self.deep_learning_pose_img.setPixmap(pixmap)
        self.deep_learning_pose_img.setScaledContents(True)  # 使图片适应标签大小
        # 获取当前时间，用于文件名
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 获取源文件名（不包含路径）
        file_name = f"{os.path.splitext(os.path.basename(self.source_img_filepath))[0]}_deep_learning_pose_{current_time}.jpg"

        # 保存图像到指定路径
        output_image_path = os.path.join(self.output_path, file_name)
        cv2.imwrite(output_image_path, cv2img)

    def show_image_size(self):
        sender: ClickableLabel = self.sender()  # 获取发送信号的 QLabel 对象
        if not sender.origin_pixmap:
            return

        # 将 QPixmap 转换为 QImage
        pixmap = sender.origin_pixmap
        image = pixmap.toImage()
        width = image.width()
        height = image.height()

        # 将 QImage 转换为 RGB 格式的 NumPy 数组
        image = image.convertToFormat(QImage.Format.Format_RGB32)  # 确保格式为RGB32
        ptr = image.bits()
        ptr.setsize(image.size().width() * image.size().height() * 4)  # 每像素4字节（RGBA）

        # 转换为 NumPy 数组并进行颜色通道转换
        arr = np.array(ptr).reshape(height, width, 4)  # (H, W, 4) 包含透明度通道
        arr = arr[:, :, :3]  # 去除 Alpha 通道，只保留 RGB 通道
        arr = arr[:, :, ::-1]  # 将 BGR 转换为 RGB 顺序

        # 显示图像及尺寸
        plt.figure(figsize=(8, 6))
        plt.imshow(arr)  # 使用 RGB 格式显示图像
        plt.title(f"Image Size: {width} x {height}")  # 设置标题
        plt.axis('off')  # 关闭坐标轴
        plt.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec())
