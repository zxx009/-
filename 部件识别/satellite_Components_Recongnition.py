import os
import cv2
import numpy as np
from SatelliteCompIdentClassic import *

input_file = r"./datasets/soyuz_hard/1_rgb.png"
templates_path = r"./datasets/soyuz_hard/2_rgb.png"
classes_file = r"./classes.txt"
output_path=r"./output"

# 单文件处理：输入单张图片 + 单张模板，返回匹配结果图和匹配率
def process_orb_images(input_file, template_path, classes_file, output_path):
    # 读取输入图片
    input_image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    if input_image is None:
        print(f"Error: Could not load input image from {input_file}")
        return
    # 在这里继续处理模板图片
    print(f"模板图片路径: {template_path}")
    print(f"处理图片路径: {input_file}")
    template_image = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    if template_image is None:
        print(f"Warning: Template image '{template_path}' could not be loaded. Skipping...")

    # ORB 匹配
    orb_img,orb_ratio,time = orb_feature_matching(input_image, template_image)

    # # 模板匹配
    # template_matching_img,template_conf = template_matching(input_image, template_image, threshold=0.6)

    return orb_img,orb_ratio,time

def process_sift_images(input_file, template_path, classes_file, output_path):
    # 读取输入图片
    input_image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    if input_image is None:
        print(f"Error: Could not load input image from {input_file}")
        return
    # 在这里继续处理模板图片
    print(f"模板图片路径: {template_path}")
    print(f"处理图片路径: {input_file}")
    template_image = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    if template_image is None:
        print(f"Warning: Template image '{template_path}' could not be loaded. Skipping...")

    # # 模板匹配
    # template_matching_img,template_conf = template_matching(input_image, template_image, threshold=0.6)

    # SIFT 匹配
    sift_img,match_ratio, time = sift_flann_matches(input_image, template_image)
    return sift_img,match_ratio, time

def draw_feature_matches(template_image, input_image, method_name):
    # 使用OpenCV的特征匹配函数
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(template_image, None)
    keypoints2, descriptors2 = sift.detectAndCompute(input_image, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 确保 input_image 是三通道（彩色图像），即使它是灰度图像
    # if len(input_image.shape) == 2:  # 如果是灰度图像
        # input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
    

    # 将模板图和输入图拼接在一起，左边为模板图，右边为输入图
    output_image = np.hstack((template_image, input_image))
    output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2RGB)

    preset_colors = [
        (255, 0, 0),    # 红色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 蓝色
        (0, 255, 255),  # 青色
        (255, 255, 0),  # 黄色
        (255, 0, 255),  # 品红
        (128, 128, 0),  # 橄榄色
        (0, 128, 128),  # 深青色
        (128, 0, 128)   # 紫色
    ]

    # 从预设颜色列表中随机选择一个颜色
    # 绘制匹配点
    # for match in matches[:10]:  # 绘制前10个匹配点
    for match in matches:  # 绘制前10个匹配点

        pt1 = (int(keypoints1[match.queryIdx].pt[0]), int(keypoints1[match.queryIdx].pt[1]))
        pt2 = (int(keypoints2[match.trainIdx].pt[0] + template_image.shape[1]), 
               int(keypoints2[match.trainIdx].pt[1]))

        # 在拼接图上绘制连线
        import random
        random_color = random.choice(preset_colors)
        cv2.line(output_image, pt1, pt2, random_color, 1)

    return output_image

# def draw_feature_matches(template_image, input_image, method_name):
#     # 使用OpenCV的特征匹配函数
#     sift = cv2.SIFT_create()
#     keypoints1, descriptors1 = sift.detectAndCompute(input_image, None)  # 输入图的关键点
#     keypoints2, descriptors2 = sift.detectAndCompute(template_image, None)  # 模板图的关键点

#     bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
#     matches = bf.match(descriptors1, descriptors2)
#     matches = sorted(matches, key=lambda x: x.distance)

#     # 确保 input_image 是三通道（彩色图像），即使它是灰度图像
#     # if len(input_image.shape) == 2:  # 如果是灰度图像
#     #     input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)

#     # 确保 template_image 也是三通道
#     # if len(template_image.shape) == 2:  # 如果是灰度图像
#     #     template_image = cv2.cvtColor(template_image, cv2.COLOR_GRAY2BGR)

#     # 将模板图和输入图拼接在一起，左边为输入图，右边为模板图
#     output_image = np.hstack((input_image, template_image))  # 左边是输入图，右边是模板图

#     # 绘制匹配点
#     for match in matches[:10]:  # 绘制前10个匹配点
#         pt1 = (int(keypoints1[match.queryIdx].pt[0]), int(keypoints1[match.queryIdx].pt[1]))
#         pt2 = (int(keypoints2[match.trainIdx].pt[0] + input_image.shape[1]),  # 修改此处
#                int(keypoints2[match.trainIdx].pt[1]))

#         # 在拼接图上绘制连线
#         cv2.line(output_image, pt1, pt2, (0, 255, 0), 1)

#     return output_image

if __name__ == "__main__":
  process_orb_images(input_file, templates_path, classes_file, output_path)