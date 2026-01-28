import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取两张图像
img1 = cv2.imread(r"image/1_muban.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(r"./image/2.jpg", cv2.IMREAD_GRAYSCALE)

if img1 is None:
    print("Error: Could not load image1")
if img2 is None:
    print("Error: Could not load image2")

# 2. 使用ORB提取特征点和描述符
orb = cv2.ORB_create()

# 提取关键点和描述符
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 3. 特征点匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# 4. 排序匹配结果
matches = sorted(matches, key = lambda x: x.distance)

# 5. 绘制匹配的特征点
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 6. 显示匹配结果
plt.figure(figsize=(10, 5))
plt.imshow(img_matches)
plt.show()