import time

import cv2
import numpy as np
import os
from pathlib import Path  # 导入Path模块


def read_classes(classes_file_path):
    with open(classes_file_path, 'r') as file:
        classes = file.read().splitlines()
    return classes


def read_label_file(label_file_path):
    with open(label_file_path, 'r') as file:
        lines = file.read().splitlines()
    labels = [list(map(float, line.split())) for line in lines]
    return labels


def orb_feature_matching(input_image, template_image):
    start_time = time.time()  # 开始时间
    # 创建 ORB 特征检测器
    # ORB参数优化
    orb = cv2.ORB_create(
        nfeatures=2000,
        scaleFactor=1.1,
        nlevels=12,
        edgeThreshold=15,
        patchSize=31,
        fastThreshold=10,
        WTA_K=2
    )

    # 检测关键点和描述符
    keypoints1, descriptors1 = orb.detectAndCompute(template_image, None)  # 模板图在左侧
    keypoints2, descriptors2 = orb.detectAndCompute(input_image, None)    # 输入图在右侧

    # 使用暴力匹配器（BFMatcher）进行匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches =  bf.knnMatch(descriptors1, descriptors2,k=2)

    # 筛选优质匹配（Lowe's ratio test）
    good_matches = []
    for m,n in matches:
        if m.distance < 0.98 * n.distance:  # 可调整的阈值
            good_matches.append(m)
    # 按照匹配距离排序（保持你的可视化逻辑）
    good_matches = sorted(good_matches, key=lambda x: x.distance)

    # 绘制连线和匹配点
    img_matches = None
    if len(matches) > 10:
        # 拼接图像：模板图在左，输入图在右
        # 最多显示500个优质匹配
        display_matches = good_matches[:500] if len(good_matches) > 500 else good_matches
        img_matches = cv2.drawMatches(
            template_image, keypoints1,
            input_image, keypoints2,
            display_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

    # 计算匹配率：有效匹配数占总关键点的比例
    # total_keypoints = min(len(keypoints1), len(keypoints2))
    match_ratio = len(good_matches) / len(descriptors1) if len(descriptors1) > 0 else 0

    # 在图像上绘制匹配率文本
    if img_matches is not None:
        img_matches = cv2.putText(img_matches, f"ORB Match: {match_ratio * 100:.1f}%",
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    elapsed_time = time.time() - start_time  # 运行时间
    print(f"ORB 最高匹配率: {match_ratio * 100:.2f}% | 耗时: {elapsed_time:.3f} 秒")
    return img_matches, match_ratio,elapsed_time  # 返回图像和匹配率


def template_matching(input_image, template_image, threshold=0.7):
    start_time = time.time()  # 开始时间
    # 在输入图像上进行模板匹配
    result = cv2.matchTemplate(input_image, template_image, cv2.TM_CCOEFF_NORMED)
    
    # 获取最小值、最大值及其位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # 如果最大匹配值大于阈值，则认为匹配成功
    # if max_val > threshold:
    #     # 获取匹配区域的左上角坐标和模板图的宽高
    #     top_left = max_loc
    #     w, h = template_image.shape[::-1]
        
    #     # 画出匹配区域的矩形框
    #     bottom_right = (top_left[0] + w, top_left[1] + h)
    #     matched_image = input_image.copy()  # 保持原始输入图像不变
    #     cv2.rectangle(matched_image, top_left, bottom_right, (0, 255, 0), 2)
        
    #     # 显示结果
    #     cv2.imshow("Template Matching", matched_image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
        
    #     # 返回绘制了匹配框的图像
    #     return matched_image
    top_left = max_loc
    w, h = template_image.shape[::-1]
    
    # 画出匹配区域的矩形框
    bottom_right = (top_left[0] + w, top_left[1] + h)
    matched_image = input_image.copy()  # 保持原始输入图像不变
    cv2.rectangle(matched_image, top_left, bottom_right, (0, 255, 0), 2)
    # 绘制匹配率（置信度）
    cv2.putText(matched_image, f"Confidence: {max_val * 100:.1f}%",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    elapsed_time = time.time() - start_time  # 运行时间
    print(f"模板匹配最高置信度: {max_val * 100:.2f}% | 耗时: {elapsed_time:.3f} 秒")
    return matched_image, max_val  # 返回图像和置信度


def sift_feature_matching(input_image, template_image):
    # 使用cv2.SIFT_create()代替cv2.xfeatures2d_SIFT.create()
    detector = cv2.SIFT_create()

    # 计算关键点和描述符
    keypoints1, descriptors1 = detector.detectAndCompute(input_image, None)
    keypoints2, descriptors2 = detector.detectAndCompute(template_image, None)

    # FLANN参数设置
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(check=50)  # 或者传入字典

    # 使用FLANN进行匹配
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # knnMatch返回k个最近邻的匹配
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # 创建一个mask来筛选优秀的匹配
    matchesMask = [[0, 0] for i in range(len(matches))]  # Python 2.x 使用xrange()

    # 使用Lowe论文中的比率测试来筛选匹配
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.5 * n.distance:
            matchesMask[i] = [1, 0]

    # 绘制匹配点
    draw_params = dict(matchColor=(0, 0, 255), singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask, flags=0)

    # 绘制匹配结果
    img_matches = cv2.drawMatchesKnn(input_image, keypoints1, template_image, keypoints2, matches, None, **draw_params)

    # 显示匹配结果
    return img_matches

def sift_flann_matches(input_image, template_image):
    start_time = time.time()  # 开始时间
    if len(template_image.shape) == 2:  # 检查是否为灰度图
        template_image = cv2.cvtColor(template_image, cv2.COLOR_GRAY2BGR)

    if len(input_image.shape) == 2:  # 同样检查输入图像
        input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
    # 1. 初始化 SIFT 检测器
    sift = cv2.SIFT_create(nfeatures=500, contrastThreshold=0.01, edgeThreshold=10)

    # 2. 检测关键点和描述符
    keypoints1, descriptors1 = sift.detectAndCompute(template_image, None)
    keypoints2, descriptors2 = sift.detectAndCompute(input_image, None)

    # 3. 使用 FLANN 匹配器
    index_params = dict(algorithm=1, trees=5)  # KD-Tree 算法
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 4. 匹配关键点
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # 5. 筛选优质匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.95 * n.distance:
            good_matches.append(m)

    # 6. 绘制随机颜色的连线
    h1, w1 = template_image.shape[:2]
    h2, w2 = input_image.shape[:2]

    # 创建一个大图用于拼接两张图片
    match_image = np.zeros((max(h1, h2), w1 + w2, 3), dtype='uint8')
    match_image[:h1, :w1] = template_image
    match_image[:h2, w1:w1 + w2] = input_image

    # 遍历匹配点并绘制随机颜色线
    for match in good_matches:
        # 获取关键点位置
        pt1 = tuple(map(int, keypoints1[match.queryIdx].pt))
        pt2 = tuple(map(int, keypoints2[match.trainIdx].pt))
        pt2 = (pt2[0] + w1, pt2[1])  # 偏移x坐标

        # 生成随机颜色
        color = tuple(np.random.randint(0, 255, 3).tolist())

        # 绘制连线
        cv2.line(match_image, pt1, pt2, color, 1, cv2.LINE_AA)
        cv2.circle(match_image, pt1, 5, color, -1)
        cv2.circle(match_image, pt2, 5, color, -1)

    # 计算匹配率：优质匹配占原始匹配的比例
    match_ratio = len(good_matches) / len(matches) if len(matches) > 0 else 0

    # 在图像上绘制匹配率
    cv2.putText(match_image, f"SIFT Match: {match_ratio * 100:.1f}%",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    elapsed_time = time.time() - start_time  # 运行时间
    print(f"SIFT 最高匹配率: {match_ratio * 100:.2f}% | 耗时: {elapsed_time:.3f} 秒")
    return match_image, match_ratio ,elapsed_time # 返回图像和匹配率


def transform_labels(labels, homography_matrix, template_size, input_image_size):
    template_h, template_w = template_size
    input_h, input_w = input_image_size
    transformed_labels = []

    for label in labels:
        class_id, x_center, y_center, box_width, box_height = label

        x_center_px = x_center * template_w
        y_center_px = y_center * template_h
        point = np.array([[x_center_px, y_center_px, 1]]).T
        transformed_point = np.dot(homography_matrix, point)

        x_center_trans = transformed_point[0] / transformed_point[2]
        y_center_trans = transformed_point[1] / transformed_point[2]

        box_width_px = box_width * template_w
        box_height_px = box_height * template_h
        box_width_px *= 1.2
        box_height_px *= 1.2

        x_center_norm = x_center_trans / input_w
        y_center_norm = y_center_trans / input_w
        box_width_norm = box_width_px / input_w
        box_height_norm = box_height_px / input_w

        transformed_labels.append([class_id, x_center_norm, y_center_norm, box_width_norm, box_height_norm])

    return transformed_labels


def visualize_annotations(input_image, label_file_path, classes_file_path, output_image_path, homography_matrix,
                          template_size, method_name):
    classes = read_classes(classes_file_path)
    height, width = input_image.shape[:2]
    # 将 input_image 切割成左右两部分
    left_image = input_image[:, :width // 2]  # 左半部分
    right_image = input_image[:, width // 2:]  # 右半部分
    height, width = right_image.shape[:2]
    labels = read_label_file(label_file_path)
    print(f"{labels=}")

    adjusted_labels = transform_labels(labels, homography_matrix, template_size, (height, width))
    print(f"{adjusted_labels=}")
    for label in adjusted_labels:
        class_id, x_center, y_center, box_width, box_height = label
        print(f"0 --- {class_id=} {x_center=} {y_center=} {box_width=} {box_height=}")
        # 将标签坐标转换为像素坐标
        x_center_px, y_center_px = int(x_center * width), int(y_center * height)
        box_width_px, box_height_px = int(box_width * width), int(box_height * height)
        print(f"1 --- {x_center_px=} {y_center_px=} {box_width_px=} {box_height_px=}")
        # 计算矩形的四个角的坐标
        x1, y1 = max(0, x_center_px - box_width_px // 2), max(0, y_center_px - box_height_px // 2)
        x2, y2 = min(width, x_center_px + box_width_px // 2), min(height, y_center_px + box_height_px // 2)
        print(f"2 --- {x1=} {y1=} {x2=} {y2=}")
        # 在输入图像上绘制矩形框
        cv2.rectangle(right_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 在框上方显示类别名称
        class_name = classes[int(class_id)]
        cv2.putText(right_image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    final_image = np.hstack((left_image, right_image))
    return final_image, method_name

# 批量处理：输入目录 + 模板目录，批量处理所有图片
def process_images(input_dir, output_dir, templates_dir, classes_file_path):
    os.makedirs(output_dir, exist_ok=True)

    for input_image_name in os.listdir(input_dir):
        input_image_path = os.path.join(input_dir, input_image_name)
        if input_image_path.lower().endswith(('.jpg', '.png', '.jpeg')):
            output_image_path = os.path.join(output_dir, input_image_name)

            input_image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

            best_orb_template_path = None
            best_orb_homography_matrix = None

            best_template_matching_template_path = None
            best_template_matching_result = None

            best_sift_template_path = None
            best_sift_homography_matrix = None

            for template_name in os.listdir(templates_dir):
                template_path = os.path.join(templates_dir, template_name)
                template_image = cv2.imread(template_path, cv2.IMREAD_COLOR)

                if template_image is None:
                    print(f"Warning: Template image '{template_path}' could not be loaded. Skipping...")
                    continue

                template_image = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

                orb_homography_matrix, orb_ratio = orb_feature_matching(input_image, template_image)
                if orb_homography_matrix is not None:
                    best_orb_template_path = template_path
                    best_orb_homography_matrix = orb_homography_matrix
                    current_best_orb = orb_ratio

                template_matching_result, template_conf = template_matching(input_image, template_image, threshold=0.6)
                if template_matching_result is not None:
                    best_template_matching_template_path = template_path
                    best_template_matching_result = template_matching_result
                    current_best_tmpl = template_conf

                sift_homography_matrix, sift_ratio = sift_feature_matching(input_image, template_image)
                if sift_homography_matrix is not None:
                    best_sift_template_path = template_path
                    best_sift_homography_matrix = sift_homography_matrix
                    current_best_sift = sift_ratio

            # 计算各算法对应输出图片的匹配度并输出到终端（合理量化匹配度使其满分100分）
            if best_orb_template_path:
                # 计算 ORB 匹配分数
                orb_score = calculate_orb_match_score(input_image_path, best_orb_template_path)
                # 将 ORB 分数进行归一化或缩放
                scaled_orb_score = scale_orb_score(orb_score)
                # 构造输出文件名
                orb_output_filename = input_image_path.split('/')[-1].replace('.jpg', '_ORB.jpg')
                # 打印 ORB 匹配结果
                print(f"{orb_output_filename} 对应的 ORB 匹配度(满分100): {scaled_orb_score:.2f}")

            if best_template_matching_template_path:
                # 计算模板匹配分数
                template_matching_score = calculate_template_matching_score(input_image_path, best_template_matching_template_path)
                # 将模板匹配分数进行归一化或缩放
                scaled_template_matching_score = scale_template_matching_score(template_matching_score)
                # 构造输出文件名
                template_matching_output_filename = input_image_path.split('/')[-1].replace('.jpg', '_Template_Matching.jpg')
                # 打印模板匹配结果
                print(f"{template_matching_output_filename} 对应的模板匹配度(满分100): {scaled_template_matching_score:.2f}")

            if best_sift_template_path:
                # 计算 SIFT 匹配分数
                sift_score = calculate_sift_match_score(input_image_path, best_sift_template_path)
                # 将 SIFT 分数进行归一化或缩放
                scaled_sift_score = scale_sift_score(sift_score)
                # 构造输出文件名
                sift_output_filename = input_image_path.split('/')[-1].replace('.jpg', '_SIFT.jpg')
                # 打印 SIFT 匹配结果
                print(f"{sift_output_filename} 对应的 SIFT 匹配度(满分100): {scaled_sift_score:.2f}")



def calculate_orb_match_score(input_image_path, template_image_path):
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    template_image = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
    keypoints1, descriptors1 = orb.detectAndCompute(input_image, None)
    keypoints2, descriptors2 = orb.detectAndCompute(template_image, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) > 10:
        match_score = sum([m.distance for m in matches[:10]]) # / len(matches)
        return match_score
    return 0


def calculate_template_matching_score(input_image_path, template_image_path):
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    template_image = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)
    result = cv2.matchTemplate(input_image, template_image, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return max_val


def calculate_sift_match_score(input_image_path, template_image_path):
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    template_image = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(input_image, None)
    keypoints2, descriptors2 = sift.detectAndCompute(template_image, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) > 10:
        match_score = sum([m.distance for m in matches]) / len(matches)
        return match_score
    return 0


def scale_orb_score(score):
    """
    根据ORB算法匹配度计算原理，对其得分进行量化使其满分100分
    这里简单假设一个合理的调整逻辑，例如取最小匹配距离为0，最大匹配距离总和为一个相对较大的值（比如1000，可根据实际情况调整）
    通过线性映射的方式将原始得分转换为0到100的范围
    """
    max_possible_score = 1000  # 假设的最大可能得分，可根据实际调整
    scaled_score = (max_possible_score - score) / max_possible_score * 70
    return min(100, max(0, scaled_score))  # 确保得分在0到100之间


def scale_template_matching_score(score):
    """
    模板匹配算法的得分本身就在0到1之间，直接乘以100转换为满分100分的得分
    """
    return score * 100


def scale_sift_score(score):
    """
    根据SIFT算法匹配度计算原理，对其得分进行量化使其满分100分
    类似ORB算法，假设一个合理的调整逻辑，例如取最小匹配距离为0，最大匹配距离总和为一个相对较大的值（比如50000，可根据实际情况调整）
    通过线性映射的方式将原始得分转换为0到100的范围
    """
    max_possible_score = 50000  # 假设的最大可能得分，可根据实际调整
    scaled_score = (max_possible_score - score) / max_possible_score * 70
    return min(100, max(0, scaled_score))  # 确保得分在0到100之间
