import cv2
import numpy as np
import os
import time


def read_classes(classes_file_path):
    print(f"Reading classes from: {classes_file_path}")
    try:
        with open(classes_file_path, 'r') as file:
            classes = file.read().splitlines()
        print(f"Successfully read {len(classes)} classes")
        return classes
    except Exception as e:
        print(f"Error reading classes file: {e}")
        return []


# def read_label_file(label_file_path):
#     print(f"Reading labels from: {label_file_path}")
#     try:
#         with open(label_file_path, 'r') as file:
#             lines = file.read().splitlines()
#         labels = [list(map(float, line.split())) for line in lines]
#         print(f"Successfully read {len(labels)} labels")
#         return labels
#     except Exception as e:
#         print(f"Error reading label file: {e}")
#         return []
def read_label_file(label_file_path):
    print(f"Reading labels from: {label_file_path}")
    try:
        with open(label_file_path, 'r') as file:
            lines = file.read().splitlines()
        labels = []
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            rest = list(map(float, parts[1:]))
            labels.append([class_id] + rest)
        print(f"Successfully read {len(labels)} labels")
        return labels
    except Exception as e:
        print(f"Error reading label file: {e}")
        return []

def sift_feature_matching(input_image, template_image):
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(input_image, None)
    keypoints2, descriptors2 = sift.detectAndCompute(template_image, None)

    if descriptors1 is None or descriptors2 is None:
        return None

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) > 20:
        print(f"SIFT found {len(matches)} matches")
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        homography_matrix, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        return homography_matrix
    print(f"SIFT found only {len(matches)} matches (need at least 10)")
    return None


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

        x_center_trans = float(transformed_point[0] / transformed_point[2])
        y_center_trans = float(transformed_point[1] / transformed_point[2])

        box_width_px = box_width * template_w
        box_height_px = box_height * template_h
        box_width_px *= 1.2
        box_height_px *= 1.2

        x_center_norm = x_center_trans / input_w
        y_center_norm = y_center_trans / input_h
        box_width_norm = box_width_px / input_w
        box_height_norm = box_height_px / input_h

        transformed_labels.append([class_id, x_center_norm, y_center_norm, box_width_norm, box_height_norm])

    return transformed_labels


# def visualize_annotations(input_image_path, label_file_path, classes_file_path, output_image_path, homography_matrix,
#                           template_size, method_name):
#     print(f"\nVisualizing annotations for {input_image_path} using {method_name}")
#
#     if not os.path.exists(input_image_path):
#         print(f"Error: Input image {input_image_path} does not exist")
#         return
#
#     classes = read_classes(classes_file_path)
#     if not classes:
#         print("Error: No classes found. Cannot proceed with visualization.")
#         return
#
#     input_image = cv2.imread(input_image_path)
#     if input_image is None:
#         print(f"Error: Could not read input image {input_image_path}")
#         return
#
#     height, width = input_image.shape[:2]
#     print(f"Input image size: {width}x{height}")
#
#     if not os.path.exists(label_file_path):
#         print(f"Error: Label file {label_file_path} does not exist")
#         return
#
#     labels = read_label_file(label_file_path)
#     if not labels:
#         print("Warning: No labels found. Continuing without annotations.")
#
#     adjusted_labels = transform_labels(labels, homography_matrix, template_size, (height, width))
#     print(f"Transformed {len(adjusted_labels)} labels")
#
#     for label in adjusted_labels:
#         class_id, x_center, y_center, box_width, box_height = label
#         x_center_px, y_center_px = int(float(x_center * width)), int(float(y_center * height))
#         box_width_px, box_height_px = int(float(box_width * width)), int(float(box_height * height))
#
#         x1, y1 = max(0, x_center_px - box_width_px // 2), max(0, y_center_px - box_height_px // 2)
#         x2, y2 = min(width, x_center_px + box_width_px // 2), min(height, y_center_px + box_height_px // 2)
#
#         cv2.rectangle(input_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         if 0 <= int(class_id) < len(classes):
#             class_name = classes[int(class_id)]
#             cv2.putText(input_image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#         else:
#             print(f"Warning: Class ID {int(class_id)} out of range (0-{len(classes) - 1})")
#     return input_image
def visualize_annotations(input_image_path, label_file_path, classes_file_path, output_image_path,
                          homography_matrix, template_size, method_name,
                          class_names, class_colors):  # class_names 应该是列表，class_colors 是字典
    print(f"\nVisualizing annotations for {input_image_path} using {method_name}")

    if not os.path.exists(input_image_path):
        print(f"Error: Input image {input_image_path} does not exist")
        return

    classes = read_classes(classes_file_path)
    if not classes:
        print("Error: No classes found. Cannot proceed with visualization.")
        return

    input_image = cv2.imread(input_image_path)
    if input_image is None:
        print(f"Error: Could not read input image {input_image_path}")
        return

    height, width = input_image.shape[:2]
    print(f"Input image size: {width}x{height}")

    if not os.path.exists(label_file_path):
        print(f"Error: Label file {label_file_path} does not exist")
        return

    labels = read_label_file(label_file_path)
    if not labels:
        print("Warning: No labels found. Skipping annotation and returning original image.")
        if output_image_path:
            cv2.imwrite(output_image_path, input_image)
        return input_image, 0

    adjusted_labels = transform_labels(labels, homography_matrix, template_size, (height, width))
    print(f"Transformed {len(adjusted_labels)} labels")
    if not adjusted_labels:
        print("No adjusted labels to draw.")
        if output_image_path:
            cv2.imwrite(output_image_path, input_image)
        return input_image, 0

    # # 返回识别率
    # total_components = len(classes)  # 总共的标准组件种类数量
    total_components = len(adjusted_labels)  # 识别到的标签标准组件种类数量
    matched_classes = set([label[0] for label in adjusted_labels])  # 获取所有被识别的类别ID（去重）
    unique_matched = len(matched_classes)

    recognition_score = (unique_matched / total_components) * 100

    print(f"识别组件数量: {unique_matched} / {total_components}")
    # Draw bounding boxes
    for label in adjusted_labels:
        class_id, x_center, y_center, box_width, box_height = label
        x_center_px, y_center_px = int(float(x_center * width)), int(float(y_center * height))
        box_width_px, box_height_px = int(float(box_width * width)), int(float(box_height * height))

        x1 = max(0, x_center_px - box_width_px // 2)
        y1 = max(0, y_center_px - box_height_px // 2)
        x2 = min(width, x_center_px + box_width_px // 2)
        y2 = min(height, y_center_px + box_height_px // 2)

        color = class_colors.get(int(class_id), (0, 255, 0))  # fallback color
        cv2.rectangle(input_image, (x1, y1), (x2, y2), color, 1)

    # Draw legend
    legend_x, legend_y = 10, 10
    for idx, (cls_id, name) in enumerate(class_names.items()):
        color = class_colors[int(cls_id)]
        y_pos = legend_y + idx * 20
        cv2.rectangle(input_image, (legend_x, y_pos), (legend_x + 15, y_pos + 15), color, -1)
        cv2.putText(input_image, name, (legend_x + 20, y_pos + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Optionally save to output path
    if output_image_path:
        cv2.imwrite(output_image_path, input_image)

    # return input_image,recognition_score
    return input_image,unique_matched
def calculate_sift_match_score(input_image_path, template_image_path):
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    template_image = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)

    if input_image is None or template_image is None:
        return 0

    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(input_image, None)
    keypoints2, descriptors2 = sift.detectAndCompute(template_image, None)

    if descriptors1 is None or descriptors2 is None:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) > 10:
        match_score = sum([m.distance for m in matches]) / len(matches)
        return match_score
    return 0


def scale_sift_score(score):
    max_possible_score = 1000
    scaled_score = (max_possible_score - score) / max_possible_score * 85
    return min(100, max(0, scaled_score))

def process_sift_images(input_image_path, templates_dir, classes_file_path, output_dir, class_names, class_colors):
    start_time = time.time()
    if not os.path.exists(input_image_path):
        print(f"Error: 输入图像 {input_image_path} 不存在")
        exit(1)

    input_image_color = cv2.imread(input_image_path)
    if input_image_color is None:
        print(f"Error: 无法读取输入图像 {input_image_path}")
        exit(1)

    input_image_gray = cv2.cvtColor(input_image_color, cv2.COLOR_BGR2GRAY)

    if not os.path.exists(templates_dir):
        print(f"Error: 模板目录 {templates_dir} 不存在")
        exit(1)

    supported_extensions = ('.jpg', '.png', '.jpeg', '.bmp', '.JPG', '.PNG', '.JPEG', '.BMP')
    template_files = [f for f in os.listdir(templates_dir) if f.lower().endswith(supported_extensions) and not f.startswith('.')]

    print(f"在模板目录中找到 {len(template_files)} 个文件")
    for file in template_files:
        print(f"  - {file}")

    best_sift_template_path = None
    best_sift_homography_matrix = None
    best_sift_score = float('inf')  # 最小越好

    for template_name in template_files:
        template_path = os.path.join(templates_dir, template_name)
        template_image_color = cv2.imread(template_path)
        if template_image_color is None:
            print(f"Warning: 无法加载模板图像 '{template_name}'，跳过...")
            continue

        template_image_gray = cv2.cvtColor(template_image_color, cv2.COLOR_BGR2GRAY)

        print(f"检查模板: {template_name}")
        homography_matrix = sift_feature_matching(input_image_gray, template_image_gray)

        if homography_matrix is not None:
            match_score = calculate_sift_match_score(input_image_path, template_path)
            print(f"  -> 匹配得分: {match_score:.2f}")

            if match_score < best_sift_score:
                best_sift_score = match_score
                best_sift_template_path = template_path
                best_sift_homography_matrix = homography_matrix

    if best_sift_template_path:
        label_base = os.path.basename(best_sift_template_path)
        label_name = os.path.splitext(label_base)[0] + '.txt'
        label_path = os.path.join(os.path.dirname(best_sift_template_path).replace('images', 'labels'), label_name)

        print(f"\n✅ 最佳匹配模板: {label_base}, 标签文件: {label_name}")
        template_shape = cv2.imread(best_sift_template_path).shape[:2]

        # result_image, recognition_score = visualize_annotations(
        result_image, match_count = visualize_annotations(
            input_image_path,
            label_path,
            classes_file_path,
            os.path.join(output_dir, os.path.basename(input_image_path)),
            best_sift_homography_matrix,
            template_shape,
            'SIFT',
            class_names,
            class_colors
        )

        scaled_score = scale_sift_score(best_sift_score)
        end_time = time.time()
        elapsed = (end_time - start_time) * 1000

        print(f"SIFT匹配度（满分100）: {scaled_score:.2f}，运行耗时：{elapsed:.2f} 毫秒")
        # print(f"部件识别率：{recognition_score:.1f}%")
        print(f"部件识别数：{match_count}")
        # return result_image, scaled_score, elapsed
        return result_image, match_count, elapsed
    else:
        print("❌ 没有找到任何合适的 SIFT 匹配模板")
        end_time = time.time()
        # ✅ 即使没有匹配，也绘制图例
        legend_x, legend_y = 10, 10
        for idx, (cls_id, name) in enumerate(class_names.items()):
            color = class_colors[int(cls_id)]
            y_pos = legend_y + idx * 20
            cv2.rectangle(input_image_color, (legend_x, y_pos), (legend_x + 15, y_pos + 15), color, -1)
            cv2.putText(input_image_color, name, (legend_x + 20, y_pos + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 保存带图例的图像
        output_path = os.path.join(output_dir, os.path.basename(input_image_path))
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(output_path, input_image_color)
        elapsed_time = (end_time - start_time) * 1000
        return input_image_color, 0, elapsed_time

if __name__ == '__main__':
    input_image_path = input("请输入单张图片的路径: ")
    templates_dir = './Recon/train/images'
    classes_file_path = './Recon/train/labels/classes.txt'
    output_dir = './Recon/output_sift'
    class_names = {0: 'main-body', 1: 'Left-Solar-Array', 2:'Right-Solar-Array',3: 'Radar', 4: 'LENS', 5: 'Antenna'}
    class_colors = {
        0: (255, 0, 0),  # 红色
        1: (0, 255, 0),  # 绿色
        2: (0, 0, 255),  # 蓝色
        3: (0, 255, 255),  # 黄色
        4: (255, 255, 0),  # 天蓝色
        5: (255, 0, 255),  # 品红
    }
    process_sift_images(input_image_path,templates_dir, classes_file_path, output_dir, class_names, class_colors)