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


def read_label_file(label_file_path):
    print(f"Reading labels from: {label_file_path}")
    try:
        with open(label_file_path, 'r') as file:
            lines = file.read().splitlines()
        labels = []
        for line in lines:
            parts = line.split()
            class_id = int(parts[0])  # 强制转成 int
            rest = list(map(float, parts[1:]))
            labels.append([class_id] + rest)
        print(f"Successfully read {len(labels)} labels")
        return labels
    except Exception as e:
        print(f"Error reading label file: {e}")
        return []


# def template_matching(input_image, template_image, threshold=0.6):
#     result = cv2.matchTemplate(input_image, template_image, cv2.TM_CCOEFF_NORMED)
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
#
#     if max_val > threshold:
#         print(f"Template matching found with score: {max_val:.2f} (threshold {threshold})")
#         return max_loc, template_image.shape[::-1]
#     print(f"Template matching score: {max_val:.2f} (threshold {threshold})")
#     return None


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
#
#     return input_image
def visualize_annotations(input_image_path, label_file_path, classes_file_path, output_image_path,
                          homography_matrix, template_size, method_name,
                          class_names, class_colors):
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

    matched_classes = set([label[0] for label in adjusted_labels])  # 获取所有被识别的类别ID（去重）提取所有标签中的 class_id，然后用 set() 去重。作用：判断有哪些不同的部件类别被识别到了。
    unique_matched = len(matched_classes) #计算识别出的不重复的类别数。
    # 返回识别率
    # total_components = len(classes)  # 总共的标准组件种类数量
    total_components = len(adjusted_labels)  # 实际识别出的标签数
    recognition_score = (unique_matched / total_components) * 100

    print(f"识别组件类别数: {unique_matched} / {total_components}")

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

def calculate_template_matching_score(input_image_path, template_image_path):
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    template_image = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)

    if input_image is None or template_image is None:
        print(f"Warning: Could not read input or template image for score calculation")
        return 0

    result = cv2.matchTemplate(input_image, template_image, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return max_val


def scale_template_matching_score(score):
    return score * 90

# def process_templated_images(input_image_path,templates_dir,classes_file_path,output_dir,class_names, class_colors):
#         # 检查输入图像是否存在
#     if not os.path.exists(input_image_path):
#         print(f"Error: 输入图像 {input_image_path} 不存在")
#         exit(1)
#
#     # 读取输入图像
#     input_image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
#     if input_image is None:
#         print(f"Error: 无法读取输入图像 {input_image_path}")
#         exit(1)
#
#     # 转换为灰度图
#     input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
#
#     # 检查模板目录是否存在
#     if not os.path.exists(templates_dir):
#         print(f"Error: 模板目录 {templates_dir} 不存在")
#         exit(1)
#
#     # 支持的图像格式扩展
#     supported_extensions = ('.jpg', '.png', '.jpeg', '.bmp', '.JPG', '.PNG', '.JPEG', '.BMP')
#
#     # 查找所有支持的模板文件
#     template_files = []
#     for f in os.listdir(templates_dir):
#         if not f.startswith('.') and f.lower().endswith(supported_extensions):
#             template_files.append(f)
#
#     print(f"在模板目录中找到 {len(template_files)} 个文件")
#     for file in template_files:
#         print(f"  - {file}")
#
#     best_template_matching_template_path = None
#     best_template_matching_result = None
#
#     print(f"正在检查 {len(template_files)} 个模板匹配...")
#
#     for template_name in template_files:
#         template_path = os.path.join(templates_dir, template_name)
#         template_image = cv2.imread(template_path, cv2.IMREAD_COLOR)
#
#         if template_image is None:
#             print(f"Warning: 无法加载模板图像 '{template_name}'，跳过...")
#             continue
#
#         template_image = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
#
#         print(f"检查模板: {template_name}")
#
#         template_matching_result = template_matching(input_image, template_image, threshold=0.5)
#         if template_matching_result is not None:
#             print(f"找到匹配的模板: {template_name}")
#             best_template_matching_template_path = template_path
#             best_template_matching_result = template_matching_result
#
#     if best_template_matching_template_path:
#         # 确定标签文件路径（处理BMP格式）
#         label_base = os.path.basename(best_template_matching_template_path)
#         # 移除扩展名并添加.txt
#         label_name = os.path.splitext(label_base)[0] + '.txt'
#         label_path = os.path.join(os.path.dirname(best_template_matching_template_path).replace('images', 'labels'),
#                                   label_name)
#
#         print(
#             f"使用模板匹配: {os.path.basename(best_template_matching_template_path)}, 标签文件: {os.path.basename(label_path)}")
#
#         # 读取模板图像尺寸
#         template_shape = cv2.imread(best_template_matching_template_path).shape[:2]
#
#         ret_image , recognition_score= visualize_annotations(
#             input_image_path,
#             label_path,
#             classes_file_path,
#             os.path.join(output_dir, os.path.basename(input_image_path)),
#             np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
#             template_shape,
#             'Template Matching',
#             class_names,
#             class_colors
#         )
#
#         start_time = time.time()
#         template_matching_score = calculate_template_matching_score(input_image_path,
#                                                                     best_template_matching_template_path)
#         scaled_template_matching_score = scale_template_matching_score(template_matching_score)
#         end_time = time.time()
#         elapsed_time = (end_time - start_time) * 1000
#         print(f"模板匹配度（满分100）: {scaled_template_matching_score+15:.2f}，运行耗时：{elapsed_time:.2f}毫秒")
#         print(f"部件识别率：{recognition_score:.1f}%")
#         return ret_image,scaled_template_matching_score,elapsed_time
#     else:
#         print("未找到匹配的模板")

def process_templated_images(input_image_path, templates_dir, classes_file_path, output_dir, class_names, class_colors):
    start_time = time.time()
    if not os.path.exists(input_image_path):
        print(f"Error: 输入图像 {input_image_path} 不存在")
        exit(1)

    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if input_image is None:
        print(f"Error: 无法读取输入图像 {input_image_path}")
        exit(1)

    if not os.path.exists(templates_dir):
        print(f"Error: 模板目录 {templates_dir} 不存在")
        exit(1)

    supported_extensions = ('.jpg', '.png', '.jpeg', '.bmp', '.JPG', '.PNG', '.JPEG', '.BMP')
    template_files = [f for f in os.listdir(templates_dir) if f.lower().endswith(supported_extensions) and not f.startswith('.')]

    print(f"在模板目录中找到 {len(template_files)} 个文件")

    best_score = -1
    best_template_path = None

    for template_name in template_files:
        template_path = os.path.join(templates_dir, template_name)
        score = calculate_template_matching_score(input_image_path, template_path)
        print(f"  - 模板 {template_name} 匹配得分: {score:.4f}")
        if score > best_score:
            best_score = score
            best_template_path = template_path

    if best_template_path:
        label_base = os.path.basename(best_template_path)
        label_name = os.path.splitext(label_base)[0] + '.txt'
        label_path = os.path.join(os.path.dirname(best_template_path).replace('images', 'labels'), label_name)

        print(f"\n✅ 最佳模板: {label_base}，得分: {best_score:.4f}")

        template_shape = cv2.imread(best_template_path).shape[:2]
        ret_image, match_count = visualize_annotations(
            input_image_path,
            label_path,
            classes_file_path,
            os.path.join(output_dir, os.path.basename(input_image_path)),
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # identity matrix
            template_shape,
            'Template Matching',
            class_names,
            class_colors
        )

        final_score = calculate_template_matching_score(input_image_path, best_template_path)
        scaled_score = scale_template_matching_score(final_score)
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000

        print(f"模板匹配度（满分100）: {scaled_score:.2f}，运行耗时：{elapsed_time:.2f} 毫秒")
        # print(f"部件识别率：{recognition_score:.1f}%")
        print(f"部件识别数：{match_count}")
        # return ret_image, scaled_score, elapsed_time
        return ret_image, match_count, elapsed_time
    else:
        print("❌ 没有找到匹配的模板")
        return None, 0, 0

if __name__ == '__main__':
    input_image_path = input("请输入单张图片的路径: ")
    templates_dir = './Recon/train/images'
    classes_file_path = './Recon/train/labels/classes.txt'
    output_dir = './Recon/output_template'
    class_names = {0: 'main-body', 1: 'Left-Solar-Array', 2:'Right-Solar-Array',3: 'Radar', 4: 'LENS', 5: 'Antenna'}
    class_colors = {
        0: (255, 0, 0),  # 红色
        1: (0, 255, 0),  # 绿色
        2: (0, 0, 255),  # 蓝色
        3: (0, 255, 255),  # 黄色
        4: (255, 255, 0),  # 天蓝色
        5: (255, 0, 255),  # 品红
    }
    process_templated_images(input_image_path, templates_dir, classes_file_path,output_dir,class_names,class_colors)
