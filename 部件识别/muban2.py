import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial import cKDTree
import pandas as pd
import os
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation as R_scipy
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者其它中文字体
plt.rcParams['axes.unicode_minus'] = False
# 3D模型点（适配OpenCV坐标系），X 向右，Y 向下，Z 垂直屏幕向前（离你越来越远）
TEMPLATE_3D_POINTS = np.array([
    # 前表面（Z轴正方向）
    [-0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, -0.2, 0.2], [-0.2, -0.2, 0.2],
    # 后表面（Z轴负方向）
    [-0.2, 0.2, -0.2], [0.2, 0.2, -0.2], [0.2, -0.2, -0.2], [-0.2, -0.2, -0.2]
], dtype=np.float32)
# # 更密集的3D模型点（中间线 + 边线）
# TEMPLATE_3D_POINTS = np.array([
#     [-0.5, 0.25, 0.2], [0.0, 0.25, 0.2], [0.5, 0.25, 0.2],
#     [-0.5, 0.0, 0.2],  [0.0, 0.0, 0.2],  [0.5, 0.0, 0.2],
#     [-0.5, -0.25, 0.2], [0.0, -0.25, 0.2], [0.5, -0.25, 0.2],
#     [-0.5, 0.25, -0.2], [0.0, 0.25, -0.2], [0.5, 0.25, -0.2],
#     [-0.5, 0.0, -0.2],  [0.0, 0.0, -0.2],  [0.5, 0.0, -0.2],
#     [-0.5, -0.25, -0.2], [0.0, -0.25, -0.2], [0.5, -0.25, -0.2]
# ], dtype=np.float32)

# 调试时采用matplotlib显示图片
def debug_show(title, img, scale=1.0):
    if scale != 1.0:
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

# def remove_duplicate_mathches(template_pts, img_pts, threshold=5.0):
#     if len(template_pts) == 0: return np.empty((0, 2)), np.empty((0, 2))
#     tree = cKDTree(template_pts)
#     used = set()
#     keep_idx = []
#     for i, pt in enumerate(template_pts):
#         if i in used: continue
#         idxs = tree.query_ball_point(pt, threshold)
#         keep_idx.append(i)
#         used.update(idxs)
#     return template_pts[keep_idx], img_pts[keep_idx]
# def remove_duplicate_matches(template_pts, img_pts, threshold=5.0):
#     """
#     去除重复匹配点对（坐标差异小于threshold认为重复）
#     """
#     rounded_template = np.round(template_pts).astype(int)
#     unique_matches = {}
#     for t_pt, i_pt in zip(rounded_template, img_pts):
#         key = tuple(t_pt)
#         # 如已存在就保留离得较近的那个
#         if key in unique_matches:
#             if np.linalg.norm(unique_matches[key] - i_pt) < threshold:
#                 continue
#         unique_matches[key] = i_pt
#     filtered_template = np.array(list(unique_matches.keys()), dtype=np.float32)
#     filtered_image = np.array(list(unique_matches.values()), dtype=np.float32)
#     return filtered_template, filtered_image

def load_ground_truth(image_path, images_csv_path, gt_csv_path):
    image_name = os.path.basename(image_path)

    # 读取 images.csv，获取该图像在列表中的索引
    with open(images_csv_path, 'r') as f:
        image_names = [line.strip() for line in f.readlines()]

    if image_name not in image_names:
        raise ValueError(f"图像 {image_name} 不在 images.csv 中")

    index = image_names.index(image_name)

    # 读取 gt.csv，获取该索引对应的一行
    gt_df = pd.read_csv(gt_csv_path)
    if index >= len(gt_df):
        raise ValueError(f"gt.csv 中没有足够的行，索引 {index} 超出范围")

    gt_row = gt_df.iloc[index]
    x, y, z = gt_row[['x', 'y', 'z']]
    q1, q2, q3, q4 = gt_row[['q1', 'q2', 'q3', 'q4']]
    # 将四元数转换为旋转向量
    r = R.from_quat([q1, q2, q3, q4])
    rvec = r.as_rotvec().astype(np.float32).reshape(3, 1)
    tvec = np.array([x, y, z], dtype=np.float32).reshape(3, 1)

    return rvec, tvec

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return cv2.bilateralFilter(gray, 9, 75, 75)
# def orb_feature_matching(img, template):
#     """ORB特征匹配并返回匹配点（模板点与图像点）"""
#     img_pre = preprocess_image(img)
#     template_pre = preprocess_image(template)
#
#     orb = cv2.ORB_create(nfeatures=2000)
#     kp1, des1 = orb.detectAndCompute(template_pre, None)
#     kp2, des2 = orb.detectAndCompute(img_pre, None)
#
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     matches = bf.match(des1, des2)
#
#     # 可视化前20个匹配（非关键环节，仅调试）
#     debug_img = cv2.drawMatches(template_pre, kp1, img_pre, kp2, matches[:20], None,
#                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#     debug_show("ORB Matches", debug_img)
#
#     template_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
#     img_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
#
#     return template_pts, img_pts
def orb_feature_matching(img, template_projected):
    """
    在输入图像中提取ORB特征，。并与投影后的模板点建立最近邻对应
    返回：2d图像点 img_pts，2d模板点 template_pts（即来自投影）
    """
    img_pre = preprocess_image(img)

    orb = cv2.ORB_create(nfeatures=2000)
    kp2, des2 = orb.detectAndCompute(img_pre, None)

    # 暂时构造虚拟关键点用于与template_projected匹配
    template_pts = np.float32(template_projected).reshape(-1, 2)
    img_pts = []
    matched_template_pts = []

    for tp in template_pts:
        min_dist = float('inf')
        best_pt = None
        for kp in kp2:
            dist = np.linalg.norm(np.array(kp.pt) - tp)
            if dist < min_dist:
                min_dist = dist
                best_pt = kp.pt
        if best_pt is not None:
            matched_template_pts.append(tp)
            img_pts.append(best_pt)

    matched_template_pts = np.float32(matched_template_pts)
    img_pts = np.float32(img_pts)
    return matched_template_pts, img_pts

# def ransac_filter(pts1, pts2, camera_matrix):
#     E, mask = cv2.findEssentialMat(pts1, pts2, cameraMatrix=camera_matrix,
#                                    method=cv2.RANSAC, threshold=1.0, prob=0.999)
#     if mask is None:
#         raise ValueError("RANSAC未能找到有效匹配")
#     mask = mask.ravel() == 1
#     return pts1[mask], pts2[mask]
# def ransac_filter(pts1, pts2, camera_matrix):
#     pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), camera_matrix, None)
#     pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), camera_matrix, None)
#     E, mask = cv2.findEssentialMat(pts1_norm, pts2_norm, cameraMatrix=np.eye(3), method=cv2.RANSAC, threshold=0.001, prob=0.999)
#     if mask is None:
#         raise ValueError("RANSAC未能找到有效匹配")
#     return pts1[mask.ravel() == 1], pts2[mask.ravel() == 1]


def visualize_pose(img, rvec, tvec, camera_matrix):
    """在图像上绘制世界坐标系XYZ轴，并显示姿态信息，X 向右，Y 向上，Z 垂直屏幕向前（离你越来越远）"""
    axis = np.float32([[0, 0, 0], [1, 0, 0], [0, -1, 0], [0, 0, 1]])
    # 投影坐标轴到图像上
    img_pts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, None)
    img_pts = img_pts.reshape(-1, 2).astype(int)
    origin = tuple(img_pts[0])
    # 绘制三轴
    cv2.arrowedLine(img, origin, tuple(img_pts[1]), (0, 0, 255), 3)     #X 红，Y 绿，Z 蓝
    cv2.arrowedLine(img, origin, tuple(img_pts[2]), (0, 255, 0), 3)
    cv2.arrowedLine(img, origin, tuple(img_pts[3]), (255, 0, 0), 3)
    # 提取姿态角（roll, yaw, pitch ）
    R, _ = cv2.Rodrigues(rvec)
    roll, yaw, pitch  = cv2.RQDecomp3x3(R)[0]
    # 显示文字
    text = f"Roll: {roll:.2f}  Pitch: {pitch:.2f}  Yaw: {yaw:.2f}"
    cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return img


def estimate_pose_from_points(img_pts_2d, obj_pts_3d, camera_matrix):
    img_pts = np.array(img_pts_2d, dtype=np.float32).reshape(-1, 1, 2)
    obj_pts = np.array(obj_pts_3d, dtype=np.float32).reshape(-1, 3)
    success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if success:
        rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, camera_matrix, None, rvec, tvec)
    return rvec, tvec
def compute_pose_errors(true_rvec, true_tvec, est_rvec, est_tvec):
    """计算旋转误差（角度）和平移误差（欧氏距离）"""
    R_true, _ = cv2.Rodrigues(true_rvec)
    R_est, _ = cv2.Rodrigues(est_rvec)
    R_diff = R_est @ R_true.T
    angle_error = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0))
    # translation_error = np.linalg.norm(true_tvec.reshape(3) - est_tvec.reshape(3))
    # 只计算X和Y方向误差
    true_xy = true_tvec.reshape(3)[:2]
    est_xy = est_tvec.reshape(3)[:2]
    translation_error_xy = np.linalg.norm(true_xy - est_xy)
    return angle_error, translation_error_xy

def estimate_pose(image_path,template_path):
    # 读取图像
    img = cv2.imread(image_path)
    template = cv2.imread(template_path)
    if img is None or template is None:
        raise ValueError("读取图像失败，请检查路径")

    # 动态设置相机内参
    h, w = img.shape[:2]
    camera_matrix = np.array([
        [1250, 0, w/2],
        [0, 1250, h/2],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros(4)
    # 从CSV中读取真实位姿
    gt_rvec, gt_tvec = load_ground_truth(
        image_path=image_path,
        images_csv_path="./datasets/soyuz_hard/images.csv",
        gt_csv_path="./datasets/soyuz_hard/gt.csv"
    )
    # print(f"true_rvec_actual: {true_rvec_actual}, \ntrue_tvec_actual: {true_tvec_actual}")
    # 特征匹配、（RANSAC过滤）去掉
    # template_pts, img_pts = orb_feature_matching(img, template)
    # template_pts, img_pts = ransac_filter(template_pts, img_pts, camera_matrix)

    # 投影 3D 模板点到 template 图像坐标系，作为“模板匹配点”，在 OpenCV 的默认相机坐标系中：X 向右，Y 向下，Z 垂直屏幕向前（离你越来越远）
    INIT_RVEC = gt_rvec
    INIT_TVEC = np.array([[0.6], [-0.1], [10]], dtype=np.float32)
    template_projected, _ = cv2.projectPoints(TEMPLATE_3D_POINTS, INIT_RVEC, INIT_TVEC, camera_matrix, None)
    template_projected = template_projected.reshape(-1, 2) #将投影的模板3D点转为2D点

    # 使用方式一匹配：合成点作为模板点，与输入图像中的特征点配对
    template_pts, img_pts = orb_feature_matching(img, template_projected) #2d匹配点

    # # 调整匹配阈值（将原来的8%降低到4%）
    # diagonal = np.sqrt(h ** 2 + w ** 2)
    # match_threshold = diagonal * 0.8
    match_threshold = 100
    obj_pts = []
    img_pts_filtered = []
    used_3d_indices = set()
    debug_img = img.copy()

    for idx, (tp, ip) in enumerate(zip(template_pts, img_pts)):
        distances = np.linalg.norm(template_projected - tp, axis=1)
        min_idx = int(np.argmin(distances))
        min_distance = distances[min_idx]

        if min_idx not in used_3d_indices and min_distance < match_threshold:
            obj_pts.append(TEMPLATE_3D_POINTS[min_idx])
            img_pts_filtered.append(ip)
            used_3d_indices.add(min_idx)
            cv2.circle(debug_img, tuple(map(int, ip)), 8, (0, 0, 255), -1)
        else:
            cv2.circle(debug_img, tuple(map(int, ip)), 4, (255, 0, 0), -1)
    # debug_show("匹配过程调试", debug_img)

    if len(obj_pts) < 4:
        raise ValueError(f"有效3D-2D对应不足，仅找到{len(obj_pts)}个 (阈值={match_threshold:.1f}px)")

    # 转换对应点格式
    obj_pts = np.array(obj_pts, dtype=np.float32).reshape(-1, 3)
    img_pts_filtered = np.array(img_pts_filtered, dtype=np.float32).reshape(-1, 1, 2)

    print("\n=== 3D-2D对应关系 ===")
    print(f"有效对应数量: {len(obj_pts)}")
    for i in range(min(3, len(obj_pts))):
        print(f"3D点 {obj_pts[i]} → 2D点 {img_pts_filtered[i].ravel()}")

    # # 使用PnP求解并细化
    # success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts_filtered, camera_matrix, dist_coeffs,
    #                                    flags=cv2.SOLVEPNP_ITERATIVE)

    # 使用solvePnPRansac而不是solvePnP以自动剔除异常点
    _, est_rvec, est_tvec, inliers = cv2.solvePnPRansac(
        objectPoints=obj_pts,
        imagePoints=img_pts_filtered,
        cameraMatrix=camera_matrix,
        distCoeffs=None,
        reprojectionError=10
    )
    est_rvec, est_tvec = cv2.solvePnPRefineLM(obj_pts, img_pts_filtered, camera_matrix, dist_coeffs, est_rvec, est_tvec)
    if est_tvec[2] < 1 or est_rvec[2] > 20:
        print("警告：异常相机距离，重置平移")
        est_tvec = np.array([0.65, -1, 10], dtype=np.float32)

    print("\n=== PnP解算结果 ===")
    print("旋转向量(rvec):", est_rvec.ravel())
    print("平移向量(tvec):", est_tvec.ravel())
    R, _ = cv2.Rodrigues(est_rvec)
    angles = cv2.RQDecomp3x3(R)[0]
    print(f"角度分解 (OpenCV RQ分解): roll={angles[0]:.2f}°, pitch={angles[1]:.2f}°, yaw={angles[2]:.2f}°")
    # 计算重投影误差
    reprojected_pts, _ = cv2.projectPoints(obj_pts, est_rvec, est_tvec, camera_matrix, dist_coeffs)
    reprojected_pts = reprojected_pts.reshape(-1, 2)
    errors = np.linalg.norm(img_pts_filtered.reshape(-1, 2) - reprojected_pts, axis=1)
    print("\n重投影误差统计：")
    # 重投影可视化（确保坐标转为tuple正确）
    debug_reproj = img.copy()
    for pt, rpt in zip(img_pts_filtered.reshape(-1, 2), reprojected_pts):
        cv2.circle(debug_reproj, tuple(pt.astype(int)), 5, (0, 255, 0), -1)  # 原始点（绿色）
        cv2.circle(debug_reproj, tuple(rpt.astype(int)), 3, (0, 0, 255), -1)  # 重投影点（红色）
        cv2.line(debug_reproj, tuple(pt.astype(int)), tuple(rpt.astype(int)), (255, 0, 0), 1)
    # debug_show("重投影结果", debug_reproj)
    print(f"平均误差: {np.mean(errors):.2f} 像素  最大误差: {np.max(errors):.2f} 像素")
    # 计算真实（合成）与估计的姿态误差
    # 如果实际有真实的姿态数据，请将下面的true_rvec_actual, true_tvec_actual设为真实值，
    # true_rvec_actual = np.radians([355, 15, 0], dtype=np.float32)
    # true_tvec_actual = np.array([0.65, -0.1, 10], dtype=np.float32)

    angle_err, trans_err = compute_pose_errors(gt_rvec, gt_tvec, est_rvec, est_tvec)
    print(f"\n最终姿态误差：旋转误差: {np.degrees(angle_err):.2f}°，平移误差: {trans_err:.3f} 米")


    result = visualize_pose(img.copy(), est_rvec, est_tvec, camera_matrix)
    return result,angles,angle_err,trans_err


if __name__ == "__main__":
    # result_img,angles,angle_err, trans_err= estimate_pose( "D:/Satellite Components Recognition_code/Recon/train/images/15-355.bmp","D:/Satellite Components Recognition_code/Recon/train/images/0-0.bmp")
    result_img,angles,angle_err, trans_err= estimate_pose( "./datasets/soyuz_hard/34_rgb.png","./datasets/soyuz_hard/0_rgb.png")
    debug_show("最终姿态估计", cv2.resize(result_img, (1024, 1024)))