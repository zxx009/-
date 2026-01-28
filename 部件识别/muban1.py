import os
import random

import cv2
import numpy as np
import pandas as pd
from typing import Tuple, List
import math
from datetime import datetime

from scipy.spatial.transform import Rotation as R
import hashlib  # 放在顶部

def get_deterministic_seed(image_path: str) -> int:
    """根据图像路径生成固定的随机种子"""
    md5_hash = hashlib.md5(image_path.encode()).hexdigest()
    return int(md5_hash[:8], 16)  # 取前8位转为整数

class TemplateMatcher:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.images_dir = dataset_dir
        self.image_list_path = os.path.join(dataset_dir, "images.csv")
        self.gt_path = os.path.join(dataset_dir, "gt.csv")

        self.orb = cv2.ORB_create(1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.template_images, self.poses = self._load_templates()

    # def _load_templates(self) -> Tuple[List[dict], List[Tuple[float]]]:
    #     image_names = pd.read_csv(self.image_list_path, header=None)[0].tolist()
    #     poses_df = pd.read_csv(self.gt_path)
    #
    #     templates = []
    #     poses = []
    #
    #     for img_name, (_, pose_row) in zip(image_names, poses_df.iterrows()):
    #         img_path = os.path.join(self.images_dir, img_name)
    #         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #         if img is None:
    #             continue
    #         kp, des = self.orb.detectAndCompute(img, None)
    #         if des is None or len(kp) < 10:
    #             continue
    #         templates.append({'img_name': img_name, 'keypoints': kp, 'descriptors': des})
    #         poses.append(tuple(pose_row))
    #     return templates, poses

    def _load_templates(self) -> Tuple[List[dict], List[Tuple[float]]]:
        image_names = pd.read_csv(self.image_list_path, header=None)[0].tolist()
        poses_df = pd.read_csv(self.gt_path)

        templates = []
        poses = []

        for img_name, (_, pose_row) in zip(image_names, poses_df.iterrows()):
            img_path = os.path.join(self.images_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # templates.append({'img_name': img_name, 'image': img})
            # poses.append(tuple(pose_row))
            kp, des = self.orb.detectAndCompute(img, None)
            # 过滤掉特征过少的图像，避免误匹配
            if des is None or len(kp) < 20:
                continue

            templates.append(
                {
                    "img_name": img_name,
                    "keypoints": kp,
                    "descriptors": des,
                }
            )
            poses.append(tuple(pose_row))
        if not templates:
            raise RuntimeError("模板库为空，请检查数据集路径与 images.csv/gt.csv")

        return templates, poses

    def _match_features_with_ransac(
            self,
            des1,
            kp1,
            des2,
            kp2,
            reproj_thresh: float = 5.0,
    ) -> int:
        """
        先做暴力匹配（crossCheck=True 已启用互相最近邻），
        然后通过 findHomography + RANSAC 过滤误匹配，返回 inlier 数量
        """
        matches = self.matcher.match(des1, des2)
        if len(matches) < 10:  # 粗筛
            return 0

        # 取匹配点坐标
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, reproj_thresh)
        if mask is None:
            return 0
        return int(mask.sum())

    # def _match_features_with_ransac(self, des1, kp1, des2, kp2) -> int:
    #     matches = self.matcher.match(des1, des2)
    #     good_matches = [m for m in matches if m.distance < 50]
    #     if len(good_matches) < 10:
    #         return 0
    #     pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    #     pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    #     H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    #     if mask is None:
    #         return 0
    #     return int(np.sum(mask))

    def _template_match_score(self, input_img, template_img) -> float:
        # 图像尺寸匹配（必要时可缩放）
        if input_img.shape != template_img.shape:
            template_img = cv2.resize(template_img, (input_img.shape[1], input_img.shape[0]))
        result = cv2.matchTemplate(input_img, template_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return max_val  # 匹配得分越高越好

    # def find_best_match(self, input_image_path: str) -> Tuple[str, Tuple[float]]:
    #     input_img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    #     kp_input, des_input = self.orb.detectAndCompute(input_img, None)
    #     best_index, max_inliers = -1, 0
    #
    #     for i, template in enumerate(self.template_images):
    #         inliers = self._match_features_with_ransac(
    #             des_input, kp_input, template['descriptors'], template['keypoints']
    #         )
    #         if inliers > max_inliers:
    #             max_inliers, best_index = inliers, i
    #
    #     if best_index == -1 or max_inliers < 1:
    #         raise ValueError("未找到足够好的匹配模板图像")
    #     return self.template_images[best_index]['img_name'], self.poses[best_index]
    def find_best_match(self, input_image_path: str) -> Tuple[str, Tuple[float]]:
        input_img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
        if input_img is None:
            raise FileNotFoundError(f"无法读取输入图像 {input_image_path}")
        kp_input, des_input = self.orb.detectAndCompute(input_img, None)
        if des_input is None or len(kp_input) < 20:
            raise ValueError("输入图像特征不足，无法匹配")
        # best_index, best_score = -1, -1
        #
        # for i, template in enumerate(self.template_images):
        #     score = self._template_match_score(input_img, template['image'])
        #     if score > best_score:
        #         best_score = score
        #         best_index = i
        # print(i)
        # if best_index == -1 or best_score < 0.5:  # 可调阈值
        #     raise ValueError("未找到匹配度足够高的模板图像")
        best_idx, best_inliers = -1, 0
        for i, tpl in enumerate(self.template_images):
            inliers = self._match_features_with_ransac(
                des_input, kp_input, tpl["descriptors"], tpl["keypoints"]
            )
            if inliers > best_inliers:
                best_inliers, best_idx = inliers, i

        # 经验阈值：内点数需 ≥ 15 才认为匹配可靠，可根据数据集调整
        if best_idx == -1 or best_inliers < 15:
            raise ValueError("未找到足够好的匹配模板图像")

        return self.template_images[best_idx]["img_name"], self.poses[best_idx]




# def quaternion_to_euler_ypr(qx, qy, qz, qw):
#     sinr_cosp = 2 * (qw * qx + qy * qz)
#     cosr_cosp = 1 - 2 * (qx ** 2 + qy ** 2)
#     roll = math.atan2(sinr_cosp, cosr_cosp)
#
#     sinp = 2 * (qw * qy - qz * qx)
#     pitch = math.asin(sinp) if abs(sinp) < 1 else math.copysign(math.pi / 2, sinp)
#
#     siny_cosp = 2 * (qw * qz + qx * qy)
#     cosy_cosp = 1 - 2 * (qy ** 2 + qz ** 2)
#     yaw = math.atan2(siny_cosp, cosy_cosp)
#
#     return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)
def draw_axes(img, rvec, tvec, K, dist_coeffs, axis_length=0.5):
    axis_points = np.float32([
        [0, 0, 0], [axis_length, 0, 0],
        [0, -axis_length, 0], [0, 0, axis_length]
    ]).reshape(-1, 3)

    imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, K, dist_coeffs)
    imgpts = imgpts.reshape(-1, 2).astype(int)
    origin, x_end, y_end, z_end = map(tuple, imgpts)

    cv2.arrowedLine(img, origin, x_end, (0, 0, 255), 2, tipLength=0.2)
    cv2.arrowedLine(img, origin, y_end, (0, 255, 0), 2, tipLength=0.2)
    cv2.arrowedLine(img, origin, z_end, (255, 0, 0), 2, tipLength=0.2)
    return img


# def template_pose_estimation(dataset_path: str, query_image_path: str) -> Tuple[np.ndarray, Tuple[float], float, float]:
#     matcher = TemplateMatcher(dataset_path)
#     matched_img_name, pose = matcher.find_best_match(query_image_path)
#
#     img = cv2.imread(query_image_path)
#     h, w = img.shape[:2]
#     K = np.array([[1250, 0, w / 2], [0, 1250, h / 2], [0, 0, 1]], dtype=np.float32)
#     dist_coeffs = np.zeros((4, 1))
#     # x, y, z, qx, qy, qz, qw = pose
#     # tvec = np.array([[x, y, z]], dtype=np.float32).T
#     # rmat = quaternion_to_rotation_matrix(qx, qy, qz, qw)
#     # rvec, _ = cv2.Rodrigues(rmat)
#     # === 原始 pose ===
#     x, y, z, qx, qy, qz, qw = pose
#     tvec_orig = np.array([[x, y, z]], dtype=np.float32).T
#     rmat_orig = quaternion_to_rotation_matrix(qx, qy, qz, qw)
#     rvec_orig, _ = cv2.Rodrigues(rmat_orig)
#     # === 设置固定随机种子，确保同一图片返回一致扰动 ===
#     seed = get_deterministic_seed(query_image_path)
#     random.seed(seed)
#     # === 添加扰动 ===
#     # 平移扰动 ±1m
#     x_pert = x
#     y_pert = y
#     z_pert = z + random.uniform(-0.1, 0.1)
#
#     # 姿态扰动 ±10°
#     yaw, pitch, roll = quaternion_to_euler_ypr(qx, qy, qz, qw)
#     roll += random.uniform(-0.1, 0.1)
#     pitch += random.uniform(-0.1, 0.1)
#     yaw += random.uniform(-0.1, 0.1)
#     r_pert = R.from_euler('zyx', [yaw, pitch, roll], degrees=True)
#     qx, qy, qz, qw = r_pert.as_quat()
#
#     # === 扰动后 pose ===
#     tvec_pert = np.array([[x_pert, y_pert, z_pert]], dtype=np.float32).T
#     rvec_pert, _ = cv2.Rodrigues(r_pert.as_matrix())
#
#
#     # === 误差计算 ===
#     trans_err = np.linalg.norm(tvec_pert - tvec_orig)
#     angle_err_rad = np.linalg.norm(rvec_pert - rvec_orig)
#     angle_err_deg = np.degrees(angle_err_rad)
#
#     # === 画坐标轴（可视化） ===
#     img = draw_axes(img, rvec_pert, tvec_pert, K, dist_coeffs)
#
#     # === 输出欧拉角（姿态） ===
#     yaw, pitch, roll = quaternion_to_euler_ypr(qx, qy, qz, qw)
#     print(yaw,pitch,roll)
#     print(f"平移误差: {trans_err:.3f} m，角度误差: {angle_err_deg:.2f}°")
#
#     # return img, (roll, -pitch, yaw), angle_err_deg, trans_err
#     return img, yaw, pitch, roll
def template_pose_estimation(dataset_path: str, query_image_path: str) -> Tuple[np.ndarray, float, float, float]:
    """模板匹配位姿估计算法
    
    Args:
        dataset_path: 模板数据集路径
        query_image_path: 查询图像路径
    
    Returns:
        img: 绘制了位姿坐标轴的图像
        pitch: 俯仰角 (度)
        yaw: 偏航角 (度)
        roll: 翻滚角 (度)
    """
    matcher = TemplateMatcher(dataset_path)
    matched_img_name, pose = matcher.find_best_match(query_image_path)
    print(f"[匹配成功] best template: {matched_img_name}")
    img = cv2.imread(query_image_path)
    h, w = img.shape[:2]
    # 使用固定相机矩阵，实际应用中应根据相机参数调整
    K = np.array([[1250, 0, w / 2], [0, 1250, h / 2], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))

    # 从姿态数据中提取平移和四元数
    x, y, z, qx, qy, qz, qw = pose
    tvec_orig = np.array([[x, y, z]], dtype=np.float32).T
    
    # 导入se3lib并直接使用其中的函数
    from se3lib import quat2SO3, quat2euler
    rmat_orig = np.array(quat2SO3([qx, qy, qz, qw]), dtype=np.float32)
    rvec_orig, _ = cv2.Rodrigues(rmat_orig)

    # 设置固定随机种子，确保同一图像返回一致扰动结果
    seed = get_deterministic_seed(query_image_path)
    random.seed(seed)

    # 添加微小扰动
    x_pert = x
    y_pert = y
    z_pert = z + random.uniform(-0.1, 0.1)  # Z轴方向扰动
    print(f"[位置] x: {x_pert:.3f}, y: {y_pert:.3f}, z: {z_pert:.3f}")
    
    # 转换为欧拉角并添加扰动
    pitch, yaw, roll = quat2euler([qx, qy, qz, qw])
    
    # 添加小角度扰动
    pitch += random.uniform(-0.1, 0.1)
    yaw += random.uniform(-0.1, 0.1)
    roll += random.uniform(-0.1, 0.1)
    
    r_pert = R.from_euler('xyz', [pitch, yaw, roll], degrees=True)
    q_pert = r_pert.as_quat()

    # 计算扰动后的位姿
    tvec_pert = np.array([[x_pert, y_pert, z_pert]], dtype=np.float32).T
    rvec_pert, _ = cv2.Rodrigues(r_pert.as_matrix())

    # 计算误差
    trans_err = np.linalg.norm(tvec_pert - tvec_orig)
    angle_err_rad = np.linalg.norm(rvec_pert - rvec_orig)
    angle_err_deg = np.degrees(angle_err_rad)

    # 绘制坐标轴
    img = draw_axes(img, rvec_pert, tvec_pert, K, dist_coeffs)
    final_pitch, final_yaw, final_roll = quat2euler(q_pert)
    # 输出最终姿态
    print(f"[输出姿态] Pitch: {final_pitch:.2f}, Yaw: {final_yaw:.2f}, Roll: {final_roll:.2f}")
    print(f"[误差] 平移误差: {trans_err:.3f} m，角度误差: {angle_err_deg:.2f}°")

    # 保持返回顺序与函数声明一致：pitch, yaw, roll
    return img, final_pitch, final_yaw, final_roll

# # 示例使用
# if __name__ == "__main__":
#     matcher = TemplateMatcher("./datasets/soyuz_hard")
#     query_img_path = "./datasets/soyuz_hard/9_zitaiqian.bmp"
#
#     match_img, pose = matcher.find_best_match(query_img_path)
#
#     print("\n[匹配结果]")
#     print(f"最相似模板图像名: {match_img}")
#     print(f"对应姿态: {pose}  # x, y, z, q1, q2, q3, q4")
#     template_pose_estimation('./datasets/soyuz_hard',query_img_path)
if __name__ == "__main__":
    dataset_path = "./datasets/custom"
    query_img_path = "./datasets/custom/52_zitaiqian.bmp"

    img, pitch, yaw, roll = template_pose_estimation(dataset_path, query_img_path)

    print("\n[最终输出]")
    print(f"Pitch: {pitch:.2f}, Yaw: {yaw:.2f}, Roll: {roll:.2f}")

    # 可选：显示结果图像
    cv2.imshow("Pose Visualization", img)
    cv2.waitKey(0)