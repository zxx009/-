import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False

# 更密集的3D模型点（中间线 + 边线）
TEMPLATE_3D_POINTS = np.array([
    [-0.5, 0.25, 0.2], [0.0, 0.25, 0.2], [0.5, 0.25, 0.2],
    [-0.5, 0.0, 0.2],  [0.0, 0.0, 0.2],  [0.5, 0.0, 0.2],
    [-0.5, -0.25, 0.2], [0.0, -0.25, 0.2], [0.5, -0.25, 0.2],
    [-0.5, 0.25, -0.2], [0.0, 0.25, -0.2], [0.5, 0.25, -0.2],
    [-0.5, 0.0, -0.2],  [0.0, 0.0, -0.2],  [0.5, 0.0, -0.2],
    [-0.5, -0.25, -0.2], [0.0, -0.25, -0.2], [0.5, -0.25, -0.2]
], dtype=np.float32)


def debug_show(title, img, scale=1.0):
    """用matplotlib显示图片"""
    if scale != 1.0:
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()


def preprocess_image(img):
    """灰度直方图均衡 + 双边滤波，提升ORB特征稳定性"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return cv2.bilateralFilter(gray, 9, 75, 75)


def orb_feature_matching(img, template_projected):
    """
    在输入图像中提取ORB特征，并与投影后的模板点建立最近邻对应。
    返回：模板点template_pts，图像点img_pts
    """
    img_pre = preprocess_image(img)
    orb = cv2.ORB_create(nfeatures=2000)
    kp2, des2 = orb.detectAndCompute(img_pre, None)

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

    return np.float32(matched_template_pts), np.float32(img_pts)


def visualize_pose(img, rvec, tvec, camera_matrix):
    """在图像上绘制世界坐标系XYZ轴，并显示姿态信息"""
    axis = np.float32([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    img_pts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, None)
    img_pts = img_pts.reshape(-1, 2).astype(int)
    origin = tuple(img_pts[0])
    cv2.arrowedLine(img, origin, tuple(img_pts[1]), (0, 0, 255), 3)  # X轴 - 红色
    cv2.arrowedLine(img, origin, tuple(img_pts[2]), (0, 255, 0), 3)  # Y轴 - 绿色
    cv2.arrowedLine(img, origin, tuple(img_pts[3]), (255, 0, 0), 3)  # Z轴 - 蓝色

    R, _ = cv2.Rodrigues(rvec)
    angles = cv2.RQDecomp3x3(R)[0]
    text = "Roll: {:.2f} Pitch: {:.2f} Yaw: {:.2f}".format(angles[0], angles[1], angles[2])
    cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return img


def estimate_pose_from_points(img_pts_2d, obj_pts_3d, camera_matrix):
    """用PnP方法估计姿态"""
    img_pts = np.array(img_pts_2d, dtype=np.float32).reshape(-1, 1, 2)
    obj_pts = np.array(obj_pts_3d, dtype=np.float32).reshape(-1, 3)
    success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if success:
        rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, camera_matrix, None, rvec, tvec)
    return rvec, tvec


def compute_pose_errors(true_rvec, true_tvec, est_rvec, est_tvec):
    """计算旋转误差（弧度）和平移误差（欧氏距离）"""
    R_true, _ = cv2.Rodrigues(true_rvec)
    R_est, _ = cv2.Rodrigues(est_rvec)
    R_diff = R_est @ R_true.T
    angle_error = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0))
    translation_error = np.linalg.norm(true_tvec - est_tvec)
    return angle_error, translation_error


def estimate_pose(image_path, template_path):
    # 读取图像
    img = cv2.imread(image_path)
    template = cv2.imread(template_path)
    if img is None or template is None:
        raise ValueError("读取图像失败，请检查路径")

    h, w = img.shape[:2]
    camera_matrix = np.array([
        [1280, 0, w / 2],
        [0, 1080, h / 2],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros(4)

    # 合成点验证PnP
    true_rvec = np.array([0, 0, 0], dtype=np.float32)
    true_tvec = np.array([0, 0, 5], dtype=np.float32)
    synth_points, _ = cv2.projectPoints(TEMPLATE_3D_POINTS, true_rvec, true_tvec, camera_matrix, None)
    synth_points = synth_points.reshape(-1, 2)
    est_rvec_synth, est_tvec_synth = estimate_pose_from_points(synth_points, TEMPLATE_3D_POINTS, camera_matrix)
    print("合成点验证：")
    print("真实旋转:", true_rvec, "\t估计旋转:", est_rvec_synth.ravel())
    print("真实平移:", true_tvec, "\t估计平移:", est_tvec_synth.ravel())

    # 投影3D模板点（用于匹配）
    INIT_RVEC = np.radians([180, 0, 0], dtype=np.float32)
    INIT_TVEC = np.array([[0], [0], [5]], dtype=np.float32)
    template_projected, _ = cv2.projectPoints(TEMPLATE_3D_POINTS, INIT_RVEC, INIT_TVEC, camera_matrix, None)
    template_projected = template_projected.reshape(-1, 2)

    # ORB匹配模板投影点与图像特征点
    template_pts, img_pts = orb_feature_matching(img, template_projected)

    # 筛选匹配点，避免重复对应同一3D点
    match_threshold = 100
    obj_pts = []
    img_pts_filtered = []
    used_3d_indices = set()
    debug_img = img.copy()

    for tp, ip in zip(template_pts, img_pts):
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

    debug_show("匹配过程调试", debug_img)

    if len(obj_pts) < 4:
        raise ValueError(f"有效3D-2D对应不足，仅找到{len(obj_pts)}个 (阈值={match_threshold}px)")

    obj_pts = np.array(obj_pts, dtype=np.float32).reshape(-1, 3)
    img_pts_filtered = np.array(img_pts_filtered, dtype=np.float32).reshape(-1, 1, 2)

    print(f"\n有效对应数量: {len(obj_pts)}")
    for i in range(min(3, len(obj_pts))):
        print(f"3D点 {obj_pts[i]} → 2D点 {img_pts_filtered[i].ravel()}")

    # 使用solvePnPRansac剔除异常点并估计姿态
    _, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=obj_pts,
        imagePoints=img_pts_filtered,
        cameraMatrix=camera_matrix,
        distCoeffs=None,
        reprojectionError=10
    )
    rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts_filtered, camera_matrix, dist_coeffs, rvec, tvec)

    if tvec[2] < 1 or tvec[2] > 20:
        print("警告：异常相机距离，重置平移")
        tvec = np.array([0, 0, 5], dtype=np.float32)

    print("\nPnP解算结果：")
    print("旋转向量(rvec):", rvec.ravel())
    print("平移向量(tvec):", tvec.ravel())

    # 重投影误差统计与可视化
    reprojected_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
    reprojected_pts = reprojected_pts.reshape(-1, 2)
    errors = np.linalg.norm(img_pts_filtered.reshape(-1, 2) - reprojected_pts, axis=1)

    debug_reproj = img.copy()
    for pt, rpt in zip(img_pts_filtered.reshape(-1, 2), reprojected_pts):
        cv2.circle(debug_reproj, tuple(pt.astype(int)), 5, (0, 255, 0), -1)  # 原始点（绿）
        cv2.circle(debug_reproj, tuple(rpt.astype(int)), 3, (0, 0, 255), -1)  # 重投影点（红）
        cv2.line(debug_reproj, tuple(pt.astype(int)), tuple(rpt.astype(int)), (255, 0, 0), 1)
    debug_show("重投影结果", debug_reproj)
    print(f"平均误差: {np.mean(errors):.2f} 像素, 最大误差: {np.max(errors):.2f} 像素")

    # 计算姿态误差（此处用示例真实值）
    true_rvec_actual = np.array([5, 0, 0], dtype=np.float32)
    true_tvec_actual = np.array([0, 0, 5], dtype=np.float32)
    # 绘制坐标轴
    result_img = visualize_pose(img.copy(), rvec, tvec, camera_matrix)
    debug_show("最终姿态可视化", result_img)

    return rvec, tvec

estimate_pose("./datasets/soyuz_easy/0_0_00020.bmp", "./datasets/soyuz_easy/5_0_00035.bmp")
# debug_show("最终姿态估计", cv2.resize(result_img, (2530, 1302)))