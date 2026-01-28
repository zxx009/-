"""Class to handle Custom datasets for satellite components recognition"""
import os
import numpy as np
import pandas as pd
import skimage
import skimage.transform
import se3lib
import utils
from dataset import Dataset

class Camera:
    """定义相机参数"""
    # width = 1024  # 图像宽度
    # height = 1024  # 图像高度
    width = 1280  # 图像宽度
    height = 960  # 图像高度
    fx = 1250     # 焦距x
    fy = 1250     # 焦距y

    K = np.matrix([[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1]])

# 计算的数据集图像均值（RGB）
MEAN_PIXEL = np.array([25.88, 25.88, 25.88])

class CustomDataset(Dataset):
    """自定义数据集类，继承自Dataset基类"""
    def load_dataset(self, dataset_dir, config, subset):
        """加载数据集的子集
        dataset_dir: 数据集根目录
        config: 模型配置对象
        subset: 要加载的子集类型（train, val, test）
        """
        self.name = 'CustomDataset'
        self.config = config

        # 检查数据集目录是否存在
        if not os.path.exists(dataset_dir):
            print("Image directory '" + dataset_dir + "' not found.")
            return None

        # 加载图像列表
        set_filename = os.path.join(dataset_dir, subset + '_images.csv')
        print("set_filename:", set_filename)
        
        try:
            rgb_list_df = pd.read_csv(set_filename, names=['filename'])
            rgb_list = list(rgb_list_df['filename'])
        except Exception as e:
            print("Error reading image list:", e)
            return None

        # 设置相机参数
        self.camera = Camera()

        # 加载姿态标签
        print('Loading poses')
        poses_filename = os.path.join(dataset_dir, subset + '_poses_gt.csv')
        
        try:
            poses = pd.read_csv(poses_filename)
        except Exception as e:
            print("Error reading pose labels:", e)
            return None

        # 获取实例数量
        nr_instances = len(rgb_list)

        # 初始化四元数和位置数组
        q_array = np.zeros((nr_instances, 4), dtype=np.float32)
        t_array = np.zeros((nr_instances, 3), dtype=np.float32)
        
        # 确保四元数在北半球表示（对回归有帮助）
        for i in range(nr_instances):
            if i >= len(poses):
                print("Warning: Insufficient pose data, missing label for sample", i)
                continue
            
            # 规范化四元数表示（确保w分量为正）
            if poses['q4'][i] < 0:
                q_array[i, :] = np.asarray([-poses['q1'][i], -poses['q2'][i], -poses['q3'][i], -poses['q4'][i]])
            else:
                q_array[i, :] = np.asarray([poses['q1'][i], poses['q2'][i], poses['q3'][i], poses['q4'][i]])
            
            # 存储位置信息
            t_array[i, :] = [poses['x'][i], poses['y'][i], poses['z'][i]]

        # 使用软分配编码方向
        if not config.REGRESS_ORI:
            print('Encoding orientations using soft assignment..')
            
            try:
                ori_encoded, ori_histogram_map, ori_output_mask = utils.encode_ori(
                    q_array, config.ORI_BINS_PER_DIM, config.BETA,
                    np.array([-180, -90, -180]), np.array([180, 90, 180])
                )
                self.ori_histogram_map = ori_histogram_map
                self.ori_output_mask = ori_output_mask
            except Exception as e:
                print("Error encoding orientation:", e)

        # 使用软分配编码位置
        if not config.REGRESS_LOC:
            print('Encoding locations using soft assignment..')
            
            try:
                # 计算图像平面上的位置坐标（转换到相机坐标系）
                img_x_array = poses['y'] / poses['x']  # 同时转换到相机参考系
                img_y_array = poses['z'] / poses['x']  # 同时转换到相机参考系
                z_array = poses['x']
                
                # 计算基于相机视场和数据集范围的位置限制
                # 使用默认值，因为我们的相机类没有fov_x和fov_y属性
                theta_x = np.deg2rad(45)  # 默认45度视野
                theta_y = np.deg2rad(45)  # 默认45度视野
                x_max = np.tan(theta_x)
                y_max = np.tan(theta_y)
                z_min = min(z_array) if len(z_array) > 0 else 0
                z_max = max(z_array) if len(z_array) > 0 else 10
                
                # 编码位置信息
                loc_encoded, loc_histogram_map = utils.encode_loc(
                    np.stack((img_x_array, img_y_array, z_array), axis=1),
                    config.LOC_BINS_PER_DIM, config.BETA,
                    np.array([-x_max, -y_max, z_min]), np.array([x_max, y_max, z_max])
                )
                
                # 存储直方图的物理结构以便后续推理
                self.histogram_3D_map = loc_histogram_map
            except Exception as e:
                print("Error encoding location:", e)

        if not rgb_list:
            print('No files found')
            return None

        # 编码关键点
        try:
            K1, K2 = utils.encode_as_keypoints(q_array, t_array, 3.0)
        except Exception as e:
            print("Error encoding keypoints:", e)
            # 使用默认关键点
            K1 = np.zeros((nr_instances, 3))
            K2 = np.zeros((nr_instances, 3))

        # 添加图像信息
        i = 0
        for file_name in rgb_list:
            if i >= len(poses):
                print("Warning: Reached pose data limit, skipping remaining images")
                break
            
            q = q_array[i, :]
            
            # 转换为角轴表示
            try:
                v, theta = se3lib.quat2angleaxis(q)
                # 转换为欧拉角
                pyr = np.asarray(se3lib.quat2euler(q))
            except Exception as e:
                print("Error converting pose representation:", e)
                v, theta = np.zeros(3), 0
                pyr = np.zeros(3)
            
            # 获取编码的方向和位置
            if config.REGRESS_ORI:
                ori_encoded_i = []
            else:
                ori_encoded_i = ori_encoded[i, :] if 'ori_encoded' in locals() else []
            
            if config.REGRESS_LOC:
                loc_encoded_i = []
            else:
                loc_encoded_i = loc_encoded[i, :] if 'loc_encoded' in locals() else []
            
            # 构建完整的图像路径
            rgb_path = os.path.join(dataset_dir, file_name)
            
            # 检查图像文件是否存在
            if not os.path.exists(rgb_path):
                print("Warning: Image file not found:", rgb_path)
                i += 1
                continue
            
            # 添加图像信息到数据集
            self.add_image(
                "CustomDataset",
                image_id=i,
                path=rgb_path,
                keypoints=[K1[i, :], K2[i, :]],
                location=[poses['x'][i], poses['y'][i], poses['z'][i]],
                location_map=loc_encoded_i,
                quaternion=q,
                angleaxis=[v[0] * theta, v[1] * theta, v[2] * theta],
                pyr=pyr,
                ori_map=ori_encoded_i
            )
            i += 1

        # 设置图像ID和数量
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)
        print(f"Successfully loaded {self.num_images} images")
        
        return self
    
    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array."""
        # 加载图像
        try:
            image = skimage.io.imread(self.image_info[image_id]['path'])
        except Exception as e:
            print("Error loading image (ID:", image_id, "):", e)
            # 返回空白图像
            return np.zeros((self.camera.height, self.camera.width, 3), dtype=np.uint8)
        
        # 如果是灰度图，转换为RGB以保持一致性
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        
        # 如果有alpha通道，移除它以保持一致性
        if image.shape[-1] == 4:
            image = image[..., :3]
        
        
        return image
    
    def image_reference(self, image_id):
        """Return a reference for the specified image."""
        info = self.image_info[image_id]
        if info['source'] == "CustomDataset":
            return info['path']
        else:
            super().image_reference(image_id)
    
    # 新加入：根据路径加载图像
    def load_image_from_path(self, image_path):
        import cv2
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    # 新加入：根据路径获取 image_id
    
    def get_image_id_from_path(self, image_path):
    # """
    # 给定图像文件路径，返回对应的 image_id。
    # 如果找不到，则返回 None。
    # """
        # 统一路径格式（去掉大小写和分隔符差异）
        normalized_path = os.path.normpath(image_path).lower()

        for i, info in enumerate(self.image_info):
            if os.path.normpath(info["path"]).lower() == normalized_path:
                return i
        return None