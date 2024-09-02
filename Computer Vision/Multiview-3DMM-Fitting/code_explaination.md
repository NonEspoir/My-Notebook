

## 配置文件参数：

### 1. `image_folder`
- **解释**: 指定输入图像的文件夹路径。
- **用途**: 该路径下存储的是不同视角拍摄的人脸图像，用于3D人脸建模或拟合的输入数据。
- **示例值**: `/root/Multiview-3DMM-Fitting/NeRSemble_New/031/images`
- **作用**: 在处理过程中，代码会从这个文件夹中读取图像文件，进行关键点检测、3D建模等操作。

### 2. `camera_folder`
- **解释**: 指定与输入图像对应的相机参数文件夹路径。
- **用途**: 该路径下存储了每个视角的相机内参和外参，用于将2D图像数据与3D世界坐标系关联起来。
- **示例值**: `/root/Multiview-3DMM-Fitting/NeRSemble_New/031/cameras`
- **作用**: 代码会从这个文件夹中读取相机参数，用于计算3D模型的投影和图像的重投影误差。

### 3. `landmark_folder`
- **解释**: 指定输出人脸关键点的文件夹路径。
- **用途**: 该路径用于存储每张图像对应的3D人脸关键点数据，通常保存为 `.npy` 文件。
- **示例值**: `/root/Multiview-3DMM-Fitting/NeRSemble_New/031/landmarks`
- **作用**: 在关键点检测过程完成后，代码会将结果保存到这个文件夹，以便后续处理或分析。

### 4. `param_folder`
- **解释**: 指定与每个帧关联的参数文件夹路径。
- **用途**: 该路径下存储的参数文件包括3D人脸模型的姿态、缩放、表情系数等信息，通常用于3D重建或拟合。
- **示例值**: `/root/Multiview-3DMM-Fitting/NeRSemble_New/031/params`
- **作用**: 代码会从这个文件夹中读取模型参数，以便在处理图像时使用或进行3D模型的拟合。

### 5. `gpu_id`
- **解释**: 指定使用的GPU设备ID。
- **用途**: 决定代码运行时使用哪块GPU进行计算。
- **示例值**: `0`
- **作用**: 在多GPU环境中，这个参数可以确保程序在指定的GPU上运行，从而有效利用计算资源。

### 6. `camera_ids`
- **解释**: 包含用于拍摄图像的相机ID列表。
- **用途**: 列表中的每个ID对应一个相机，用于标识不同视角拍摄的图像。
- **示例值**: `['220700191', '221501007', '222200036', ...]`
- **作用**: 代码会根据这些相机ID来查找并处理对应视角的图像和相机参数。

### 7. `image_size`
- **解释**: 指定图像的处理尺寸（宽度和高度）。
- **用途**: 决定在处理过程中，图像被调整到的分辨率。
- **示例值**: `720`
- **作用**: 图像会被调整到 720x720 的分辨率，这可以在保证处理速度的同时，保留足够的细节用于3D建模。

### 8. `face_model`
- **解释**: 指定使用的人脸3D模型类型。
- **用途**: 决定使用哪种3D人脸模型进行拟合或重建。
- **示例值**: `'FLAME'`
- **作用**: 该参数指定了使用 FLAME 模型，这是一种常用的3D人脸模型，具有较高的灵活性和表达能力。

### 9. `reg_id_weight`
- **解释**: 身份正则化项的权重。
- **用途**: 控制身份正则化项在损失函数中的影响大小，通常用于保持个体的身份特征。
- **示例值**: `1e-7`
- **作用**: 较小的权重表示身份正则化对整体损失的影响较小，这有助于在优化过程中更多地关注其他目标。

### 10. `reg_exp_weight`
- **解释**: 表情正则化项的权重。
- **用途**: 控制表情正则化项在损失函数中的影响大小，通常用于限制表情变形的范围。
- **示例值**: `1e-8`
- **作用**: 类似于身份正则化，这个权重帮助平衡模型在表达能力和保持真实表情之间的权衡。

### 11. `save_vertices`
- **解释**: 是否保存3D人脸模型的顶点数据。
- **用途**: 决定是否在处理完成后，保存3D模型的顶点信息。
- **示例值**: `False`
- **作用**: 如果设置为 `True`，程序将保存生成的3D人脸模型的顶点信息；如果为 `False`，则不保存，节省存储空间。

### 12. `visualize`
- **解释**: 是否启用可视化。
- **用途**: 决定是否在处理过程中或处理完成后，进行结果的可视化。
- **示例值**: `True`
- **作用**: 如果设置为 `True`，程序可能会显示或保存中间处理结果的图像或模型，用于调试或结果展示。



## `detect_landmarks.py`

```python
import os
import torch
import tqdm
import glob
import numpy as np
import cv2
import face_alignment
import argparse

# 导入配置文件模块
from config.config import config

if __name__ == '__main__':
    # 设置命令行参数解析器，并指定配置文件路径
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/NeRSemble_031.yaml')
    arg = parser.parse_args()

    # 加载配置文件
    cfg = config()
    cfg.load(arg.config)
    cfg = cfg.get_cfg()

    # 设置GPU设备
    device = torch.device('cuda:%d' % cfg.gpu_id)
    torch.cuda.set_device(cfg.gpu_id)

    # 初始化FaceAlignment模型，用于检测3D人脸关键点
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, 
                                      flip_input=False, 
                                      face_detector='blazeface', 
                                      device='cuda:%d' % cfg.gpu_id)

    # 设置输入图像文件夹和输出关键点文件夹
    source_folder = cfg.image_folder
    output_folder = cfg.landmark_folder

    # 获取输入图像文件夹中的所有帧
    frames = sorted(os.listdir(source_folder))
    for frame in tqdm.tqdm(frames):
        # 跳过包含 'background' 字样的帧
        if 'background' in frame:
            continue
        
        # 设置当前帧的输入和输出文件夹路径
        source_frame_folder = os.path.join(source_folder, frame)
        output_frame_folder = os.path.join(output_folder, frame)
        os.makedirs(output_frame_folder, exist_ok=True)  # 创建输出文件夹，如果不存在的话

        # 根据配置文件中的相机ID加载图像路径
        if len(cfg.camera_ids) > 0:
            image_paths = [source_frame_folder + '/image_%s.jpg' % camera_id for camera_id in cfg.camera_ids]
        else:
            image_paths = sorted(glob.glob(source_frame_folder + '/image_*.jpg'))

        # 读取图像，并调整大小到指定尺寸
        images = np.stack([cv2.resize(cv2.imread(image_path)[:, :, ::-1], (cfg.image_size, cfg.image_size)) for image_path in image_paths])
        images = torch.from_numpy(images).float().permute(0, 3, 1, 2).to(device)  # 将图像转换为PyTorch张量并转移到GPU

        # 使用FaceAlignment模型批量获取图像中的3D人脸关键点
        results = fa.get_landmarks_from_batch(images, return_landmark_score=True)

        # 处理每张图像的结果
        for i in range(len(results[0])):
            # 如果没有检测到人脸，则用全零填充关键点和分数
            if results[1][i] is None:
                results[0][i] = np.zeros([68, 3], dtype=np.float32)  # 68个关键点，每个关键点3D坐标
                results[1][i] = [np.zeros([68], dtype=np.float32)]  # 68个关键点的分数
            # 如果检测到多个人脸，选择得分最高的那个
            if len(results[1][i]) > 1:
                total_score = 0.0
                for j in range(len(results[1][i])):
                    if np.sum(results[1][i][j]) > total_score:
                        total_score = np.sum(results[1][i][j])
                        landmarks_i = results[0][i][j*68:(j+1)*68]  # 选择得分最高的人脸的关键点
                        scores_i = results[1][i][j:j+1]  # 对应的分数
                results[0][i] = landmarks_i  # 更新结果为得分最高的关键点
                results[1][i] = scores_i  # 更新结果为得分最高的分数
                
        # 将关键点的2D坐标与得分结合，形成最终输出
        landmarks = np.concatenate([np.stack(results[0])[:, :, :2], np.stack(results[1]).transpose(0, 2, 1)], -1)
        
        # 保存每张图像的关键点数据
        for i, image_path in enumerate(image_paths):
            landmarks_path = os.path.join(output_frame_folder, image_path.split('/')[-1].replace('image_', 'lmk_').replace('.jpg', '.npy'))
            np.save(landmarks_path, landmarks[i])  # 保存关键点为.npy文件

```

这段代码的主要功能是**从一组图像中检测3D人脸关键点，并将这些关键点保存为 `.npy` 文件**。具体来说，它完成了以下任务：

1. **读取配置文件**：从配置文件中读取各种参数，如输入图像文件夹路径、输出关键点文件夹路径、相机ID、图像尺寸、GPU设备ID等。
2. **初始化模型**：使用 `face_alignment` 库中的 `FaceAlignment` 模型来检测图像中的3D人脸关键点。这个模型能够检测68个3D关键点，并返回它们的坐标。
3. **处理多视角图像**：
   - 对每个图像帧，代码会加载与之相关联的多视角图像。如果配置中提供了相机ID，则只加载这些ID对应的图像；否则，加载所有视角的图像。
   - 图像被调整为指定的尺寸，并被转换为PyTorch张量以便在GPU上进行处理。
4. **人脸关键点检测**：
   - 使用 `face_alignment` 模型对每张图像进行人脸检测，并获取68个3D关键点的坐标及其对应的检测得分。
   - 如果检测到多个脸部，代码会选择得分最高的那个脸部的关键点。如果未检测到脸部，代码会用全零数组来代替关键点数据。
5. **保存关键点数据**：
   - 检测到的3D关键点数据会被转换为2D坐标加得分的形式，并保存为 `.npy` 文件。
   - 每个视角的关键点数据文件的保存路径与输入图像的路径结构一致。

在这段代码中，68个关键点的数量是由 `face_alignment` 库的 `FaceAlignment` 模型指定的。这个库使用了一些预训练的人脸检测和关键点检测模型，这些模型通常基于68个特定的面部关键点。

### 68个关键点的来源

- **标准化面部关键点标注**：68个面部关键点是一个常用的标准，用于描述人脸的关键部位。这些关键点通常包括眼睛、眉毛、鼻子、嘴巴和面部轮廓的特定点。这个标准最早来自于 300-W 人脸数据库的标注格式，它已经被广泛用于各种人脸识别和表情分析任务中。
  
- **`face_alignment` 库的模型**：`face_alignment` 库使用了预训练的模型，这些模型被设计为输出68个关键点。具体来说，`face_alignment.FaceAlignment` 类的默认设置是使用68个点来描述人脸的三维几何结构。这些点的具体位置通常如下：
  - **17个点**：用于描述面部轮廓。
  - **5个点**：用于左眉毛。
  - **5个点**：用于右眉毛。
  - **9个点**：用于鼻子。
  - **7个点**：用于左眼。
  - **7个点**：用于右眼。
  - **12个点**：用于嘴巴外轮廓。
  - **8个点**：用于嘴巴内轮廓。



在代码中，使用了以下代码初始化 `face_alignment` 模型：

```python
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, 
                                  flip_input=False, 
                                  face_detector='blazeface', 
                                  device='cuda:%d' % cfg.gpu_id)
```

- **`face_alignment.FaceAlignment`**：这是一个类，用于初始化人脸关键点检测模型。
- **`LandmarksType.THREE_D`**：指定检测 3D 关键点（即每个关键点有3个坐标：x, y, z）。这也是指 68 个 3D 关键点的配置。

68个面部关键点的数量由 `face_alignment` 库内部的预训练模型决定，该模型专门设计为检测这68个点。这些点代表了面部的重要特征，广泛应用于人脸分析、识别、表情分析等领域。用户通常不需要在代码中手动指定这个数量，它是由模型的架构和训练数据决定的。

### 最后的输出

**最终的输出是保存到磁盘中的 `.npy` 文件，每个文件包含一张图像中的3D人脸关键点数据**。具体而言：

- **每个 `.npy` 文件**：存储的是对应图像的3D人脸关键点数据。这些数据包括68个3D关键点的2D坐标和它们的检测得分。文件名与输入图像的文件名相关联，如 `image_001.jpg` 对应的关键点文件会命名为 `lmk_001.npy`。
- **保存位置**：关键点文件被保存到指定的输出文件夹中，路径结构与输入图像的文件夹结构相对应。

这些 `.npy` 文件可以用于多种计算机视觉任务中，例如：
- **3D人脸重建**：利用这些关键点重建3D人脸模型。
- **表情识别和分析**：通过分析不同时间点或不同视角下的关键点位置变化来识别和分析表情。
- **姿态估计**：利用关键点的空间分布来估计人脸的旋转和平移姿态。
- **人脸对齐**：将不同视角或不同时间的图像中的人脸对齐，用于进一步的图像处理或模型训练。

这些数据通常是3D人脸分析任务中的重要输入。



## 66个关键点`LandmarkDataset.py`

在这段代码中，确实使用了66个面部关键点，而不是通常的68个。这个变化是通过有意地移除两个关键点实现的。以下是详细的解释：

### 代码分析

```python
landmark = np.load(os.path.join(self.landmark_folder, frame, 'lmk_%s.npy' % camera_ids[v]))
landmark = np.vstack([landmark[0:48], landmark[49:54], landmark[55:68]])
```

- **`np.load`**：首先从 `.npy` 文件中加载面部关键点数据，原始数据包含68个关键点，每个关键点有3个坐标值（x, y, z）。
  
- **`np.vstack([landmark[0:48], landmark[49:54], landmark[55:68]])`**：
  - 这行代码的作用是通过垂直堆叠的方式重新组织关键点，**故意跳过了索引为48和54的两个关键点**。
  - 具体来说：
    - **`landmark[0:48]`**：选择前48个关键点。
    - **`landmark[49:54]`**：跳过第49个（索引为48的点），选择第50到54个关键点。
    - **`landmark[55:68]`**：跳过第55个（索引为54的点），选择第56到68个关键点。
  - 最终的 `landmark` 变量只包含66个关键点，而不是原始的68个。

### 移除的关键点

- 移除的是**索引为48和54的两个关键点**。
- 这些关键点在面部标准化标注中通常位于嘴唇的内部：
  - **索引48的点**：通常位于嘴巴外部下唇与脸部皮肤的连接处。
  - **索引54的点**：通常位于嘴巴内部或下唇与下巴之间的位置。

### 为什么要移除这两个关键点？

有几种可能的原因，为什么在项目中选择移除这两个关键点：

1. **特定任务需求**：
   - 某些应用（如3D面部重建、表情捕捉等）可能不需要这些点，尤其是涉及嘴唇内部的点。移除这些点可以简化模型，并减少噪声。

2. **提高鲁棒性**：
   - 在一些情况下，嘴唇区域的关键点容易受到表情变化的影响（如张嘴、笑等）。移除这些点可以使得剩下的关键点更加稳定。

3. **简化计算**：
   - 通过减少关键点的数量，可以稍微降低计算复杂性，尤其是在需要处理大量数据或进行实时计算的场景中。

4. **数据不一致性**：
   - 有时，不同数据集或标注工具可能在某些点的标注上不一致。如果这些点在实际应用中不重要或不稳定，可以选择移除以提高数据的一致性。

### 最终输出

这段代码最终返回了一个包含所有帧的关键点、相机内参和外参的数组。由于每一帧的关键点数都被固定为66个，而不是原始的68个，因此输出的 `landmarks` 数组中，每一帧每个视角的关键点数据都是 `66 x 3` 的矩阵。

- **`landmarks`**：大小为 `[num_frames, num_cameras, 66, 3]`，包含了每个帧、每个视角的66个关键点的3D坐标。
- **`extrinsics` 和 `intrinsics`**：分别包含相机的外参（3x4矩阵）和内参（3x3矩阵），用于3D重建和投影。

### 总结

在这个项目中，将68个关键点减少到66个，通常是为了简化任务或提高模型的鲁棒性，特别是处理嘴唇区域的关键点时，这些点的变化可能对某些应用来说是不稳定或不必要的。通过这种调整，模型可以更专注于面部最重要的特征点。



## `preprocess_nersemble.py`

以下是对该代码的详细注释：

```python
import os
import numpy as np
import cv2
import glob
import json

# 定义一个函数，用于裁剪图像和调整相机内参矩阵K
def CropImage(left_up, crop_size, image=None, K=None):
    crop_size = np.array(crop_size).astype(np.int32)
    left_up = np.array(left_up).astype(np.int32)

    # 如果提供了相机内参K，则调整其光心位置以反映裁剪操作
    if not K is None:
        K[0:2,2] = K[0:2,2] - np.array(left_up)

    # 如果提供了图像，执行裁剪操作
    if not image is None:
        # 处理裁剪区域在图像外的情况，向图像边界外添加黑色填充
        if left_up[0] < 0:
            image_left = np.zeros([image.shape[0], -left_up[0], image.shape[2]], dtype=np.uint8)
            image = np.hstack([image_left, image])
            left_up[0] = 0
        if left_up[1] < 0:
            image_up = np.zeros([-left_up[1], image.shape[1], image.shape[2]], dtype=np.uint8)
            image = np.vstack([image_up, image])
            left_up[1] = 0
        if crop_size[0] + left_up[0] > image.shape[1]:
            image_right = np.zeros([image.shape[0], crop_size[0] + left_up[0] - image.shape[1], image.shape[2]], dtype=np.uint8)
            image = np.hstack([image, image_right])
        if crop_size[1] + left_up[1] > image.shape[0]:
            image_down = np.zeros([crop_size[1] + left_up[1] - image.shape[0], image.shape[1], image.shape[2]], dtype=np.uint8)
            image = np.vstack([image, image_down])

        # 执行裁剪操作，获取裁剪后的图像
        image = image[left_up[1]:left_up[1]+crop_size[1], left_up[0]:left_up[0]+crop_size[0], :]

    return image, K

# 定义一个函数，用于调整图像大小并调整相机内参矩阵K
def ResizeImage(target_size, source_size, image=None, K=None):
    # 如果提供了相机内参K，更新其以反映图像的缩放
    if not K is None:
        K[0,:] = (target_size[0] / source_size[0]) * K[0,:]
        K[1,:] = (target_size[1] / source_size[1]) * K[1,:]

    # 如果提供了图像，调整其大小
    if not image is None:
        image = cv2.resize(image, dsize=target_size)
    return image, K

# 定义一个函数，用于从视频中提取帧，并保存处理后的图像和相机参数
def extract_frames(id_list):

    for id in id_list:
        # 加载每个id对应的相机参数文件
        camera_path = os.path.join(DATA_SOURCE, 'camera_params', id, 'camera_params.json')
        with open(camera_path, 'r') as f:
            camera = json.load(f)

        fids = {}
        for camera_id in camera['world_2_cam'].keys():
            fids[camera_id] = 0  # 初始化帧ID计数器

            # 加载背景图像，裁剪并调整大小，然后保存
            background_path = os.path.join(DATA_SOURCE, '031_background', id, 'BACKGROUND', 'image_%s.jpg' % camera_id)
            background = cv2.imread(background_path)
            background, _ = CropImage(LEFT_UP, CROP_SIZE, background, None)
            background, _ = ResizeImage(SIZE, CROP_SIZE, background, None)
            os.makedirs(os.path.join(DATA_OUTPUT, id, 'background'), exist_ok=True)
            cv2.imwrite(os.path.join(DATA_OUTPUT, id, 'background', 'image_' + camera_id + '.jpg'), background)
        
        # 处理每个id对应的所有视频文件夹
        video_folders = glob.glob(os.path.join(DATA_SOURCE, '*', id, '*'))
        for video_folder in video_folders:
            # 跳过与舌头、眼镜、自定义自由移动和背景相关的文件夹
            if ('tongue' in video_folder) or ('GLASSES' in video_folder) or ('FREE' in video_folder) or ('BACKGROUND' in video_folder):
                continue
            video_paths = glob.glob(os.path.join(video_folder, 'cam_*'))
            for video_path in video_paths:
                camera_id = video_path[-13:-4]  # 从视频路径提取相机ID
                extrinsic = np.array(camera['world_2_cam'][camera_id][:3])  # 提取相机外参
                intrinsic = np.array(camera['intrinsics'])  # 提取相机内参
                _, intrinsic = CropImage(LEFT_UP, CROP_SIZE, None, intrinsic)
                _, intrinsic = ResizeImage(SIZE, CROP_SIZE, None, intrinsic)
                
                # 打开视频文件并逐帧处理
                cap = cv2.VideoCapture(video_path)
                count = -1
                while(1): 
                    _, image = cap.read()  # 读取一帧图像
                    if image is None:
                        break
                    count += 1
                    if count % 3 != 0:  # 每3帧处理一次，跳过其他帧
                        continue
                    
                    # 生成全白的可见性图像
                    visible = (np.ones_like(image) * 255).astype(np.uint8)
                    
                    # 裁剪并调整帧和可见性图像的大小
                    image, _ = CropImage(LEFT_UP, CROP_SIZE, image, None)
                    image, _ = ResizeImage(SIZE, CROP_SIZE, image, None)
                    visible, _ = CropImage(LEFT_UP, CROP_SIZE, visible, None)
                    visible, _ = ResizeImage(SIZE, CROP_SIZE, visible, None)
                    
                    # 生成低分辨率图像
                    image_lowres = cv2.resize(image, SIZE_LOWRES)
                    visible_lowres = cv2.resize(visible, SIZE_LOWRES)

                    # 创建目录并保存处理后的图像和相机参数
                    os.makedirs(os.path.join(DATA_OUTPUT, id, 'images', '%04d' % fids[camera_id]), exist_ok=True)
                    cv2.imwrite(os.path.join(DATA_OUTPUT, id, 'images', '%04d' % fids[camera_id], 'image_' + camera_id + '.jpg'), image)
                    cv2.imwrite(os.path.join(DATA_OUTPUT, id, 'images', '%04d' % fids[camera_id], 'image_lowres_' + camera_id + '.jpg'), image_lowres)
                    cv2.imwrite(os.path.join(DATA_OUTPUT, id, 'images', '%04d' % fids[camera_id], 'visible_' + camera_id + '.jpg'), visible)
                    cv2.imwrite(os.path.join(DATA_OUTPUT, id, 'images', '%04d' % fids[camera_id], 'visible_lowres_' + camera_id + '.jpg'), visible_lowres)
                    
                    os.makedirs(os.path.join(DATA_OUTPUT, id, 'cameras', '%04d' % fids[camera_id]), exist_ok=True)
                    np.savez(os.path.join(DATA_OUTPUT, id, 'cameras', '%04d' % fids[camera_id], 'camera_' + camera_id + '.npz'), extrinsic=extrinsic, intrinsic=intrinsic)
                    
                    fids[camera_id] += 1  # 增加帧ID计数器

                    
if __name__ == "__main__":
    # 设置图像裁剪和调整大小的参数
    LEFT_UP = [0,504]  # 图像裁剪的左上角坐标
    CROP_SIZE = [2200,2200]  # 裁剪后图像的尺寸
    SIZE = [2200,2200]  # 调整大小后的图像尺寸
    SIZE_LOWRES = [256, 256]  # 低分辨率图像的尺寸
    
    # 设置数据源和输出文件夹
    DATA_SOURCE = "/data/guoyang/data/Nersemble_initial_data/"
    DATA_OUTPUT = '../NeRSemble_New'
    
    # 调用函数处理指定ID的帧
    extract_frames(['031'])
```

### 代码功能概述

这段代码的主要功能是从多个视频中提取图像帧，对图像进行裁剪和调整大小，并保存处理后的图像及其相机参数。具体来说，它完成了以下任务：

1. **图像裁剪与调整大小**：
   - 代码通过 `CropImage` 和 `ResizeImage` 函数裁剪和调整图像的大小。这些操作可能会对图像的原始尺寸和内容进行一些预处理，如裁剪边缘部分或调整到特定的分辨率。

2. **背景图像处理**：
   - 从背景文件夹中读取背景图像，进行

裁剪和调整大小，然后保存为JPEG文件。

3. **视频帧提取**：
   - 从视频中按固定间隔提取帧（每3帧提取一次），并对这些帧进行裁剪、调整大小，然后保存为高分辨率和低分辨率的JPEG文件。
   - 还会生成一个全白的可见性图像，表示所有像素都可见。

4. **相机参数保存**：
   - 提取相机的内参和外参，将它们保存为 `.npz` 文件。

5. **输出结构**：
   - 对于每个处理的ID，代码将图像和相机参数保存到指定的输出目录中，目录结构与处理的ID和帧数相关联。

### 最后的输出

最终，这段代码会生成以下内容并保存到指定的输出文件夹（`DATA_OUTPUT`）中：

1. **处理后的图像**：
   - `image_*.jpg`：裁剪并调整大小后的图像。
   - `image_lowres_*.jpg`：低分辨率版本的图像。
   - `visible_*.jpg`：全白的可见性图像，表示所有像素都可见。
   - `visible_lowres_*.jpg`：低分辨率的可见性图像。

2. **相机参数文件**：
   - `camera_*.npz`：包含相机内参和外参的 `.npz` 文件。

这些文件被保存到指定的文件夹结构中，以便后续的使用或进一步的处理。