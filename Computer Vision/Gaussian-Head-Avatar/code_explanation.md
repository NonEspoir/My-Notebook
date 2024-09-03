## remove_background_nersemble.py

```python
import argparse
import torch
import os
import shutil
import json
import glob
import cv2
import imageio
import numpy as np
from tqdm import tqdm

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms.functional import to_pil_image
from threading import Thread

from dataset import ImagesDataset, ZipDataset
from dataset import augmentation as A
from model import MattingBase, MattingRefine
from inference_utils import HomographicAlignment


def preprocess_nersemble(args, data_folder, camera_ids):
    # 设置运行设备
    device = torch.device(args.device)

    # 根据模型类型选择模型并初始化
    if args.model_type == 'mattingbase':
        model = MattingBase(args.model_backbone)  # 使用 MattingBase 模型
    if args.model_type == 'mattingrefine':
        model = MattingRefine(
            args.model_backbone,
            args.model_backbone_scale,
            args.model_refine_mode,
            args.model_refine_sample_pixels,
            args.model_refine_threshold,
            args.model_refine_kernel_size)  # 使用 MattingRefine 模型

    # 将模型加载到设备上（CPU或GPU），并设置为评估模式
    model = model.to(device).eval()

    # 加载模型权重（从检查点加载）
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=device), strict=False)

    # 获取 images 文件夹中所有帧的文件名，并按顺序排序
    fids = sorted(os.listdir(os.path.join(data_folder, 'images')))

    # 遍历所有指定的相机 ID
    for v in range(len(camera_ids)):
        # 遍历每一帧的文件名
        for fid in tqdm(fids):
            # 构建图像路径和背景图像路径
            image_path = os.path.join(data_folder, 'images', fid, 'image_%s.jpg' % camera_ids[v])
            background_path = os.path.join(data_folder, 'background', 'image_%s.jpg' % camera_ids[v])

            # 如果图像不存在，则跳过
            if not os.path.exists(image_path):
                continue

            # 读取图像并进行归一化处理，将其转换为 PyTorch 张量
            image = imageio.imread(image_path)
            src = (torch.from_numpy(image).float() / 255).permute(2,0,1)[None].to(device, non_blocking=True)

            # 如果背景图像存在，则读取并处理，否则用全零图像代替
            if os.path.exists(background_path):
                background = imageio.imread(background_path)
                bgr = (torch.from_numpy(background).float() / 255).permute(2,0,1)[None].to(device, non_blocking=True)
            else:
                bgr = src * 0.0  # 用全零图像代替背景

            # 禁用梯度计算，进行前向推理
            with torch.no_grad():
                if args.model_type == 'mattingbase':
                    pha, fgr, err, _ = model(src, bgr)  # 对应 MattingBase 的输出
                elif args.model_type == 'mattingrefine':
                    pha, fgr, _, _, err, ref = model(src, bgr)  # 对应 MattingRefine 的输出

            # 生成前景遮罩（mask），将其转换为 uint8 类型
            mask = (pha[0].repeat([3, 1, 1]) * 255).permute(1,2,0).cpu().numpy().astype(np.uint8)
            # 将遮罩图像调整为低分辨率（256x256）
            mask_lowres = cv2.resize(mask, (256, 256))

            # 保存高分辨率的遮罩图像
            mask_path = os.path.join(data_folder, 'images', fid, 'mask_%s.jpg' % camera_ids[v])
            imageio.imsave(mask_path, mask)

            # 保存低分辨率的遮罩图像
            mask_lowres_path = os.path.join(data_folder, 'images', fid, 'mask_lowres_%s.jpg' % camera_ids[v])
            imageio.imsave(mask_lowres_path, mask_lowres)


if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description='Inference images')

    # 添加设备参数（CPU 或 CUDA）
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    # 添加模型类型参数（MattingBase 或 MattingRefine）
    parser.add_argument('--model-type', type=str, default='mattingrefine', choices=['mattingbase', 'mattingrefine'])
    # 添加模型主干网络参数（ResNet 或 MobileNet）
    parser.add_argument('--model-backbone', type=str, default='resnet101', choices=['resnet101', 'resnet50', 'mobilenetv2'])
    # 添加模型主干网络缩放比例参数
    parser.add_argument('--model-backbone-scale', type=float, default=0.25)
    # 添加模型权重文件路径参数
    parser.add_argument('--model-checkpoint', type=str, default='assets/pytorch_resnet101.pth')
    # 添加模型细化模式参数
    parser.add_argument('--model-refine-mode', type=str, default='sampling', choices=['full', 'sampling', 'thresholding'])
    # 添加细化模式下的采样像素数量参数
    parser.add_argument('--model-refine-sample-pixels', type=int, default=80_000)
    # 添加细化模式下的阈值参数
    parser.add_argument('--model-refine-threshold', type=float, default=0.7)
    # 添加细化模式下的卷积核大小参数
    parser.add_argument('--model-refine-kernel-size', type=int, default=3)

    # 解析命令行参数
    args = parser.parse_args()

    # 设置数据源目录和相机 ID 列表
    DATA_SOURCE = '../NeRSemble'
    CAMERA_IDS = ['220700191', '221501007', '222200036', '222200037', '222200038', '222200039', '222200040', '222200041',
                  '222200042', '222200043', '222200044', '222200045', '222200046', '222200047', '222200048', '222200049']

    # 遍历数据源目录中的所有子文件夹，每个子文件夹对应一个数据 ID
    ids = sorted(os.listdir(DATA_SOURCE))
    for id in ids:
        data_folder = os.path.join(DATA_SOURCE, id)  # 构建数据文件夹路径
        preprocess_nersemble(args, data_folder, CAMERA_IDS)  # 调用推理处理函数
```

1. **模块导入**：
   - 引入了多个必要的库和模块，包括处理图像的 `cv2`、`imageio`，处理数据的 `numpy`，以及深度学习相关的 `torch`。

2. **函数 `preprocess_nersemble`**：
   - 主要用于执行推理操作，根据给定的模型和数据对图像生成前景遮罩，并保存结果。

3. **命令行参数解析**：
   - 使用 `argparse` 模块解析命令行输入的参数，允许用户指定模型的类型、使用的设备、模型权重路径等。

4. **遍历数据源和相机 ID**：
   - 脚本会遍历指定数据源目录下的所有文件夹，并根据相机 ID 列表对每个相机的图像进行推理。

5. **推理过程**：
   - 对图像进行预处理、加载背景图像、进行模型推理，并保存高分辨率和低分辨率的遮罩图像。

6. **脚本入口**：
   - 解析命令行参数后，脚本开始遍历数据源目录，并调用 `preprocess_nersemble` 函数进行图像处理。





```python
            # 检查背景图像路径是否存在
            if os.path.exists(background_path):
                # 如果存在，则读取背景图像
                background = imageio.imread(background_path)
                
                # 将背景图像转换为 PyTorch 张量
                # 1. 将图像从 NumPy 数组转换为浮点数张量，并将像素值归一化到 [0, 1] 范围内（除以 255）。
                # 2. 调整张量的维度顺序：从 (height, width, channels) 转换为 (channels, height, width)。
                # 3. 增加一个批量维度（None 相当于在最前面添加一个维度），使张量的形状变为 (1, channels, height, width)。
                # 4. 将张量移动到指定的设备上（CPU 或 GPU），并启用非阻塞模式，以提高数据加载效率。
                bgr = (torch.from_numpy(background).float() / 255).permute(2,0,1)[None].to(device, non_blocking=True)
```

详细说明：

1. **`os.path.exists(background_path)`**:
   - 该行代码检查指定的 `background_path` 是否存在，即是否有对应的背景图像文件。

2. **`background = imageio.imread(background_path)`**:
   - 如果背景图像文件存在，使用 `imageio.imread` 函数读取该图像，并将其存储为一个 NumPy 数组 `background`。

3. **`torch.from_numpy(background)`**:
   - 将 NumPy 数组 `background` 转换为 PyTorch 张量。这一步将图像数据从 NumPy 格式转换为 PyTorch 格式，以便后续在神经网络中使用。

4. **`float() / 255`**:
   - 将图像像素值从整数范围 [0, 255] 归一化为浮点数范围 [0, 1]。这是深度学习模型常见的预处理步骤，使得输入张量的值更适合神经网络的计算。

5. **`permute(2,0,1)`**:
   - 调整张量的维度顺序：原始图像的形状为 `(height, width, channels)`，但 PyTorch 通常使用 `(channels, height, width)` 作为张量的默认顺序，因此需要进行维度排列。

6. **`[None]`**:
   - 在张量的最前面增加一个新的维度，表示批量大小（batch size）。因为神经网络通常处理批量数据，这里增加一个维度将单个图像视为一个批次。

7. **`to(device, non_blocking=True)`**:
   - 将张量移动到指定的设备上（例如 `cuda` 表示 GPU，`cpu` 表示 CPU）。`non_blocking=True` 选项允许在数据传输过程中不阻塞计算，从而提高效率。

通过这些步骤，背景图像被处理为适合模型输入的张量，并准备与前景图像一起进行推理。如果背景图像不存在，代码则跳过这部分处理。



### `permute(2,0,1)` 

`permute(2,0,1)` 是 PyTorch 中用于重新排列张量维度的方法。它允许你根据指定的顺序对张量的维度进行重新排列。具体来说，这个操作将改变张量的维度顺序。

具体示例与解释：

假设我们有一个图像张量 `image`，它的形状是 `(height, width, channels)`，即 `(H, W, C)`。这个形状通常是从 NumPy 数组或图像文件读取时的默认格式。

```python
# 原始形状为 (H, W, C)
image.shape  # 输出: torch.Size([H, W, C])
```

`permute(2, 0, 1)` 的作用是将这个张量的维度重新排列为 `(channels, height, width)`，即 `(C, H, W)`。

```python
# 使用 permute(2, 0, 1) 将维度顺序调整为 (C, H, W)
image_permuted = image.permute(2, 0, 1)
image_permuted.shape  # 输出: torch.Size([C, H, W])
```

**维度解释：**

- **`0`**：表示原始张量的第一个维度（在这里是 `height`，即 `H`）。
- **`1`**：表示原始张量的第二个维度（在这里是 `width`，即 `W`）。
- **`2`**：表示原始张量的第三个维度（在这里是 `channels`，即 `C`）。

在 `permute(2, 0, 1)` 中：
- `2` 表示将原始的第三个维度（`channels`）放在新的第一个维度位置。
- `0` 表示将原始的第一个维度（`height`）放在新的第二个维度位置。
- `1` 表示将原始的第二个维度（`width`）放在新的第三个维度位置。

**为什么要使用 `permute`？**

深度学习库（如 PyTorch）通常要求图像数据的格式为 `(C, H, W)`，即 `(channels, height, width)`。然而，许多图像处理库（如 OpenCV 或 imageio）读取的图像数据格式是 `(H, W, C)`。因此，使用 `permute` 可以将数据转换为 PyTorch 所期望的格式，以便正确地传递到模型中。

**总结：**

`permute(2, 0, 1)` 是一种常见操作，特别是在处理图像数据时，用于将图像的维度顺序从 `(height, width, channels)` 转换为 `(channels, height, width)`。