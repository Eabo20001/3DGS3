import torch
import numpy as np

def save_ply(points, filename):
    """
    将形状为[N, 3]的PyTorch张量保存为PLY格式的点云文件。

    参数:
        points (torch.Tensor): 输入点云，形状为[N, 3]。
        filename (str): 输出的PLY文件名。
    
    异常:
        ValueError: 如果输入点的形状不符合[N, 3]。
    """
    # 检查输入形状
    if points.dim() != 2 or points.size(1) != 3:
        raise ValueError("输入点云的形状必须为[N, 3]，但得到的是 {}".format(points.shape))
    
    # 转换为NumPy数组（自动处理设备位置）
    points_np = points.detach().cpu().numpy()
    
    # 构建PLY头部信息
    header = [
        'ply',
        'format ascii 1.0',
        'element vertex {}'.format(len(points_np)),
        'property float x',
        'property float y',
        'property float z',
        'end_header'
    ]
    header = '\n'.join(header)
    
    # 使用NumPy保存数据，包括头部
    np.savetxt(
        filename,
        points_np,
        fmt='%.6f',        # 保留6位小数以确保精度
        header=header,
        comments='',      # 避免在头部前添加注释符号
        encoding='ascii'
    )