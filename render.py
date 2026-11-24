import torch
import os
import json
import numpy as np
import math
from tqdm import tqdm
from os import makedirs
from argparse import ArgumentParser
from PIL import Image
import torchvision
import copy
import sys

# 3DGS 核心模块
from gaussian_renderer import render, GaussianModel
from scene.cameras import Camera
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state
from utils.system_utils import searchForMaxIteration

# ================= 1. 核心工具函数 (保持不变) =================

def pack_float_to_rgba(f_tensor):
    val = torch.clamp(f_tensor, 0.0, 1.0)
    r = torch.floor(val * 255.0)
    rem_g = (val * 255.0 - r) * 255.0
    g = torch.floor(rem_g)
    rem_b = (rem_g - g) * 255.0
    b = torch.floor(rem_b)
    rem_a = (rem_b - b) * 255.0
    a = torch.floor(rem_a)
    return torch.stack([r, g, b, a], dim=-1).byte().cpu().numpy()

def compute_motion_vectors(view_now, view_prev, depth_now):
    H, W = view_now.image_height, view_now.image_width
    device = depth_now.device
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    
    ndc_x = (grid_x.float() / (W - 1)) * 2.0 - 1.0
    ndc_y = (grid_y.float() / (H - 1)) * 2.0 - 1.0 
    
    z_linear = depth_now[0] 

    tanfovx = np.tan(view_now.FoVx * 0.5)
    tanfovy = np.tan(view_now.FoVy * 0.5)
    
    p_view = torch.stack([ndc_x * tanfovx * z_linear, ndc_y * tanfovy * z_linear, z_linear, torch.ones_like(z_linear)], dim=-1)

    c2w = view_now.world_view_transform.inverse()
    p_world = (p_view.reshape(-1, 4) @ c2w).reshape(H, W, 4)
    p_clip_prev = (p_world.reshape(-1, 4) @ view_prev.full_proj_transform).reshape(H, W, 4)
    
    ndc_prev = p_clip_prev[..., :2] / (p_clip_prev[..., 3:4] + 1e-8)
    
    mv_x = (ndc_x - ndc_prev[..., 0]) * 0.5
    mv_y = (ndc_y - ndc_prev[..., 1]) * 0.5
    
    return mv_x, -mv_y

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

# ================= 2. 相机加载与生成逻辑 =================

def load_cameras_from_json(model_path):
    json_path = os.path.join(model_path, "cameras.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Could not find cameras.json at {json_path}")
        
    with open(json_path, 'r') as file:
        cam_json = json.load(file)
    
    camera_list = []
    # 按 ID 排序
    cam_json.sort(key=lambda x: x['id']) 
    
    for cam_data in cam_json:
        Rt = np.zeros((4, 4))
        Rt[-1, -1] = 1.0
        Rt[:3, :3] = np.array(cam_data['rotation'])
        Rt[:3, 3] = np.array(cam_data['position'])
        
        # 转换逻辑
        Rt = np.linalg.inv(Rt)
        R = Rt[:3, :3].transpose()
        T = Rt[:3, 3]
        
        width = cam_data['width']
        height = cam_data['height']
        fx = cam_data['fx']
        fy = cam_data['fy']
        
        fovx = focal2fov(fx, width)
        fovy = focal2fov(fy, height)
        
        cam = Camera(
            colmap_id=cam_data['id'],
            R=R,
            T=T,
            FoVx=fovx,
            FoVy=fovy,
            image_name=cam_data['img_name'],
            uid=cam_data['id'],
            resolution=(width, height),
            data_device="cuda"
        )
        camera_list.append(cam)
        
    return camera_list

def gene_lr_hr_cams_new(raw_cam: Camera, pre_downscale, sr_scale, lr_res=(512, 512)):
    lr_w, lr_h = lr_res
    hr_w, hr_h = int(lr_w * sr_scale), int(lr_h * sr_scale)
    hr_res = (hr_w, hr_h)

    lr_fx = fov2focal(raw_cam.FoVx, lr_w)
    lr_fy = fov2focal(raw_cam.FoVy, lr_h)
    hr_fx = fov2focal(raw_cam.FoVx, hr_w)
    hr_fy = fov2focal(raw_cam.FoVy, hr_h)

    lr_f = (lr_fx + lr_fy) / 2 
    hr_f = (hr_fx + hr_fy) / 2
    
    lr_fovx = focal2fov(lr_f, lr_w)
    lr_fovy = focal2fov(lr_f, lr_h)
    hr_fovx = focal2fov(hr_f, hr_w)
    hr_fovy = focal2fov(hr_f, hr_h)
    
    lrcam = Camera(
        resolution=lr_res,
        colmap_id=raw_cam.colmap_id,
        R=raw_cam.R,
        T=raw_cam.T,
        FoVx=lr_fovx,
        FoVy=lr_fovy,
        image_name=f"{raw_cam.image_name}_LR",
        uid=raw_cam.uid,
        data_device="cuda"
    )
    
    hrcam = Camera(
        resolution=hr_res,
        colmap_id=raw_cam.colmap_id,
        R=raw_cam.R,
        T=raw_cam.T,
        FoVx=hr_fovx,
        FoVy=hr_fovy,
        image_name=f"{raw_cam.image_name}_HR",
        uid=raw_cam.uid,
        data_device="cuda"
    )
    return lrcam, hrcam 

def generate_xshift_cams(demo_cam, total_distance=0.15, steps=5):
    step_size = total_distance / steps
    
    beg_w2c = np.eye(4, 4)
    beg_w2c[:3, :3] = demo_cam.R
    beg_w2c[:3, 3] = demo_cam.T
    
    beg_c2w = np.linalg.inv(beg_w2c)
    
    current_pos = beg_c2w[:3, 3]
    current_x_dir = beg_c2w[:3, 0] 
    
    cams = [] 
    
    for i in range(steps):
        new_pos = current_pos + current_x_dir * (step_size * i)
        
        now_c2w = beg_c2w.copy()
        now_c2w[:3, 3] = new_pos
        
        w2c44 = np.linalg.inv(now_c2w)
        
        cams.append(
            Camera(
                colmap_id=demo_cam.colmap_id,
                R=w2c44[:3, :3],
                T=w2c44[:3, 3],
                FoVx=demo_cam.FoVx,
                FoVy=demo_cam.FoVy,
                image_name=f"{demo_cam.image_name}_shift_{i}",
                uid=demo_cam.uid * 100 + i, 
                resolution=(demo_cam.image_width, demo_cam.image_height),
                data_device="cuda"
            )
        )
        
    return cams

# ================= 3. 渲染逻辑 =================
def render_sequence_to_folder(cam_list, gaussians, pipeline, background, kernel_size, target_path, resolution):
    width, height = resolution
    subs = ["view", "depth", "motion_vectors_x", "motion_vectors_y"]
    for sub in subs:
        makedirs(os.path.join(target_path, sub), exist_ok=True)

    if not cam_list: return
    z_far = cam_list[0].zfar 
    
    # --- 修改点 1: 维护一个“上一个MobFGSR输入帧”的变量 ---
    # 我们只关心 0, 2, 4... 这些会被 MobFGSR 读取的帧
    # 它们的 MV 必须是相对于上一个“读取帧”的
    last_input_view = None 
    
    for idx, view in enumerate(cam_list):
        view.image_width, view.image_height = width, height
        
        result = render(view, gaussians, pipeline, background, kernel_size=kernel_size, require_depth=True)
        rgb_image = result["render"]
        mask = result["mask"][0]
        mv_depth_smooth = result["expected_depth"] 
        save_depth = result["expected_depth"][0]
        
        img_id = f"{idx:04d}.png" 

        # 保存 RGB 和 Depth (这些不论是谁都一样)
        torchvision.utils.save_image(rgb_image, os.path.join(target_path, "view", img_id))

        depth_final = save_depth / z_far
        depth_final[mask < 0.01] = 1.0 
        depth_packed = pack_float_to_rgba(depth_final)
        Image.fromarray(depth_packed).save(os.path.join(target_path, "depth", img_id))

        # --- 修改点 2: 智能计算 Motion Vector ---
        # 逻辑：
        # 如果是偶数帧 (0, 2, 4...): 这些是 MobFGSR 的输入帧。
        #   -> 必须计算相对于“上一个偶数帧”的 MV (跨度为2)。
        # 如果是奇数帧 (1, 3...): 这些是 GT 帧，MobFGSR 不会读它们的 MV。
        #   -> 我们可以不存，或者为了兼容性存一个相对于上一帧的(跨度为1)。
        
        mv_x, mv_y = torch.zeros((height, width), device='cuda'), torch.zeros((height, width), device='cuda')
        
        # 只有当 idx 是偶数 (0, 2, 4)，且不是第0帧时，我们需要计算“跨越两帧”的 MV
        if idx % 2 == 0:
            if last_input_view is not None:
                # 关键：这里计算的是 Current(idx) 相对于 Last_Input(idx-2) 的 MV
                mv_x, mv_y = compute_motion_vectors(view, last_input_view, mv_depth_smooth)
            
            # 更新 last_input_view 为当前帧，供 idx+2 使用
            last_input_view = copy.deepcopy(view)
        
        # (可选) 如果你想让奇数帧也有 MV 用于调试，可以在这里加 else 逻辑
        # 但对于你的 MobFGSR 评测任务，奇数帧的 MV 是没用的，写全0也没问题
        
        # Mask 过滤 & 保存
        mv_x[mask < 0.01] = 0.0
        mv_y[mask < 0.01] = 0.0
        
        mv_x_packed = pack_float_to_rgba((mv_x + 1.0) * 0.5)
        mv_y_packed = pack_float_to_rgba((mv_y + 1.0) * 0.5)
        
        Image.fromarray(mv_x_packed).save(os.path.join(target_path, "motion_vectors_x", img_id))
        Image.fromarray(mv_y_packed).save(os.path.join(target_path, "motion_vectors_y", img_id))

# ================= 4. 单个场景处理函数 =================

def process_single_scene(dataset_path, args, lp, pp, cam_cut):
    """
    处理单个数据集的核心逻辑
    """
    scene_name = os.path.basename(dataset_path)
    print(f"\n[{scene_name}] Starting processing...")

    # 临时修改 args 中的 model_path，以便 ModelParams 正确提取参数
    # 注意：这里需要深拷贝一份 args，以免影响外层循环（虽然后面会覆盖，但保险起见）
    current_args = copy.deepcopy(args)
    current_args.model_path = dataset_path

    # 1. 初始化模型
    safe_state(False)
    # 必须在这里实例化 GaussianModel，确保显存清理和状态重置
    gaussians = GaussianModel(lp.extract(current_args).sh_degree)
    
    # 寻找 checkpoint
    try:
        if current_args.iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(dataset_path, "point_cloud"))
        else:
            loaded_iter = current_args.iteration
        
        print(f"[{scene_name}] Loading point cloud iteration: {loaded_iter}")
        gaussians.load_ply(os.path.join(dataset_path,
                                        "point_cloud",
                                        "iteration_" + str(loaded_iter),
                                        "point_cloud.ply"))
    except Exception as e:
        print(f"[ERROR] Failed to load point cloud for {scene_name}: {e}")
        return # 跳过此场景

    # 2. 加载相机
    try:
        train_cameras = load_cameras_from_json(dataset_path)
        print(f"[{scene_name}] Loaded {len(train_cameras)} cameras.")
    except Exception as e:
        print(f"[ERROR] Failed to load cameras.json for {scene_name}: {e}")
        return

    # 3. 筛选前 cam_cut 个相机
    real_cut = min(cam_cut, len(train_cameras))
    target_cameras = train_cameras[:real_cut]
    
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    k_size = lp.extract(current_args).kernel_size
    
    # 分辨率配置
    LR_RES = (1280, 720) 
    SR_SCALE = 2
    HR_RES = (LR_RES[0] * SR_SCALE, LR_RES[1] * SR_SCALE)
    
    # 输出路径: dataset_path/MobFGSR_Inputs
    input_root = os.path.join(dataset_path, "MobFGSR_Inputs")
    
    # 4. 循环生成并渲染
    with torch.no_grad():
        for i, base_cam in enumerate(tqdm(target_cameras, desc=f"Rendering {scene_name}")):
            # 生成 X 轴平移序列 (5帧)
            sequence_cams = generate_xshift_cams(base_cam, total_distance=0.3, steps=5)
            
            lr_seq = []
            hr_seq = []
            for cam in sequence_cams:
                lrc, hrc = gene_lr_hr_cams_new(cam, 1.0, SR_SCALE, lr_res=LR_RES)
                lr_seq.append(lrc)
                hr_seq.append(hrc)
            
            # 定义当前序列的输出路径: .../dataset/MobFGSR_Inputs_.../0/...
            seq_out_path = os.path.join(input_root, str(i))
            
            # 渲染 HR
            render_sequence_to_folder(
                hr_seq, gaussians, pp.extract(current_args), background, k_size,
                os.path.join(seq_out_path, "HR_Inputs"), HR_RES
            )
            
            # 渲染 LR
            render_sequence_to_folder(
                lr_seq, gaussians, pp.extract(current_args), background, k_size,
                os.path.join(seq_out_path, "LR_Inputs"), LR_RES
            )

    # 主动清理显存，防止多场景累积导致 OOM
    del gaussians
    torch.cuda.empty_cache()
    print(f"[{scene_name}] Done! Saved to {input_root}")

# ================= 5. 主程序入口 =================

if __name__ == "__main__":
    parser = ArgumentParser(description="Batch MobFGSR Pipeline")
    
    # 标准参数
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--cam_cut", default=20, type=int, help="Number of base cameras to use per scene") 
    
    # --- 修改开始：增加容错逻辑 ---
    try:
        # 尝试标准加载（如果路径正好是一个具体的模型文件夹，这里会成功）
        args = get_combined_args(parser)
    except FileNotFoundError:
        # 如果找不到 cfg_args（说明路径可能是一个包含多个数据集的根目录）
        # 我们就退回到普通的参数解析，跳过读取 cfg_args
        print("Warning: 'cfg_args' not found in root path. Assuming batch mode and using CLI arguments/defaults.")
        args = parser.parse_args()
    except Exception as e:
        # 其他未知错误
        print(f"Error parsing arguments: {e}")
        args = parser.parse_args()
    # --- 修改结束 ---

    root_path = args.model_path
    
    # 1. 扫描所有子目录
    if not os.path.exists(root_path):
        print(f"Error: Path {root_path} does not exist.")
        sys.exit(1)

    # 获取所有可能的子文件夹
    subfolders = [os.path.join(root_path, f) for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
    
    # 过滤：只保留包含 cameras.json 的文件夹
    valid_datasets = []
    
    # 情况A: 用户直接指向了一个单独的数据集
    if os.path.exists(os.path.join(root_path, "cameras.json")):
        valid_datasets.append(root_path)
    # 情况B: 用户指向了包含多个数据集的根目录
    else:
        for folder in subfolders:
            if os.path.exists(os.path.join(folder, "cameras.json")):
                valid_datasets.append(folder)
    
    valid_datasets.sort()

    print(f"Found {len(valid_datasets)} valid datasets: {[os.path.basename(p) for p in valid_datasets]}")
    
    # 2. 遍历处理
    for dataset_path in valid_datasets:
        try:
            # 注意：在 process_single_scene 内部，lp.extract(args) 会使用当前的默认参数
            # 如果每个子场景的训练参数（如 sh_degree）不同，这里其实使用的是 CLI 的默认值（通常是3）
            # 对于大多数情况这没问题。
            process_single_scene(dataset_path, args, lp, pp, args.cam_cut)
        except Exception as e:
            print(f"CRITICAL ERROR processing {dataset_path}: {e}")
            # 打印详细错误堆栈，方便调试
            import traceback
            traceback.print_exc()
            continue 

    print("\n[ALL JOBS FINISHED]")