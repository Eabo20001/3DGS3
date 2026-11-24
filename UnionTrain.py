import sys
import argparse
import yaml
import loss
import torch
import torchvision
import numpy as np
from tqdm import tqdm
import os
import json

import numpy as np
import torchvision

from scene.cameras import Camera
from gaussian_renderer import GaussianModel, render
from utils.system_utils import searchForMaxIteration

from utils.image_utils import psnr

import toml
from utils.graphics_utils import focal2fov,fov2focal

from mv_tools.fwd_warper import DifferentiableSimpleDepthAwareForwardWarp
from utils.motion_vectors import CudaForwardMotionVector, CudaBackwardMotionVector

from UnionModel import InterpolateNet, SuperResolutionNet
from TemporalLoss import VGGPerceptualLoss

def prepare_renderfunc_cameras(config_fn:str):
    # [load model]
    config = toml.load(config_fn)
    dataset = ModelParams(config['dataset'])
    pipeline = PipelineParams(config['pipeline'])
    gaussians = GaussianModel(dataset.sh_degree)
    loaded_iter = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))
    gaussians.load_ply(os.path.join(dataset.model_path,
                                                        "point_cloud",
                                                        "iteration_" + str(loaded_iter),
                                                        "point_cloud.ply"))
    background = torch.tensor([1,1,1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    
    # [load cams]
    with open(os.path.join(dataset.model_path, "cameras.json"), 'r') as file:
        cam_list = json.load(file)
    def dict_to_cam(dc:dict):
        Rt = np.zeros((4,4))
        Rt[-1,-1] = 1
        for i in range(3):
            for j in range(3):
                Rt[i,j] = dc['rotation'][i][j]
        for i in range(3):
            Rt[i, 3] = dc['position'][i]
        Rt = np.linalg.inv(Rt)
        
        fx = dc['fx']
        fy = dc['fy']
        w = dc['width']
        h = dc['height']
        fovx = focal2fov(fx, w)
        fovy = focal2fov(fy, h)
        return Camera(
            dc['id'],Rt[:3,:3].transpose(),Rt[:3, 3],fovx,fovy, dc['img_name'],0,resolution=(h,w)
        )
        
    cams = [dict_to_cam(x) for x in cam_list]

    def render_cam(pcam):
        with torch.no_grad():
            return render(pcam, gaussians, pipeline, background, dataset.kernel_size)
    
    sr_cfg = config.get("sr", {"predown":8, "scale":4})
    return render_cam, cams,sr_cfg

"""
获取简单的平移相机
"""
def generate_xshift_cams(demo_cam:Camera, total_distance=1, steps=10):
    step_size = total_distance / steps
    beg_w2c = np.eye(4,4)
    beg_w2c[:3,:3] = demo_cam.R
    beg_w2c[3, :3] = demo_cam.T
    beg_c2w = np.linalg.inv(beg_w2c)
    current_pos = beg_c2w[3,:3]
    current_x_dir = beg_c2w[0,:3]
    cams = [demo_cam]
    for id in range(steps-1):
        current_pos += current_x_dir * step_size
        now_c2w = beg_c2w.copy()
        now_c2w[3, :3] = current_pos
        w2c44 = np.linalg.inv(now_c2w)
        cams.append(
            Camera(
                0, w2c44[:3,:3], w2c44[3,:3], demo_cam.FoVx, demo_cam.FoVy, f"{demo_cam.image_name}-generate_xshift_cams-{id}", 0, resolution=(demo_cam.image_height, demo_cam.image_width)
            )
        )
    return cams

class ModelParams: 
    def __init__(self, config):
        self.sh_degree = config['sh_degree']
        self.source_path = config['source_path']
        self.model_path = config['model_path']
        self.images = config['images']
        self.depths = config['depths']
        self.resolution = config['resolution']
        self.white_background = config['white_background']
        self.train_test_exp = config['train_test_exp']
        self.data_device = config['data_device']
        self.eval = config['eval']
        self.kernel_size = config['kernel_size']
class PipelineParams:
    def __init__(self, config):
        self.convert_SHs_python = config['convert_SHs_python']
        self.compute_cov3D_python = config['compute_cov3D_python']
        self.debug = config['debug']
        self.antialiasing = config['antialiasing']

def JSON_to_camera(camera_entry):
    resolution = (camera_entry['height'], camera_entry['width'])
    resolution = (424, 648)
    return Camera(resolution = resolution, 
                  colmap_id=camera_entry['colmap_id'], 
                  R=np.array(camera_entry['R']), 
                  T=np.array(camera_entry['T']), 
                  FoVx=camera_entry['FoVx'], 
                  FoVy=0.8296521927446754, #camera_entry['FoVy'], 
                  image_name=camera_entry['image_name'], 
                  uid=camera_entry['id'])


def gene_lr_hr_cams_new(raw_cam: Camera, pre_downscale, sr_scale, lr_res=(512, 512)):
    
    hr_res = (lr_res[0] * sr_scale,lr_res[1]*sr_scale)

    lr_fx = fov2focal(raw_cam.FoVx, lr_res[1])
    lr_fy = fov2focal(raw_cam.FoVy, lr_res[0])
    hr_fx = fov2focal(raw_cam.FoVx, hr_res[1])
    hr_fy = fov2focal(raw_cam.FoVy, hr_res[0])

    lr_f = (lr_fx + lr_fy) / 2 
    hr_f = (hr_fx + hr_fy) / 2
    
    
    lr_fovx = focal2fov(lr_f, lr_res[1])
    lr_fovy = focal2fov(lr_f, lr_res[0])
    
    hr_fovx = focal2fov(hr_f, hr_res[1])
    hr_fovy = focal2fov(hr_f, hr_res[0])
    
    lrcam = Camera(
        resolution=lr_res,
        colmap_id = 0,
        R = raw_cam.R,
        T = raw_cam.T,
        FoVx= lr_fovx,
        FoVy= lr_fovy,
        image_name=f"{raw_cam.image_name}-lr",
        uid = 0,
    )
    hrcam = Camera(
        resolution= hr_res,
        colmap_id = 0,
        R = raw_cam.R,
        T = raw_cam.T,
        FoVx= hr_fovx,
        FoVy= hr_fovy,
        image_name=f"{raw_cam.image_name}-hr",
        uid = 0,
    )
    return lrcam, hrcam 

class UnionModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gene_lr_hr_cams_func = gene_lr_hr_cams_new

        self.FIModel = InterpolateNet()
        self.SRModel = SuperResolutionNet()

    def forward(self, views, render_fn, sr_cfg, gen_cams_func = None):
        if len(views) % 2 == 0:
            views = views[:-1]

        predown = sr_cfg['predown']
        scale = sr_cfg['scale']

        if gen_cams_func is None:
            gen_cams_func = self.gene_lr_hr_cams_func

        def get_lr(pkg:dict):
            result = {}
            for key, value in pkg.items():
                if key in ["render", "normal", "median_depth", "dcolor_dp"]:
                    result[key] = value[:, ::4, ::4]
                else:
                    result[key] = value
            result["hr_depth"] = pkg["median_depth"]
            return result

        render_pkgs = []
        render_pkgs_lr = []
        for view in views:
            _, view_hr = gen_cams_func(view, predown, scale)
            render_pkg = render_fn(view_hr)
            render_pkg['view'] = view_hr
            render_pkgs.append(render_pkg)
            render_pkgs_lr.append(get_lr(render_pkg))

        result = []
        for id, view in enumerate(views):
            if id==0:
                result.append(self.SRModel(None, render_pkgs_lr[id]))
            elif id % 2 == 1:
                result.append(self.FIModel(result[-1], view, render_pkgs_lr[id + 1]))
            else:
                result.append(self.SRModel(result[-1], render_pkgs_lr[id]))

        return {
            "render_pkgs": render_pkgs,
            "result": result,
        }
        
class LossFunc(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.SR_loss = config['SR_loss']
        self.FI_loss = config['FI_loss']
        self.mse_loss = torch.nn.MSELoss()
        self.vgg = VGGPerceptualLoss()
        self.differentiable_forward_warping = DifferentiableSimpleDepthAwareForwardWarp()
        
    def SRLoss(self, SR_result, render_pkg):
        pred_img_sr = SR_result['SR_result'].squeeze(0)
        gt_imgage_hr = render_pkg["render"]
        
        loss_tmp = []

        assert torch.isnan( pred_img_sr).sum() == 0, print( pred_img_sr)

        for loss_cfg in self.FI_loss:
            loss_tmp.append(loss_cfg['args']['loss_weight'] * loss.loss(loss_cfg, img = pred_img_sr, gt = gt_imgage_hr))
        
        loss_tol = sum(loss_tmp)
        assert torch.isnan( loss_tol).sum() == 0, print( loss_tol)
        return {
            "loss": loss_tol / loss_tol.detach(),
            "loss_show": self.mse_loss(pred_img_sr.detach(), gt_imgage_hr).item()
        }
    
    def FILoss(self, FI_result, render_pkg):
        pred_img_sr = FI_result['FI_result_sr'].squeeze(0)
        mask_sr = FI_result["mask_last"].squeeze(0)
        weight = FI_result["weight"].squeeze(0)
        gt_imgage_hr = render_pkg["render"]

        loss_tmp = []

        loss_weight =  10 * (1 - mask_sr)
        loss_tmp.append(self.mse_loss(loss_weight * pred_img_sr, loss_weight * gt_imgage_hr))

        # loss_tmp.append(0.2 * self.mse_loss(1 - mask_sr, weight))
        loss_tmp.append(0.1 * (weight * (1 - weight)).mean() )
        loss_tmp.append(self.mse_loss(weight, torch.abs(gt_imgage_hr - FI_result['image_last'].squeeze(0)).max(dim=0, keepdim=True)[0]))
        
        for loss_cfg in self.FI_loss:
            loss_tmp.append(loss_cfg['args']['loss_weight'] * loss.loss(loss_cfg, img = pred_img_sr, gt = gt_imgage_hr))

        assert torch.isnan( pred_img_sr).sum() == 0, print( pred_img_sr)
        assert torch.isnan( mask_sr).sum() == 0, print( mask_sr)
        assert torch.isnan( weight).sum() == 0, print( weight)
        assert torch.isnan(  FI_result['image_last']).sum() == 0, print(  FI_result['image_last'])
        
        
        loss_tol = sum(loss_tmp)
        assert torch.isnan(  loss_tol).sum() == 0, print(  loss_tol)
        return {
            "loss": loss_tol / loss_tol.detach(),
            "loss_show": self.mse_loss(pred_img_sr.detach(), gt_imgage_hr).item()
        }
    
    def FusionLoss(self, FI_result, render_pkg):
        pred_fusion = FI_result['fusion_result_sr'].squeeze(0)
        gt_imgage_hr = render_pkg["render"]
    
        loss_tmp = []

        for loss_cfg in self.FI_loss:
            loss_tmp.append(loss_cfg['args']['loss_weight'] * loss.loss(loss_cfg, img = pred_fusion, gt = gt_imgage_hr))
        
        loss_tol = sum(loss_tmp)
        return {
            "loss": loss_tol / loss_tol.detach(),
            "loss_show": self.mse_loss(pred_fusion.detach(), gt_imgage_hr).item()
        }
    
    def TempLoss(self, SR_result_last, FI_result, SR_result_next, FI_render):
        SR_img_last = SR_result_last['SR_result'       ]
        FI_img      = FI_result     ['fusion_result_sr']
        SR_img_next = SR_result_next['SR_result'       ]

        def WarpLoss(FI_result, FI_render, SR_result):
            motion_vector, depth = CudaForwardMotionVector(FI_result['view'], FI_render['median_depth'], SR_result['view'])

            motion_vector = motion_vector.permute(2,0,1)
            depth = depth.permute(2,0,1)

            data = FI_result['fusion_result_sr'].squeeze(0)
            GT   = SR_result['SR_result'   ].squeeze(0)
            warped_data, _ = self.differentiable_forward_warping.forward(data, motion_vector, depth)

            img  = warped_data[:-1, :, :]
            mask = warped_data[-1:, :, :]
            return self.mse_loss(img * mask, GT * mask)
        
        loss_tmp = []
        loss_tmp.append(self.vgg(input = FI_img, target = SR_img_last))
        loss_tmp.append(self.vgg(input = FI_img, target = SR_img_next))
        loss_tmp.append(WarpLoss(FI_result, FI_render, SR_result_last))
        loss_tmp.append(WarpLoss(FI_result, FI_render, SR_result_next))

        
        loss_tol = sum(loss_tmp)
        return {
            "loss": loss_tol / loss_tol.detach(),
        }
    
class DataLoader:
    def __init__(self):
        self.train_views = ['bicycle.toml', 'kitchen.toml', 'stump.toml', 'truck.toml']
        self.test_views = ['garden.toml', ]

    def get_test_views(self):
        render_fn, cams, sr_cfg = prepare_renderfunc_cameras(os.path.join("config/exp-configs", self.test_views[0]))
        return render_fn, cams, sr_cfg
    
    def get_train_views(self):
        for fn in self.train_views:
            render_fn, cams, sr_cfg = prepare_renderfunc_cameras(os.path.join("config/exp-configs", fn))
            yield render_fn, cams, sr_cfg
    

class Trainer:
    def __init__(self, config):
        self.config = config
        self.max_epoch = self.config['max_epoch']

        self.model = UnionModel(config).cuda()
        self.loss = LossFunc(config).cuda()
        self.dataloader = DataLoader()
        
        self.optim_params = []
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                self.optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
                
        if self.config['optimizer_type'] == 'adam':
            self.optimizer = torch.optim.Adam(self.optim_params, lr=self.config['optimizer_lr'],
                                weight_decay=self.config['optimizer_wd'])
        else:
            raise NotImplementedError

        if self.config['scheduler_type'] == 'MultiStepLR':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                            [int(x*self.config['max_epoch'])for x in self.config['scheduler_milestones']],
                                                            self.config['scheduler_gamma']
                                                            )
        elif self.config['scheduler_type'] == 'CosineAnnealingWarmRestarts':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
                                                            self.config['scheduler_periods'],
                                                            self.config['scheduler_restart_weights'],
                                                            self.config['scheduler_eta_min']
                                                            )
        else:
            raise NotImplementedError
    
    #@profile
    def forward(self, current_step, views, render_fn, sr_cfg, is_train = True):
        torch.cuda.empty_cache()
        net_output = self.model(views, render_fn, sr_cfg)
        render_pkgs = net_output["render_pkgs"]
        result = net_output["result"]

        loss_tmp = []
        loss_show = 0.0
        for id, render_pkg in enumerate(render_pkgs):
            if id % 2 == 0:
                loss = self.loss.SRLoss(result[id], render_pkg)
                loss_tmp.append(loss['loss'])
                loss_show += loss['loss_show']
            else:
                loss = self.loss.FusionLoss(result[id], render_pkg)
                loss_tmp.append(loss["loss"])

                if current_step > 6:
                    loss_tmp.append((0.1 + (current_step / self.max_epoch) ) * self.loss.FILoss(result[id], render_pkg)['loss'])
                if current_step > 12:
                    loss_tmp.append(self.loss.TempLoss(result[id-1], result[id], result[id+1], render_pkg)["loss"])
                loss_show += loss['loss_show']
        loss_tol = sum(loss_tmp)
        
        if is_train:
            return {
                "loss": loss_tol,
                "loss_show": loss_show,
            }
        else:
            result_data = {
                "loss_show": loss_show,
                "render_pkgs": render_pkgs,
                "result": result,
            }
            return result_data
        
    def optimize_parameters(self, current_step, views, render_fn, sr_cfg):
        self.optimizer.zero_grad()
        loss_tol = self.forward(current_step, views, render_fn, sr_cfg)
        loss_tol["loss"].backward()

        self.optimizer.step()
        return loss_tol["loss_show"]
    
    def test(self, current_step, views, render_fn, sr_cfg):
        with torch.no_grad():
            result_data = self.forward(current_step, views, render_fn, sr_cfg, is_train = False)
        return result_data

    def train(self):
        max_epoch = self.max_epoch
        # iter_num = len(self.train_views)

        cam_seq_num = 8
        train_seq_num = 7
    
        for epoch in tqdm(range(max_epoch), position=0, desc="epoch", leave=False, colour='green'):
            if epoch % self.config['checkpoint']['save_interval'] == 0:
                self.save_checkpoint(epoch)

            self.model.train()
            interval = epoch % 5 + 1
            for render_fn, cams, sr_cfg in self.dataloader.get_train_views():
                iter_num = cam_seq_num - train_seq_num + 1
                with tqdm(total=iter_num * len(cams), position=1, desc="iter", leave=False, colour='red') as t:
                    for c in cams:
                        shifting_cams = generate_xshift_cams(c, 0.3 * interval, cam_seq_num)
                        for id in range(iter_num):
                            views = shifting_cams[id: id + train_seq_num]
                            loss_tol = self.optimize_parameters(epoch, views, render_fn, sr_cfg)
                            t.set_postfix(loss=loss_tol)
                            t.update()
            self.scheduler.step()
            
        self.save_checkpoint(max_epoch)

    def save_checkpoint(self, current_step):
        path = os.path.join(self.config['checkpoint']['path'], self.config['checkpoint']['name'],'epoch_{:06d}'.format(current_step))
        os.makedirs(path, exist_ok=True)
        self.model.eval()
        
        torch.save({
            'epoch': current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, os.path.join(path, 'checkpoint.pth'))

        # result_data = self.test(current_step)
        render_fn, cams, sr_cfg = self.dataloader.get_test_views()
        shifting_cams = generate_xshift_cams(cams[0], 0.3 * 5, 25)

        result_data = self.test(current_step, shifting_cams[-7:], render_fn, sr_cfg)

        FI_pred = result_data["result"][-2]["FI_result_sr"].squeeze(0)
        FI_fuse = result_data["result"][-2]["fusion_result_sr"].squeeze(0)
        FI_weight = result_data["result"][-2]["weight"].squeeze(0).repeat(3, 1, 1)
        FI_GT = result_data["render_pkgs"][-2]["render"]
        FI_psnr = psnr(FI_pred, FI_GT).mean()
        torchvision.utils.save_image(torch.cat([
            FI_pred,
            FI_fuse,
            FI_weight,
            FI_GT, 
        ], dim = -1), os.path.join(path, 'FI_psnr{:.4f}.png').format(FI_psnr))

        SR_pred = result_data["result"][-1]["SR_result"].squeeze(0)
        SR_GT = result_data["render_pkgs"][-1]["render"]
        SR_psnr = psnr(SR_pred, SR_GT).mean()
        torchvision.utils.save_image(torch.cat([
            SR_pred,
            SR_GT,
        ], dim = -1), os.path.join(path, 'SR_psnr{:.4f}.png').format(SR_psnr))

    def load_checkpoint(self):
        checkpoint_path = os.path.join(self.config['checkpoint']['path'], self.config['checkpoint']['name'])
        saved_epochs = [int(fname.split("_")[-1]) if fname.startswith("epoch_") else -1 for fname in os.listdir(checkpoint_path)]
        max_epochs = max(saved_epochs)
        print(max_epochs)
        checkpoint = torch.load(os.path.join(checkpoint_path, 'epoch_{:06d}'.format(max_epochs), "checkpoint.pth"))
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def benchmark(self):
        import time
        NUM_ITERATIONS = 100
        start_time = time.perf_counter()
        for _ in range(NUM_ITERATIONS):
            result_data = self.test(-1, self.test_views[0], self.test_views[4], self.test_views[8])
        end_time = time.perf_counter()
        total_time = end_time - start_time
        total_time_ms = total_time * 1000
        total_num_images = NUM_ITERATIONS  # * len(views)
        avg_fps = total_num_images / total_time
        avg_ms_per_image = total_time_ms / total_num_images
        print(f'Number of test set renders: {NUM_ITERATIONS}\n')
        print(f'Total rendering time: {total_time_ms:.2f} ms\n')
        print(f'Average rendering time per image: {avg_ms_per_image:.2f} ms\n')
        print(f'Average FPS: {avg_fps:.2f}\n')
        print(f'Average FPS: {avg_fps:.2f} ({avg_ms_per_image:.2f} ms)')

                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config/union_model.yaml", help='path to config file')

    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        

    # if os.system("git switch Train_auto_save") == 0:
    #     os.system("git add .")
    #     os.system(f"git commit -m {config['checkpoint']['name']}")

    trainer = Trainer(config)
    # with tqdm(total=max_epoch * len(trainer.scene.getTrainCameras())) as t:
    #     for epoch in tqdm(range(max_epoch)):
    #         for view in trainer.scene.getTrainCameras():
    #             trainer.test(epoch, view)
    # exit(0)

    trainer.train()

    #trainer.load_checkpoint()

    # trainer.benchmark()
    #trainer.save_checkpoint(-1)

