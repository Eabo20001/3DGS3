import torch
import torch.nn as nn

from utils.motion_vectors import CudaForwardMotionVector, CudaBackwardMotionVector
from mv_tools.fwd_warper import  SimpleDepthAwareForwardWarp, DifferentiableSimpleDepthAwareForwardWarp

class CAB(nn.Module):
    def __init__(self):
        super(CAB, self).__init__()
        def conv_block(in_channels, out_channels, kernel_size=3, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.LeakyReLU(),
            )
        self.conv = conv_block(2, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=-3)
        max_pool, _ = torch.max(x, dim=-3)
        cat = torch.cat([avg_pool, max_pool], dim = -3)
        w = self.conv(cat)
        return w * x + x
    
class BaseBlock(nn.Module):
    def __init__(self, channels = 64):
        super(BaseBlock, self).__init__()
        def conv_block(in_channels, out_channels, kernel_size=3, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.LeakyReLU(),
            )
        self.channels = channels
        self.conv = conv_block(self.channels, self.channels)
        self.pool = nn.MaxPool2d(kernel_size=4)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x1 = self.pool(x)
        x1 = torch.nn.functional.interpolate(x1, size=x.shape[-2:], align_corners=True, mode='bilinear' )

        x2 = self.conv(x)
        LeakyReLU = nn.LeakyReLU()
        return LeakyReLU(x1+x2)

class InterpolateFusionNet(nn.Module):
    def __init__(self, mid_channels=64, out_channels=3, histDim=32, scale = 4):
        super().__init__()
        def conv_block(in_channels, out_channels, kernel_size=3, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.PReLU(),
            )

        self.color_embed_last = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=2),
            nn.PReLU(),
            nn.Conv2d(16, mid_channels//4, kernel_size=3, padding=1, stride=2),
            nn.PReLU(),
        )
    
        self.mask_embed_last = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=2),
            nn.PReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=2),
            nn.PReLU(),
        )

        self.color_embed_next = conv_block(16, mid_channels//4)
        self.mask_embed_next = conv_block(1, 1)

        self.encoder = conv_block(mid_channels//2 + histDim, mid_channels)
 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.PixelShuffle(2)
        self.downsample = nn.PixelUnshuffle(2)

        self.encoder1 = BaseBlock(mid_channels)
        self.encoder2 = BaseBlock(mid_channels)
        
        self.bottleneck = BaseBlock(mid_channels)
        
        self.decoder2 = conv_block(mid_channels//4 + mid_channels, mid_channels)
        self.decoder1 = conv_block(mid_channels//4 + mid_channels, mid_channels)

        self.skip_connection1 = CAB()
        self.skip_connection2 = CAB()

        self.conv_history1 = nn.Sequential(
            nn.Conv2d(mid_channels, histDim, kernel_size=1),
            nn.Tanh(),
        )
        self.conv_history2 = nn.Sequential(
            nn.Conv2d(mid_channels, histDim, kernel_size=1),
            nn.Sigmoid(),
        )

        self.conv_final_upsample = nn.ConvTranspose2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=scale, stride=scale, padding=0, output_padding=0, bias=False)

        # self.weight_decoder = nn.ConvTranspose2d(in_channels=mid_channels, out_channels=1, kernel_size=scale, stride=scale, padding=0, output_padding=0, bias=False)

        self.weight_decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=mid_channels, out_channels=1, kernel_size=scale, stride=scale, padding=0, output_padding=0, bias=False),
            nn.Sigmoid(),
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def color_mask_encode_last(self, color, mask):
        x = self.color_embed_last(color)
        # t = self.mask_embed_last(mask)
        t = self.mask_embed_next(mask)
        return t * x
    
    def color_mask_encode_next(self, color, mask):
        x = self.color_embed_next(color)
        return x
        # t = self.mask_embed_next(mask)
        # return t * x

    def forward(self, net_input_data):
        # net_input_data require key: image_last, mask_last, image_with_grad_next, mask_next, warped_history

        x = self.encoder(torch.cat([self.color_mask_encode_last(net_input_data['image_last'],           net_input_data['mask_next']),
                                    net_input_data['warped_history'],
                                    self.color_mask_encode_next(net_input_data['image_with_grad_next'], net_input_data['mask_next'])], dim=-3))

        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1)) 
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc2)) 
        
        # Decoder
        dec2 = self.upsample(bottleneck)
        dec2 = torch.cat((dec2, self.skip_connection2(enc2)), dim=-3)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upsample(dec2)
        dec1 = torch.cat((dec1, self.skip_connection1(enc1)), dim=-3)
        dec1 = self.decoder1(dec1)
        # output = self.conv_final(dec1)
        output_upsample = self.conv_final_upsample(dec1)
        weight = self.weight_decoder(dec1)

        return {
            "history": self.conv_history1(dec1) + self.conv_history2(dec1) * net_input_data['warped_history'],
            "fusion_result_sr": output_upsample,
            "weight": weight,
            "FI_result_sr":  (1 + (weight - 1) * net_input_data['mask_last']) * output_upsample.detach() + (1 - weight) * net_input_data['image_last'],
        }
    
class InterpolateNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.FusionModel = InterpolateFusionNet()
        self.differentiable_forward_warping = DifferentiableSimpleDepthAwareForwardWarp()
        self.forward_warping = SimpleDepthAwareForwardWarp()
        self.grad_color_encode = nn.Sequential(
            nn.Conv2d(9, 16, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.PReLU(),
        )

    def warping_last(self, render_pkg, view):
        motion_vector, depth = CudaForwardMotionVector(render_pkg['view'], render_pkg['median_depth'], view)

        motion_vector = motion_vector.permute(2,0,1)
        depth = depth.permute(2,0,1)

        # motion_vector_sr = torch.nn.functional.interpolate(motion_vector.unsqueeze(0), scale_factor=4, mode='nearest').squeeze(0) * 4
        # depth_sr = torch.nn.functional.interpolate(depth.unsqueeze(0), scale_factor=4, mode='bilinear').squeeze(0)

        motion_vector_sr, depth_sr = CudaForwardMotionVector(render_pkg['view'], render_pkg['hr_depth'], view)
        motion_vector_sr = motion_vector_sr.permute(2,0,1)
        depth_sr = depth_sr.permute(2,0,1)


        # warp image
        data = render_pkg["SR_result"].squeeze(0).detach()
        warped_data, _ = self.forward_warping.forward(data, motion_vector_sr, depth_sr)

        # warp history
        data = render_pkg["history"].squeeze(0)
        warped_history, _ = self.differentiable_forward_warping.forward(data, motion_vector, depth)

        return {
            "image_last": warped_data[:-1, :, :].unsqueeze(0),
            "mask_last": warped_data[-1:, :, :].unsqueeze(0),
            "warped_history": warped_history[:-1, :, :].unsqueeze(0),
        }
    
    def warping_next(self, render_pkg, view):
        motion_vector, depth = CudaForwardMotionVector(render_pkg['view'], render_pkg['median_depth'], view)

        motion_vector = motion_vector.permute(2,0,1)
        depth = depth.permute(2,0,1)

        data = self.grad_color_encode(torch.cat([render_pkg["render"], render_pkg["dcolor_dp"]], dim=0))
        # data = render_pkg["render"]
        warped_data, _ = self.differentiable_forward_warping.forward(data, motion_vector, depth)

        return {
            "image_with_grad_next": warped_data[:-1, :, :].unsqueeze(0),
            "mask_next": warped_data[-1:, :, :].unsqueeze(0),
        }
        
    def forward(self, render_pkg_last:dict, view_now, render_pkg_next:dict):
        # render_pkg_last require key: median_depth(lr), view, SR_result(sr), history(lr)
        # render_pkg_next require key: median_depth(lr), view, render(lr), dcolor_dp(lr)
        data = {}
        data.update(
            self.warping_last(render_pkg_last, view_now)
        )
        data.update(
            self.warping_next(render_pkg_next, view_now)
        )

        assert torch.isnan(data["image_last"]).sum() == 0, print(data["image_last"])
        assert torch.isnan( data["mask_last"]).sum() == 0, print( data["mask_last"])
        assert torch.isnan( data["mask_next"]).sum() == 0, print( data["mask_next"])
        
        result = {
            "view": view_now,
            "image_last": data["image_last"],
            "mask_last": data["mask_last"],
            "mask_next": data["mask_next"],
        }

        result.update(
            self.FusionModel(data)
        )
        return result
    
class GRURefine_XOnly(nn.Module):
    def __init__(self, inputDim=11, histDim=32, numBasis=81):
        super().__init__()
        # <- Hist weight ->
        self.updateWeight = nn.Conv2d(inputDim+histDim, histDim, 3, 1, 1, 1, 1, True)
        self.resetWeight = nn.Conv2d(inputDim+histDim, histDim, 3, 1, 1, 1, 1, True)
        self.histTransform = nn.Conv2d(histDim, histDim, 3, 1, 1, 1, 1, False)
        
        # <- New info weight ->
        self.xWeight = nn.Sequential(
            nn.Conv2d(inputDim, histDim, 3, 1, 1, 1, 1, True, padding_mode="replicate"),
            nn.PReLU(),
            nn.Conv2d(histDim, histDim, 3, 1, 1, 1, 1, True, padding_mode="zeros"),
            nn.PReLU()
        )
        self.hist2Residual = nn.Sequential(
            nn.Conv2d(histDim, histDim, 3,1,1,1,1,True),
            nn.PReLU(),
            nn.Conv2d(histDim, numBasis, 3,1,1,1,1,True),
            nn.PReLU(),
        )
        
        # <- Save the cfg ->
        self.inputDim=inputDim
        self.histDim = histDim
        self.numBasis = numBasis
        
    def forward(self, x:torch.Tensor, hist:torch.Tensor):
        concatInput = torch.concat([x, hist], dim=-3)
        uWei = nn.functional.sigmoid(self.updateWeight(concatInput))
        rWei = nn.functional.sigmoid(self.resetWeight(concatInput))
        hInfo = self.histTransform(hist)
        xEnc = self.xWeight(x)
        updateInfo = nn.functional.tanh(rWei * hInfo + xEnc)
        updatedHist = uWei * hist + (1-uWei) * updateInfo
        residual = self.hist2Residual(updatedHist)
        return residual, updatedHist

    def getEmptyHist(self, x:torch.Tensor):
        *B, _, H, W = x.shape
        hist = torch.zeros((*B, self.histDim, H, W), dtype=x.dtype, device=x.device)
        return hist
    
class EmbedInterpolation(nn.Module):
    def __init__(self, scale=4, inputDims=9, outputDims=3, numBasis=81):
        super().__init__()
        self.embed = nn.Conv2d(inputDims, numBasis, kernel_size=3, padding=1, bias=False, padding_mode="replicate")
        self.basis =  nn.ConvTranspose2d(in_channels=numBasis, out_channels=outputDims, kernel_size=scale, stride=scale, padding=0, output_padding=0, bias=False)
        
    def forward_naive(self, x):
        return self.basis(self.embed(x))
    
    def forward(self, x, injectResidual):
        em = self.embed(x)
        newEm = em + injectResidual
        return self.basis(newEm)
    
class SuperResolutionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.GRUModel = GRURefine_XOnly()
        self.SRModel = EmbedInterpolation()
        self.differentiable_forward_warping = DifferentiableSimpleDepthAwareForwardWarp()
        # self.differentiable_forward_warping = SimpleDepthAwareForwardWarp()

        self.i = None
        self.j = None

    def warp_history(self, render_pkg_history, render_pkg):
        h, w = render_pkg['median_depth'].shape[-2:]
        motion_vector, _ = CudaBackwardMotionVector(render_pkg_history['view'], render_pkg['view'], render_pkg['median_depth'])
        # if torch.isnan( motion_vector).sum() != 0:
        #     print(render_pkg['median_depth'][0, 248, 508])

        motion_vector = torch.nan_to_num(motion_vector, nan=0.0, posinf=0.0, neginf=0.0)
        assert torch.isnan( motion_vector).sum() == 0, print( torch.where(torch.isnan( motion_vector)))

        if self.i == None:
            device = motion_vector.device
            i, j = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            self.i = i.to(device)
            self.j = j.to(device)
        
        grid_cord = torch.stack([ 2.0 * (self.j + motion_vector[:, :, 0]) / w  - 1, 2.0 * (self.i + motion_vector[:, :, 1]) / h - 1], -1).unsqueeze(0)
        assert torch.isnan( grid_cord).sum() == 0, print( grid_cord)
        assert torch.isnan( render_pkg_history['history']).sum() == 0, print( render_pkg_history['history'])
        grid_cord = torch.nan_to_num(grid_cord, nan=0.0, posinf=0.0, neginf=0.0)

        warped_history = torch.nn.functional.grid_sample(render_pkg_history['history'], grid_cord, align_corners=True)#.squeeze(0)
        # print(grid_cord.shape)
        # exit(0)
        if torch.isnan( warped_history).sum() != 0:
            print(warped_history[0, 0, 16, 148])
            print((grid_cord[0, 16, 148, 0], grid_cord[0, 16, 148, 1]))

        assert torch.isnan( warped_history).sum() == 0, print( torch.where(torch.isnan( warped_history)))
        return warped_history
        
    def forward(self, render_pkg_history, render_pkg_now:dict):
        # render_pkg_history require key: history(lr), view
        # render_pkg_now require key: median_depth(lr), view, render(lr), dcolor_dp(lr) Gbuffer
        
        current_nov = render_pkg_now['normal'][-1:, ...]
        current_dep = render_pkg_now['median_depth'] / render_pkg_now['median_depth'].max()
        current_frame_input = torch.concat([render_pkg_now['render'], render_pkg_now['dcolor_dp'], current_nov, current_dep], dim=0).unsqueeze(0)
    
        if render_pkg_history is None:
            warped_history = self.GRUModel.getEmptyHist(current_frame_input)
            assert torch.isnan( warped_history).sum() == 0, print( warped_history)
        else:
            warped_history = self.warp_history(render_pkg_history, render_pkg_now)
            assert torch.isnan( warped_history).sum() == 0, print( warped_history)

        assert torch.isnan( current_frame_input).sum() == 0, print( current_frame_input)
        assert torch.isnan( warped_history).sum() == 0, print( warped_history)
        
        residual, updated_history = self.GRUModel(current_frame_input, warped_history)
        sr_image = self.SRModel(current_frame_input[:, :9, ...], residual)

        assert torch.isnan( updated_history).sum() == 0, print( updated_history)
        assert torch.isnan( sr_image).sum() == 0, print( sr_image)

        return {
            "history": updated_history,
            "view": render_pkg_now["view"],
            "SR_result": sr_image,
            "median_depth": render_pkg_now['median_depth'],
            "hr_depth": render_pkg_now['hr_depth'],
        }