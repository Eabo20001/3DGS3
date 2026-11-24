import torch
import torch.nn as nn
import torch.nn.functional as F
from models import register

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


@register('interpolate_net')
class InterpolateNet(nn.Module):
    def __init__(self, color_channels=9, mask_channels=1, depth_channels=2, grad_channels = 14, mid_channels=64, out_channels=3, scale = 4):
        super(InterpolateNet, self).__init__()
        def conv_block(in_channels, out_channels, kernel_size=3, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.PReLU(),
            )
        self.need_downsample = False
        self.color_embed = conv_block(color_channels * (4 if self.need_downsample else 1), mid_channels)
        self.mask_embed = conv_block(mask_channels * (4 if self.need_downsample else 1), 1)
        
        self.depth_embed = conv_block(depth_channels * (4 if self.need_downsample else 1), 8)

        self.grad_embed = conv_block(grad_channels * (4 if self.need_downsample else 1), 32)
        
        self.img_encoder = conv_block(9, (mid_channels//2)//(4 if self.need_downsample else 1))

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

        if self.need_downsample:
            self.conv_final = nn.Conv2d(mid_channels, out_channels * 4, kernel_size=1)
            self.conv_final_upsample = nn.ConvTranspose2d(in_channels=mid_channels, out_channels=out_channels * 4, kernel_size=scale, stride=scale, padding=0, output_padding=0, bias=False)
        else:
            self.conv_final = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
            self.conv_final_upsample = nn.ConvTranspose2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=scale, stride=scale, padding=0, output_padding=0, bias=False)

        self.grad_decoder = nn.Conv2d(mid_channels, 24, kernel_size=1)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def image_encode(self, color, grad):
        x = torch.cat([color, grad], dim=-3)
        # x = self.downsample(x)
        return self.img_encoder(x)
    
    def color_mask_encode(self, color, mask):
        x1 = self.color_embed(color)
        t = self.mask_embed(mask)
        # x = self.downsample(x)
        return t * x1

    def forward(self, color, mask, image_last, image_next, output_lr = True):
        if self.need_downsample:
            color = self.color_mask_encode(self.downsample(color), self.downsample(mask))
            image_last = self.downsample(image_last)
            image_next = self.downsample(image_next)
        else:
            color = self.color_mask_encode(color, mask)

        x = torch.cat([image_last, image_next], dim=-3) + color
        # x = color

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
        
        if self.need_downsample:
            if output_lr:
                output = self.upsample(self.conv_final(dec1))
            output_upsample = self.upsample(self.conv_final_upsample(dec1))
        else:
            if output_lr:
                output = self.conv_final(dec1)
            output_upsample = self.conv_final_upsample(dec1)
        # grad = self.upsample(self.grad_decoder(dec1))
        # return output
        # return output, output_upsample, grad
        if output_lr:
            return output, output_upsample
        else:
            return output_upsample

if __name__ == '__main__':
    device = "cuda:0"
    x = torch.rand([1, 18, 1296, 840], device=device).half()
    y = torch.rand([1, 2, 1296, 840], device=device).half()
    z = torch.rand([1, 2, 1296, 840], device=device).half()
    w = torch.rand([1, 14, 1296, 840], device=device).half()
    inter = InterpolateNet().half().to(device)
    torch.cuda.synchronize()

    import time
    NUM_ITERATIONS = 5000
    start_time = time.perf_counter()
    for _ in range(NUM_ITERATIONS):
        output = inter(x, y, z, w)
        
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

   
    print(output.shape)