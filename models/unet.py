import torch
import torch.nn as nn
import torch.nn.functional as F
from models import register


@register('unet')
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=9):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels, kernel_size=3, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = conv_block(512, 1024)
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64)
        
        self.conv_final = nn.Conv2d(64, out_channels, kernel_size=1)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.conv_final(dec1)
    
@register('tiny_unet')
class TinyUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=9):
        super(TinyUNet, self).__init__()

        def conv_block(in_channels, out_channels, kernel_size=3, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = conv_block(128, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64)
        
        
        self.conv_final = nn.Conv2d(64, out_channels, kernel_size=1)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc2))
        
        # Decoder
        dec2 = self.upconv2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim=-3)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=-3)
        dec1 = self.decoder1(dec1)
        
        return self.conv_final(dec1)
    
@register('diy_unet')
class DiyUNet(nn.Module):
    def __init__(self, in_channels=3, encoder1_channels = 64, encoder2_channels = 128, bottleneck_channels = 256, upconv2_channels = 128, decoder2_channels = 128, upconv1_channels = 64, decoder1_channels = 64, out_channels=9):
        super(DiyUNet, self).__init__()

        def conv_block(in_channels, out_channels, kernel_size=3, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = conv_block(in_channels, encoder1_channels)
        self.encoder2 = conv_block(encoder1_channels, encoder2_channels)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = conv_block(encoder2_channels, bottleneck_channels)
        
        self.upconv2 = nn.ConvTranspose2d(bottleneck_channels, upconv2_channels, kernel_size=2, stride=2)
        self.decoder2 = conv_block(upconv2_channels + encoder2_channels, decoder2_channels)
        
        self.upconv1 = nn.ConvTranspose2d(decoder2_channels, upconv1_channels, kernel_size=2, stride=2)
        self.decoder1 = conv_block(upconv1_channels + encoder1_channels, decoder1_channels)
        
        
        self.conv_final = nn.Conv2d(decoder1_channels, out_channels, kernel_size=1)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc2))
        
        # Decoder
        dec2 = self.upconv2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim=-3)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=-3)
        dec1 = self.decoder1(dec1)
        
        return self.conv_final(dec1)

class hswish(nn.Module):
    def __init__(self, inplace: bool = False, negative_slope: float = 1e-2,):
        super(hswish, self).__init__()
        self.relu6 = nn.ReLU6(inplace = inplace)
        self.negative_slope = negative_slope
    def forward(self, x):
        return x * (self.relu6(x+3)/6 + self.negative_slope)

@register('6_conv_unet')
class MyUNet(nn.Module):
    def __init__(self, in_channels=3, encoder1_channels = 64, encoder2_channels = 128, bottleneck1_channels = 256, bottleneck2_channels = 128, decoder2_channels = 128, decoder1_channels = 64, out_channels=9):
        super(MyUNet, self).__init__()

        def conv_block(in_channels, out_channels, kernel_size=3, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                hswish(inplace=True),
                # nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                # nn.ReLU(inplace=True)
            )

        self.encoder1 = conv_block(in_channels, encoder1_channels)
        self.encoder2 = conv_block(encoder1_channels, encoder2_channels)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck1 = conv_block(encoder2_channels, bottleneck1_channels)
        self.bottleneck2 = conv_block(bottleneck1_channels, bottleneck2_channels)
        self.upsample = nn.PixelShuffle(2)
        
        # self.upconv2 = nn.ConvTranspose2d(bottleneck_channels, upconv2_channels, kernel_size=2, stride=2)
        self.decoder2 = conv_block(bottleneck2_channels // 4 + encoder2_channels, decoder2_channels)
        
        
        self.decoder1 = conv_block(decoder2_channels // 4 + encoder1_channels, decoder1_channels)
        
        
        self.conv_final = nn.Conv2d(decoder1_channels, out_channels, kernel_size=1)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1)) 
        
        # Bottleneck
        bottleneck1 = self.bottleneck1(self.pool(enc2)) 
        bottleneck2 = self.bottleneck2(bottleneck1) 
        
        # Decoder
        dec2 = self.upsample(bottleneck2)
        dec2 = torch.cat((dec2, enc2), dim=-3)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upsample(dec2)
        dec1 = torch.cat((dec1, enc1), dim=-3)
        dec1 = self.decoder1(dec1)
        
        return self.conv_final(dec1)
    
@register('depth_unet')
class DepthUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=9):
        super(DepthUNet, self).__init__()

        def conv_block(in_channels, out_channels, kernel_size=3, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = conv_block(128, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64)
        
        
        self.conv_final = nn.Conv2d(64, out_channels, kernel_size=1)

        self.conv_depth = nn.Conv2d(out_channels, 1, padding=1, kernel_size=3)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc2))
        
        # Decoder
        dec2 = self.upconv2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim=-3)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=-3)
        dec1 = self.decoder1(dec1)

        final = self.conv_final(dec1)
        
        return final, self.conv_depth(final)

'''# Example usage
model = UNet(in_channels=3, out_channels=9)
input_image = torch.randn(1, 3, 16, 16)  # Batch size of 1, 256x256 image
output_image = model(input_image)
print(output_image.shape)  # Should be [1, 9, 256, 256]'''