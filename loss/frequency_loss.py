import torch
from loss import register

@register('frequency_loss')
def Frequency_Loss(img, gt, **kwargs):
    L1_loss = torch.nn.L1Loss()
    gt_fft = torch.fft.fft2(gt, dim=(-2,-1))
    gt_fft = torch.stack((gt_fft.real, gt_fft.imag), -1)
    

    img_fft = torch.fft.fft2(img, dim=(-2,-1))
    img_fft = torch.stack((img_fft.real, img_fft.imag), -1)
    print(img_fft)
    return L1_loss(img_fft, gt_fft)

