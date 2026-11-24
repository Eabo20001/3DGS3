import torch
from loss import register

@register('L1_loss')
def L1_Loss(img, gt, **kwargs):
    L1_loss = torch.nn.L1Loss()
    return L1_loss(img, gt)

@register('Charbonnier_loss')
def L1_Charbonnier_loss(img, gt, **kwargs):
    eps = 1e-6
    diff = torch.add(gt, -img)
    error = torch.sqrt(diff * diff + eps)
    loss = torch.mean(error)
    return loss