import torch
import torch.nn.functional as F
from loss import register


def gauss_kernel(channels=3, device=torch.device('cuda') ):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel


def regression_loss(logit, target, loss_type='l1', weight=None):
    """
    Alpha reconstruction loss
    :param logit:
    :param target:
    :param loss_type: "l1" or "l2"
    :param weight: tensor with shape [N,1,H,W] weights for each pixel
    :return:
    """
    if weight is None:
        if loss_type == 'l1':
            return F.l1_loss(logit, target)
        elif loss_type == 'l2':
            return F.mse_loss(logit, target)
        else:
            raise NotImplementedError("NotImplemented loss type {}".format(loss_type))
    else:
        if loss_type == 'l1':
            return F.l1_loss(logit * weight, target * weight, reduction='sum') / (torch.sum(weight) + 1e-8)
        elif loss_type == 'l2':
            return F.mse_loss(logit * weight, target * weight, reduction='sum') / (torch.sum(weight) + 1e-8)
        else:
            raise NotImplementedError("NotImplemented loss type {}".format(loss_type))

@register('lap_loss')    
def Lap_Loss(img, gt, gauss_filter = None, loss_type='l1', weight=None, **kwargs):
    '''
    Based on FBA Matting implementation:
    https://gist.github.com/MarcoForte/a07c40a2b721739bb5c5987671aa5270
    '''
    def conv_gauss(x, kernel):
        x = F.pad(x, (2,2,2,2), mode='reflect')
        x = F.conv2d(x, kernel, groups=x.shape[1])
        return x

    def downsample(x):
        return x[..., ::2, ::2]

    def upsample(x):
        N, C, H, W = x.shape
        cc = torch.cat([x, torch.zeros(N,C,H,W).cuda()], dim = 3)
        cc = cc.view(N, C, H*2, W)
        cc = cc.permute(0,1,3,2)
        cc = torch.cat([cc, torch.zeros(N, C, W, H*2).cuda()], dim = 3)
        cc = cc.view(N, C, W*2, H*2)
        x_up = cc.permute(0,1,3,2)
        return conv_gauss(x_up, kernel=4*gauss_filter)

    def lap_pyramid(x, kernel, max_levels=3):
        current = x
        pyr = []
        for level in range(max_levels):
            filtered = conv_gauss(current, kernel)
            down = downsample(filtered)
            up = upsample(down)
            _, _, H, W = current.shape
            diff = current - up[...,:H,:W]
            pyr.append(diff)
            current = down
        return pyr

    def weight_pyramid(x, max_levels=3):
        current = x
        pyr = []
        for level in range(max_levels):
            down = downsample(current)
            pyr.append(current)
            current = down
        return pyr
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    if len(gt.shape) == 3:
        gt = gt.unsqueeze(0)
    
    if gauss_filter is None:
        gauss_filter = gauss_kernel(channels = img.shape[1])

    pyr_logit = lap_pyramid(x = img, kernel = gauss_filter, max_levels = 5)
    pyr_target = lap_pyramid(x = gt, kernel = gauss_filter, max_levels = 5)
    if weight is not None:
        pyr_weight = weight_pyramid(x = weight, max_levels = 5)
        return sum(regression_loss(A[0], A[1], loss_type=loss_type, weight=A[2]) * (2**i) for i, A in enumerate(zip(pyr_logit, pyr_target, pyr_weight)))
    else:
        return sum(regression_loss(A[0], A[1], loss_type=loss_type, weight=None) * (2**i) for i, A in enumerate(zip(pyr_logit, pyr_target)))
    

if __name__ == '__main__':
    print(Lap_Loss(torch.rand((1, 112, 120)).to("cuda"), torch.rand((1,112,120)).to("cuda")))