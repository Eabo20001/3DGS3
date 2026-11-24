# class WarpLayer warps image x based on optical flow flo.
import torch
import torch.nn as nn
from fill_with_order import fill_with_order_inplace,fill_with_order_ret, warp_image, DifferentiableForwardWarping


class ForwardWarp(nn.Module):
    """docstring for WarpLayer"""
    def __init__(self,):
        super(ForwardWarp, self).__init__()
    

    def forward(self, img, flo):
        """
            -img: image (N, C, H, W)
            -flo: optical flow (N, 2, H, W)
            elements of flo is in [0, H] and [0, W] for dx, dy

        """
        

        # (x1, y1)        (x1, y2)
        # +---------------+
        # |                  |
        # |    o(x, y)       |
        # |                  |
        # |                  |
        # |                  |
        # |                  |
        # +---------------+
        # (x2, y1)        (x2, y2)


        N, C, _, _ = img.size()
        
        # translate start-point optical flow to end-point optical flow
        y = flo[:, 0:1 :, :]
        x = flo[:, 1:2, :, :]

        x = x.repeat(1, C, 1, 1)
        y = y.repeat(1, C, 1, 1)
    
        # Four point of square (x1, y1), (x1, y2), (x2, y1), (y2, y2)
        x1 = torch.floor(x)
        x2 = x1 + 1
        y1 = torch.floor(y)
        y2 = y1 + 1

        # firstly, get bilinear weights
        w11, w12, w21, w22 = self.get_bilinear_weights(x, y, x1, x2, y1, y2)

        # secondly, sample each weighted corner 
        img11, o11 = self.sample_one(img, x1, y1, w11)
        img12, o12 = self.sample_one(img, x1, y2, w12)
        img21, o21 = self.sample_one(img, x2, y1, w21)
        img22, o22 = self.sample_one(img, x2, y2, w22)


        imgw = img11 + img12 + img21 + img22
        o = o11 + o12 + o21 + o22

        return imgw, o



    def get_bilinear_weights(self, x, y, x1, x2, y1, y2):
        w11 = torch.exp(-((x - x1)**2 + (y - y1)**2))
        w12 = torch.exp(-((x - x1)**2 + (y - y2)**2))
        w21 = torch.exp(-((x - x2)**2 + (y - y1)**2))
        w22 = torch.exp(-((x - x2)**2 + (y - y2)**2))
        tot = w11 + w12 + w21 + w22
        return w11/tot, w12/tot, w21/tot, w22/tot


    def sample_one(self, img, shiftx, shifty, weight):
        """
        Input:
            -img (N, C, H, W)
            -shiftx, shifty (N, c, H, W)
        """

        N, C, H, W = img.size()
        dev = img.device


        # flatten all (all restored as Tensors)
        flat_shiftx = shiftx.view(-1)
        flat_shifty = shifty.view(-1)
        flat_basex = torch.arange(0, H, requires_grad=False).view(-1, 1)[None, None].to(dev).long().repeat(N, C, 1, W).view(-1)
        flat_basey = torch.arange(0, W, requires_grad=False).view(1, -1)[None, None].to(dev).long().repeat(N, C, H, 1).view(-1)
        flat_weight = weight.flatten()
        flat_img = img.flatten()


        # The corresponding positions in I1
        idxn = torch.arange(0, N, requires_grad=False).view(N, 1, 1, 1).long().to(dev).repeat(1, C, H, W).view(-1)
        idxc = torch.arange(0, C, requires_grad=False).view(1, C, 1, 1).long().to(dev).repeat(N, 1, H, W).view(-1)
        # ttype = flat_basex.type()
        idxx = flat_shiftx.long() + flat_basex
        idxy = flat_shifty.long() + flat_basey
        mask = idxx.ge(0) & idxx.lt(H) & idxy.ge(0) & idxy.lt(W)

        ids = (idxn*C*H*W + idxc*H*W + idxx*W + idxy)

        ids_mask = torch.masked_select(ids, mask).clone().to(dev)

        img_warp = torch.zeros([N*C*H*W, ]).to(dev)
        img_warp.put_(ids_mask, torch.masked_select(flat_img*flat_weight, mask), accumulate=True)

        one_warp = torch.zeros([N*C*H*W, ]).to(dev) 
        one_warp.put_(ids_mask, torch.masked_select(flat_weight, mask), accumulate=True)
        return img_warp.view(N, C, H, W), one_warp.view(N, C, H, W)




class DepthAwareForwardWarp(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, img, flo,point_localcoord:torch.Tensor):
        """
            -img: image (N, C, H, W)
            -point_localcoord:  (N, 3,H, W)
            -flo: optical flow (N, 2, H, W)
        """
        
        y = flo[0:1 :, :]
        x = flo[1:2, :, :]
        
        z_vals = point_localcoord[2:3, :, :]
        
        # Four point of square (x1, y1), (x1, y2), (x2, y1), (y2, y2)
        x1 = torch.floor(x)
        x2 = x1 + 1
        y1 = torch.floor(y)
        y2 = y1 + 1

        # firstly, get bilinear weights
        w11, w12, w21, w22 = self.get_bilinear_weights(x, y, x1, x2, y1, y2)

        # secondly, sample each weighted corner 
        img11, o11 = self.sample_one(img, x1, y1, w11, z_vals)
        img12, o12 = self.sample_one(img, x1, y2, w12, z_vals)
        img21, o21 = self.sample_one(img, x2, y1, w21, z_vals)
        img22, o22 = self.sample_one(img, x2, y2, w22, z_vals)


        imgw = w11 * img11 + w12 * img12 + w21 * img21 + w22 * img22
        o = o11 + o12 + o21 + o22

        return imgw, o



    def get_bilinear_weights(self, x, y, x1, x2, y1, y2):
        w11 = torch.exp(-((x - x1)**2 + (y - y1)**2))
        w12 = torch.exp(-((x - x1)**2 + (y - y2)**2))
        w21 = torch.exp(-((x - x2)**2 + (y - y1)**2))
        w22 = torch.exp(-((x - x2)**2 + (y - y2)**2))
        tot = w11 + w12 + w21 + w22
        return w11/tot, w12/tot, w21/tot, w22/tot


    def sample_one(self, img, shiftx, shifty, weight, z_vals:torch.Tensor):
        """
        Input:
            -img (C, H, W)
            -shiftx, shifty (1, H, W)
            -z_vals (1, H, W)
        """

        img_warp, img_warp_order = warp_image(img, shiftx, shifty, z_vals)
        one_warp, one_warp_depth = warp_image(weight, shiftx, shifty, z_vals)
        
        return img_warp, one_warp.repeat(3,1,1)

    


class SimpleDepthAwareForwardWarp(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, img:torch.Tensor, flo:torch.Tensor, depth:torch.Tensor):
        """
            -img: image (C, H, W)
            -point_localcoord:  (3,H, W)
            -flo: optical flow (2, H, W)
        """
        
        y = flo[0:1 :, :]
        x = flo[1:2, :, :]
        
        # Four point of square (x1, y1), (x1, y2), (x2, y1), (y2, y2)
        x1 = torch.floor(x)
        x2 = x1 + 1
        y1 = torch.floor(y)
        y2 = y1 + 1

        # firstly, get bilinear weights
        w11, w12, w21, w22 = self.get_bilinear_weights(x, y, x1, x2, y1, y2)

        # secondly, sample each weighted corner 
        img11, o11 = self.sample_one(img, x1, y1, torch.ones_like(w11), depth)

        return img11, o11



    def get_bilinear_weights(self, x, y, x1, x2, y1, y2):
        w11 = torch.exp(-((x - x1)**2 + (y - y1)**2))
        w12 = torch.exp(-((x - x1)**2 + (y - y2)**2))
        w21 = torch.exp(-((x - x2)**2 + (y - y1)**2))
        w22 = torch.exp(-((x - x2)**2 + (y - y2)**2))
        tot = w11 + w12 + w21 + w22
        return w11/tot, w12/tot, w21/tot, w22/tot


    def sample_one(self, img, shiftx, shifty, weight, z_vals:torch.Tensor):
        """
        Input:
            -img (C, H, W)
            -shiftx, shifty (1, H, W)
            -z_vals (1, H, W)
        """

        img_warp, depth_warp = warp_image(torch.cat([img, weight], dim=0), shiftx, shifty, z_vals)
        
        return img_warp, depth_warp

    
def depth_aware_forward_warp(img:torch.Tensor, flo:torch.Tensor, depth:torch.Tensor):
    y = flo[0:1 :, :]
    x = flo[1:2, :, :]
    
    x1 = torch.floor(x)
    y1 = torch.floor(y)
    img_warp, depth_warp = warp_image(torch.cat([img, torch.ones_like(depth)], dim=0),x1, y1, depth)
    return img_warp, depth_warp


class DifferentiableSimpleDepthAwareForwardWarp(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, img:torch.Tensor, flo:torch.Tensor, depth:torch.Tensor):
        y = flo[0:1 :, :]
        x = flo[1:2, :, :]
        
        x1 = torch.floor(x)
        y1 = torch.floor(y)
        img_warp, depth_warp = DifferentiableForwardWarping.apply(torch.cat([img, torch.ones_like(depth)], dim=0),x1, y1, depth)
        return img_warp, depth_warp

if __name__ == "__main__":
    h, w = 10, 10
    j, i = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    
    x = i.float().unsqueeze(0).cuda()
    x.requires_grad = True
    print(x, x.shape)
    mv = torch.zeros(2,h,w).float().cuda()
    mv[0, :, 0::2] = 1
    # print(mv)
    fw = DifferentiableSimpleDepthAwareForwardWarp()
    x_warp, depth_warp = fw.forward(x, mv, torch.ones(1,h,w).cuda())
    # print(.shape, depth_warp.shape)
    print(x_warp[:-1, :, :])
    loss = torch.sum(x_warp**2)
    loss.backward()
    print(x.grad)
