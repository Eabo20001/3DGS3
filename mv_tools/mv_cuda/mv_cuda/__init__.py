import torch
import mv_cuda._C as backend

def fwdmv(
    depth: torch.Tensor,
    fx,fy,
    prev_w2c: torch.Tensor,
    curr_w2c: torch.Tensor,
):
    H = depth.shape[-2]
    W = depth.shape[-1]
    
    output_coord_mv = torch.zeros(H,W,5, dtype=torch.float32, device=depth.device)
    backend.fwdmv(depth,fx,fy,prev_w2c, curr_w2c, output_coord_mv)
    
    return output_coord_mv[:,:,3:], output_coord_mv[:,:,:3]

def bwdmv(
    depth: torch.Tensor,
    fx,fy,
    prev_w2c: torch.Tensor,
    curr_w2c: torch.Tensor,
):
    H = depth.shape[-2]
    W = depth.shape[-1]
    
    output = torch.zeros(H,W,5, dtype=torch.float32, device=depth.device)
    backend.bwdmv(depth,fx,fy,prev_w2c, curr_w2c, output)
    return output[:,:,3:], output[:,:,:3]
    
def py_grad_transfrom(mv:torch.Tensor, gradxy:torch.Tensor):
    # <- Params ->
    # mv: [2, H, W] 
    # gradxy: [6, H, W]
    
    _, H, W = mv.shape
    C, _,_ = gradxy.shape
    C = C // 2
    
    gradxy = gradxy.view(2, -1, H, W).permute(2,3,0,1).reshape(H*W,2,-1)
    # <- pad the mv ->
    mv = mv.permute(1,2,0).reshape(-1,2) # [H*W, 2]
    buf = torch.zeros(H*W,4, dtype=mv.dtype,device=mv.device) # [H*W,4]
    backend.getinvjacobi_fwdmv(mv, buf, W, H) 
    invjac = buf.view(-1, 2, 2)
    
    transformed_grad = torch.bmm(invjac, gradxy) # [H*W, 2, C]
    transformed_grad = transformed_grad.permute(1,2,0).reshape(2*C,H,W)
    return transformed_grad
    
    
    
    
    