import fill_with_order._C as backend
import torch

def fill_with_order_ret(
    val: torch.Tensor,
    idx: torch.Tensor,
    order: torch.Tensor,
    target_len: int,
    order_init = 1e9
):
    target_val = torch.zeros(target_len, dtype=torch.float32, device="cuda")
    target_order = torch.ones(target_len, dtype=torch.float32, device="cuda") * order_init
    
    if idx.dtype != torch.int32:
        idx = idx.int()
    if val.dtype != torch.float32:
        val = val.float()
    if order.dtype != torch.float32:
        order = order.float()
        
    if len(idx.shape) > 1:
        idx = idx.flatten()
    if len(val.shape) > 1:
        val = val.flatten()
    if len(order.shape) > 1:
        order = order.flatten()    
    backend.fill_with_order(val.contiguous(), idx.contiguous(), order.contiguous(), target_val.contiguous(), target_order.contiguous())
    return target_val, target_order
    
def fill_with_order_inplace(
    target_val:torch.Tensor,
    target_order :torch.Tensor,
    val: torch.Tensor,
    idx: torch.Tensor,
    order: torch.Tensor,
    target_len: int,
    order_init = 1e9
):
    if idx.dtype != torch.int32:
        idx = idx.int()
    if val.dtype != torch.float32:
        val = val.float()
    if order.dtype != torch.float32:
        order = order.float()
        
    if len(idx.shape) > 1:
        idx = idx.flatten()
    if len(val.shape) > 1:
        val = val.flatten()
    if len(order.shape) > 1:
        order = order.flatten()    
    
    backend.fill_with_order(val, idx, order, target_val, target_order)

def warp_image(
    img: torch.Tensor,
    idx: torch.Tensor,
    idy: torch.Tensor,
    order: torch.Tensor,
    order_init = 1e9
):
    target_val = torch.zeros_like(img)
    target_order = torch.ones_like(order) * order_init
    
    if idx.dtype != torch.int32:
        idx = idx.int()
    if idy.dtype != torch.int32:
        idy = idy.int()
          
    backend.warp_image(img, idx, idy, order, target_val, target_order)
    return target_val, target_order


# differentiable operator
class DifferentiableForwardWarping(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        img: torch.Tensor,
        idx: torch.Tensor,
        idy: torch.Tensor,
        order: torch.Tensor,
        order_init = 1e9
    ):
        target_val = torch.zeros_like(img)
        target_order = torch.ones_like(order) * order_init
        
        source_buffer = torch.full_like(order, -1, dtype=torch.int32)
        if idx.dtype != torch.int32:
            idx = idx.int()
        if idy.dtype != torch.int32:
            idy = idy.int()
            
        backend.differentiable_warp_image_f(img, idx, idy, order, target_val, target_order, source_buffer)
        
        ctx.save_for_backward(source_buffer)
        return target_val, target_order

    @staticmethod
    def backward(ctx, g_val, g_order):
        source_buffer = ctx.saved_tensors[0]
        
        ret_val, ret_order = None, None
        if g_val is not None:
            ret_val = torch.zeros_like(g_val)
            backend.differentiable_warp_image_b(g_val, source_buffer, ret_val)
        if g_order is not None:
            ret_order = torch.zeros_like(g_order)
            backend.differentiable_warp_image_b(g_order, source_buffer, ret_order)
        return ret_val, None, None, ret_order

def diff_warp_image(
    img: torch.Tensor,
    idx: torch.Tensor,
    idy: torch.Tensor,
    order: torch.Tensor,
    order_init = 1e9
):
    return DifferentiableForwardWarping.apply(
        img, idx, idy, order, order_init
    )
