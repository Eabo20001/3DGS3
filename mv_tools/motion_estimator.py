import torch
import torch.nn  as nn

class EQVIEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        
    
    def forward(self, mv2neg2:torch.Tensor, mv2neg1:torch.Tensor, mv2pos1:torch.Tensor):
        # input frame has shape: [B, Dim]
        A = torch.tensor(
            [[-2,2],
             [-1,0.5],
             [1,0.5]], device=mv2neg2.device, dtype=mv2neg2.dtype
        ).view(1, 3, 2)
        
        inv_AtA = torch.bmm(A.transpose(1,2), A).inverse()
        stacked = torch.stack([mv2neg2, mv2neg1, mv2pos1], dim=1) # [B, 3, Dim]
        Atb = torch.bmm(A.transpose(1,2), stacked) 
        sol = torch.bmm(inv_AtA , Atb) # [B, 2, Dim] Row0 is V Row1 is A
        
        return sol[:,0,:], sol[:,1,:]
    def forward_abs_pos(self, pos_neg2, pos_neg1, pos_0, pos_pos1):
        return self.forward(pos_neg2 - pos_0, pos_neg1 - pos_0, pos_pos1 - pos_0)