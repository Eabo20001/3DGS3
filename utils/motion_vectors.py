import torch
import math
import mv_cuda 

def ForwardMotionVector(pre_cam, pre_depth_map, post_cam):
    w, h = pre_cam.image_width, pre_cam.image_height
    fx = 2 * math.tan(pre_cam.FoVx / 2) / w
    fy = 2 * math.tan(pre_cam.FoVy / 2) / h
    
    i, j = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    i = i.to(pre_depth_map.device)
    j = j.to(pre_depth_map.device)
    dirs = torch.stack([(j -w/2.0) * fx, (i - h/2.0) * fy, torch.ones_like(i)], -1)

    c_coord = pre_depth_map.squeeze(0).unsqueeze(-1) * dirs
    pre_c2w_transform = pre_cam.world_view_transform.inverse()
    w_coord = torch.matmul(c_coord, pre_c2w_transform[:3,:3]) + pre_c2w_transform[3,:3]
    project_coord = torch.matmul(w_coord, post_cam.world_view_transform[:3,:3]) + post_cam.world_view_transform[3,:3]
    depth =  project_coord[:,:,2:]

    j_new = project_coord[...,0]/(fx * project_coord[...,2]) + w/2.0 
    i_new = project_coord[...,1]/(fy * project_coord[...,2]) + h/2.0
    return torch.stack([j_new - j, i_new - i], -1), depth

def CudaForwardMotionVector(pre_cam, pre_depth_map, post_cam):
    w, h = pre_cam.image_width, pre_cam.image_height
    fx = 2 * math.tan(pre_cam.FoVx / 2) / w
    fy = 2 * math.tan(pre_cam.FoVy / 2) / h
    
    prevm = pre_cam.world_view_transform
    currm = post_cam.world_view_transform
    mv, coord = mv_cuda.fwdmv(pre_depth_map, fx,fy,prevm, currm)
    return mv, coord[:,:,2:]

def CudaBackwardMotionVector(pre_cam, post_cam, post_depth_map):
    w, h = pre_cam.image_width, pre_cam.image_height
    fx = 2 * math.tan(pre_cam.FoVx / 2) / w
    fy = 2 * math.tan(pre_cam.FoVy / 2) / h
    
    prevm = pre_cam.world_view_transform
    currm = post_cam.world_view_transform
    mv, coord = mv_cuda.bwdmv(post_depth_map, fx,fy,prevm, currm)
    
    return mv, coord[:,:,2:]    


def ProjBack(pre_cam, pre_depth_map):
    w, h = pre_cam.image_width, pre_cam.image_height
    fx = 2 * math.tan(pre_cam.FoVx / 2) / w
    fy = 2 * math.tan(pre_cam.FoVy / 2) / h
    
    i, j = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    i = i.to(pre_depth_map.device)
    j = j.to(pre_depth_map.device)
    dirs = torch.stack([(j -w/2.0) * fx, (i - h/2.0) * fy, torch.ones_like(i)], -1)

    c_coord = pre_depth_map.squeeze(0).unsqueeze(-1) * dirs
    pre_c2w_transform = pre_cam.world_view_transform.inverse()
    w_coord = torch.matmul(c_coord, pre_c2w_transform[:3,:3]) + pre_c2w_transform[3,:3]
    return w_coord



def BackwardMotionVector(pre_cam, post_cam, post_depth_map):
    w, h = pre_cam.image_width, pre_cam.image_height
    fx = 2 * math.tan(pre_cam.FoVx / 2) / w
    fy = 2 * math.tan(pre_cam.FoVy / 2) / h
    
    i, j = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    i = i.to(post_depth_map.device)
    j = j.to(post_depth_map.device)
    dirs = torch.stack([(j -w/2.0) * fx, (i - h/2.0) * fy, torch.ones_like(i)], -1)

    c_coord = post_depth_map.squeeze(0).unsqueeze(-1) * dirs
    post_c2w_transform = post_cam.world_view_transform.inverse()
    w_coord = torch.matmul(c_coord, post_c2w_transform[:3,:3]) + post_c2w_transform[3,:3]
    project_coord = torch.matmul(w_coord, pre_cam.world_view_transform[:3,:3]) + pre_cam.world_view_transform[3,:3]

    j_new = project_coord[...,0]/(fx * project_coord[...,2]) + w/2.0 
    i_new = project_coord[...,1]/(fy * project_coord[...,2]) + h/2.0
    return torch.stack([j - j_new, i - i_new], -1)
