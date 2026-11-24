/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "mv_cuda.h"
#include <iostream>

void WrappedForwardMv(
    const torch::Tensor &depth_map,
    float fx, float fy,
    const torch::Tensor &prev,
    const torch::Tensor &curr,
    torch::Tensor &output
)
{
  size_t h = depth_map.size(-2);
  size_t w = depth_map.size(-1);
  ForwardMv(w, h, fx, fy, prev.contiguous().data_ptr<float>(), curr.contiguous().data_ptr<float>(), depth_map.contiguous().data_ptr<float>(), output.contiguous().data_ptr<float>());
}

void WrappedBackwardMv(
    const torch::Tensor &depth_map,
    float fx, float fy,
    const torch::Tensor &prev,
    const torch::Tensor &curr,
    torch::Tensor &output)
{
  size_t h = depth_map.size(-2);
  size_t w = depth_map.size(-1);
  BackwardMv(w, h, fx, fy, prev.contiguous().data_ptr<float>(), curr.contiguous().data_ptr<float>(), depth_map.contiguous().data_ptr<float>(), output.contiguous().data_ptr<float>());
}

void WarppedGetInvJacobi_Fwdmv(
    const torch::Tensor &mv,
    torch::Tensor &output,
    size_t w, size_t h){
  GetInvJacobi_Fwdmv(
    w, h, mv.contiguous().data_ptr<float>(), output.contiguous().data_ptr<float>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("fwdmv", &WrappedForwardMv);
  m.def("bwdmv", &WrappedBackwardMv);
  m.def("getinvjacobi_fwdmv", &WarppedGetInvJacobi_Fwdmv);
}
