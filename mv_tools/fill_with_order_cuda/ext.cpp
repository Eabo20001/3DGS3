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
#include "fill_with_order.h"
#include <iostream>

void FillWithOrderAPI(
    const torch::Tensor &original_value,
    const torch::Tensor &original_tofill_idx,
    const torch::Tensor &original_order,
    torch::Tensor &filled_value,
    torch::Tensor &filled_order){
    int P = original_value.size(0);
    int Q = filled_value.size(0);
    FillWithOrder(P, Q, original_value.contiguous().data_ptr<float>(), original_tofill_idx.contiguous().data_ptr<int>(), original_order.contiguous().data_ptr<float>(), filled_value.contiguous().data_ptr<float>(), filled_order.contiguous().data_ptr<float>());
}

void WarpImageAPI(
    const torch::Tensor &image,
    const torch::Tensor &dx,
    const torch::Tensor &dy,
    const torch::Tensor &original_order,
    torch::Tensor &filled_value,
    torch::Tensor &filled_order){
    int C = image.size(0);
    int H = image.size(1);
    int W = image.size(2);
    WarpImage(C, H, W, image, dx, dy, original_order, filled_value, filled_order);
}



void GWarpImageFAPI(
    const torch::Tensor &image,
    const torch::Tensor &dx,
    const torch::Tensor &dy,
    const torch::Tensor &original_order,
    torch::Tensor &filled_value,
    torch::Tensor &filled_order,
    torch::Tensor &source_buffer){
    int C = image.size(-3);
    int H = image.size(-2);
    int W = image.size(-1);
    GWarpImage_F(C, H, W, image.contiguous().data_ptr<float>(), dx.contiguous().data_ptr<int>(), dy.contiguous().data_ptr<int>(), original_order.contiguous().data_ptr<float>(), filled_value.contiguous().data_ptr<float>(), filled_order.contiguous().data_ptr<float>(), source_buffer.contiguous().data_ptr<int>());
}

void GWarpImageBAPI(
  const torch::Tensor &g_in,
  const torch::Tensor &source_buffer,
  torch::Tensor &output){
  int C = g_in.size(-3);
  int H = g_in.size(-2);
  int W = g_in.size(-1);

  GWarpImage_B(C, H, W, g_in.contiguous().data_ptr<float>(),source_buffer.contiguous().data_ptr<int>(), output.contiguous().data_ptr<float>());

}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("fill_with_order", &FillWithOrderAPI);
  m.def("warp_image"     , &WarpImageAPI    );
  m.def("differentiable_warp_image_f"     , &GWarpImageFAPI    );
  m.def("differentiable_warp_image_b"     , &GWarpImageBAPI    );
}
