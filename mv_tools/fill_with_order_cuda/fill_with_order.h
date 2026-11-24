#pragma once
#include <torch/extension.h>

void FillWithOrder(
    int num_ori,int num_fill,       // number of values to fill
    const float * original_value,   // values to fill
    const int * to_fill_indices,    // respective index of filled_value
    const float * order,            // order (smaller)
    float * filled_value,           // output buffer (filled value)
    float * filled_order            // output buffer (filled order)
);

void WarpImage(
    int C,
    int H,
    int W,
    const torch::Tensor & original_value,   // values to fill
    const torch::Tensor & dx,    // respective index x of filled_value
    const torch::Tensor & dy,    // respective index y of filled_value
    const torch::Tensor & order,            // order (smaller)
    torch::Tensor & filled_value,           // output buffer (filled value)
    torch::Tensor & filled_order            // output buffer (filled order)
);

void GWarpImage_F(
    int C,
    int H,
    int W,
    const float * original_value,   // values to fill
    const int * dx,    // respective index x of filled_value
    const int * dy,    // respective index y of filled_value
    const float * order,            // order (smaller)
    float * filled_value,           // output buffer (filled value)
    float * filled_order,            // output buffer (filled order)
    int * source_buffer            // output buffer (filled order)
);
void GWarpImage_B(
    int C,
    int H,
    int W,
    const float * grad_in,   // values to fill
    const int * source_buffer     ,       // order (smaller)
    float * output           // output buffer (filled value)
);